"""Train a shared mini-board encoder and compile it to an int8 lookup table."""

import math

import torch
from torch import nn
from torch.nn import functional as F


PATTERNS = 3 ** 9
LEARNED = 23
CHANNELS = LEARNED + 9


def incidence():
    matrix = torch.zeros(9, 8)
    lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
             (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
    for i, line in enumerate(lines):
        matrix[list(line), i] = 1
    return matrix


def winning_cells(own_lines, other_lines, available, membership):
    unblocked_twos = (own_lines - 1).clamp(0, 1) * (1 - other_lines).clamp_min(0)
    return (unblocked_twos @ membership.T).clamp_max(1) * available


class PatternEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lines", incidence())
        self.network = nn.Sequential(nn.Linear(52, 128), nn.ReLU(), nn.Linear(128, 128),
                                     nn.ReLU(), nn.Linear(128, LEARNED), nn.Tanh())

    def forward(self, own, other):
        # Only marks enter this encoder. Forced moves, closed boards and the
        # macro state can change while the mini-board pattern stays identical.
        a, b = own @ self.lines, other @ self.lines
        empty = 1 - own - other
        take = winning_cells(a, b, empty, self.lines)
        block = winning_cells(b, a, empty, self.lines)
        features = torch.cat((own, other, a / 3, b / 3, take, block), dim=-1)
        return torch.cat((self.network(features), take + 2 * block), dim=-1)


class PatternLookup(nn.Module):
    def __init__(self, table, scale):
        super().__init__()
        self.register_buffer("table", table)
        self.register_buffer("scale", scale)
        self.register_buffer("powers", (3 ** torch.arange(9)).float().reshape(9, 1))

    def forward(self, own, other):
        # All integers through 19,682 are exactly representable in float32.
        row = ((own + 2 * other) @ self.powers).squeeze(-1).long()
        return self.table[row].float() * self.scale


class BoardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lines", incidence())
        self.register_buffer("diagonal", torch.eye(9))
        self.board = nn.Linear(CHANNELS + 18 + 8, 64)
        self.context = nn.Linear(9 * 64, 112)
        self.context_mix = nn.Linear(112, 112)
        self.slots = nn.Linear(112, 9 * 24)
        self.refine = nn.Linear(64 + 24, 64)
        self.source = nn.Linear(64, 56)
        self.destination = nn.Linear(64, 56)
        self.move = nn.Linear(11, 56)
        self.positions = nn.Parameter(torch.empty(1, 9, 9, 56))
        nn.init.normal_(self.positions, std=0.02)
        self.policy = nn.Sequential(nn.Linear(56, 48), nn.ReLU(), nn.Linear(48, 1))
        self.value = nn.Sequential(nn.Linear(112 + 64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, state, own, other, pattern):
        legal = state[:, 162:243].reshape(-1, 9, 9)
        won, lost, drawn = state[:, 243:252], state[:, 252:261], state[:, 261:270]
        opened = 1 - won - lost - drawn
        available = (1 - own - other) * opened.unsqueeze(2)
        # One byte per cell encodes all four exact (can win, can block) cases.
        code = pattern[:, :, LEARNED:]
        can_block = (code - 1).clamp(0, 1)
        take = (code - 2 * can_block) * opened.unsqueeze(2)
        block = can_block * opened.unsqueeze(2)
        won_lines, lost_lines, draw_lines = won @ self.lines, lost @ self.lines, drawn @ self.lines
        meta_win = winning_cells(won_lines, lost_lines + draw_lines, opened, self.lines)
        meta_loss = winning_cells(lost_lines, won_lines + draw_lines, opened, self.lines)
        forced = opened * (state[:, 270:279] + state[:, 279:280])

        last_empty = (2 - available.sum(2, keepdim=True)).clamp(0, 1) * available
        closes = (last_empty + take).clamp_max(1)
        target_open = opened.unsqueeze(1) * (1 - self.diagonal * closes)
        free = available * (1 - target_open)
        threats = block.sum(2)
        reply = target_open * (threats.unsqueeze(1) - self.diagonal * block).clamp(0, 1)
        forced_loss = reply * meta_loss.unsqueeze(1)
        danger = meta_loss * threats.clamp_max(1)
        other_danger = (danger.sum(1, keepdim=True) - danger).unsqueeze(2)
        remaining = (threats.unsqueeze(2) - block).clamp(0, 1) * (1 - closes)
        free_loss = free * (other_danger + meta_loss.unsqueeze(2) * remaining).clamp_max(1)
        global_take = take * meta_win.unsqueeze(2)
        losing_reply = (free_loss + forced_loss) * (1 - global_take) * available

        status = torch.stack((won, lost, drawn, forced, meta_win, meta_loss,
                              legal.mean(2), losing_reply.mean(2)), dim=-1)
        boards = F.relu(self.board(torch.cat((pattern, own, other, status), dim=-1)))
        context = F.relu(self.context(boards.flatten(1)))
        context = F.relu(context + self.context_mix(context))
        slots = self.slots(context).reshape(-1, 9, 24)
        boards = F.relu(boards + self.refine(torch.cat((boards, slots), dim=-1)))

        moves = torch.stack((available, legal, take, block, global_take, block * meta_loss.unsqueeze(2),
                             closes, free, reply, losing_reply, pattern[:, :, :9]), dim=-1)
        dest = self.destination(boards)
        redirected = free.unsqueeze(-1)
        next_board = ((1 - redirected) * dest.unsqueeze(1)
                      + redirected * dest.mean(1, keepdim=True).unsqueeze(1))
        action = F.relu(self.source(boards).unsqueeze(2) + next_board + self.move(moves) + self.positions)
        value = self.value(torch.cat((context, boards.amax(1)), dim=1))
        return self.policy(action).reshape(-1, 81), value


class BoardTable(nn.Module):
    def __init__(self, encoder=None, head=None):
        super().__init__()
        self.encoder = PatternEncoder() if encoder is None else encoder
        self.head = BoardHead() if head is None else head

    def forward(self, state):
        own = state[:, :81].reshape(-1, 9, 9)
        other = state[:, 81:162].reshape(-1, 9, 9)
        return self.head(state, own, other, self.encoder(own, other))

    def estimated_export_bytes(self):
        head_bytes = sum(p.numel() * p.element_size() for p in self.head.parameters())
        return PATTERNS * CHANNELS + head_bytes + 20 * 1024


@torch.inference_mode()
def compile_table(model, states):
    """Enumerate all patterns, check float lookup equivalence, and measure int8 drift."""
    powers = 3 ** torch.arange(9)
    parts = []
    for ids in torch.arange(PATTERNS).split(1024):
        trits = (ids[:, None] // powers) % 3
        parts.append(model.encoder((trits == 1).float(), (trits == 2).float()))
    table = torch.cat(parts)
    scales = table.abs().amax(0).clamp_min(1e-8) / 127
    scales[LEARNED:] = 1  # Tactical codes are exact integers from zero to three.
    packed = (table / scales).round().clamp(-127, 127).to(torch.int8)
    baseline = BoardTable(PatternLookup(table, torch.ones(CHANNELS)), model.head).eval()
    candidate = BoardTable(PatternLookup(packed, scales), model.head).eval()
    kl_sum = value_sum = value_max = 0.0
    live = agreement = 0
    for batch in states.split(128):
        expected = model(batch)
        for raw, implicit in zip(baseline(batch), expected):
            torch.testing.assert_close(raw, implicit, rtol=1e-4, atol=2e-4)
        logits, values = candidate(batch)
        diff = (values - expected[1]).abs()
        value_max = max(value_max, diff.max().item())
        value_sum += diff.sum().item()
        legal = batch[:, 162:243].bool()
        valid = legal.any(1)
        if valid.any():
            a = expected[0][valid].masked_fill(~legal[valid], -1e9).log_softmax(1)
            b = logits[valid].masked_fill(~legal[valid], -1e9).log_softmax(1)
            kl_sum += (a.exp() * (a - b)).sum().item()
            agreement += (a.argmax(1) == b.argmax(1)).sum().item()
            live += len(a)
    if not live:
        raise ValueError("INT8 validation requires live positions")
    metrics = {"positions": len(states), "livePositions": live, "topLegalAgreement": agreement / live,
               "legalPolicyKL": kl_sum / live, "valueMAE": value_sum / len(states),
               "valueMaxError": value_max}
    if (not all(math.isfinite(v) for v in metrics.values())
            or metrics["legalPolicyKL"] > 0.002 or metrics["valueMAE"] > 0.01 or value_max > 0.08
            or live >= 64 and metrics["topLegalAgreement"] < 0.95):
        raise ValueError(f"Quantized board-table predictions exceed drift limits: {metrics}")
    return candidate, metrics


def verify_cached_graph(path):
    import onnx

    graph = onnx.load(path).graph
    rows = [t for t in graph.initializer if t.data_type == onnx.TensorProto.INT8
            and tuple(t.dims) == (PATTERNS, CHANNELS) and len(t.raw_data) == PATTERNS * CHANNELS]
    if len(rows) != 1:
        raise ValueError("Expected one dense int8 mini-board table")
    value = rows[0].name
    for kind in ("Gather", "Cast", "Mul"):
        consumers = [n for n in graph.node if value in n.input and n.op_type == kind]
        if len(consumers) != 1:
            raise ValueError(f"Expected {kind} in the mini-board lookup path")
        value = consumers[0].output[0]
