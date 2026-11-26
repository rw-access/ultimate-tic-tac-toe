"""A symmetry-equivariant policy/value network over winning lines and directed moves."""

import torch
from torch import nn
from torch.nn import functional as F


class LineStudent(nn.Module):
    """Each playable cell (b, c) is an edge from board b to the next board c."""

    def __init__(self):
        super().__init__()
        lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
        incidence = torch.zeros(9, 8)
        for line, squares in enumerate(lines):
            incidence[list(squares), line] = 1
        self.register_buffer("incidence", incidence)
        self.register_buffer("degree", incidence.sum(1))
        self.register_buffer("diagonal", torch.eye(9))

        self.cell_mlp = nn.Sequential(nn.Linear(12, 48), nn.ReLU(), nn.Linear(48, 48), nn.ReLU())
        self.board_input = nn.Linear(2 * 48 + 12, 160)
        self.line_mlp = nn.Sequential(nn.Linear(160 + 3, 128), nn.ReLU(), nn.Linear(128, 160))
        self.board_mix = nn.Linear(160, 160)

        self.edge_input = nn.Linear(48 + 8, 80)
        self.edge_source = nn.Linear(160, 80)
        self.edge_destination = nn.Linear(160, 80)
        self.edge_mix = nn.Linear(80, 80)
        self.board_update = nn.Sequential(nn.Linear(160 + 2 * 80, 160), nn.ReLU(), nn.Linear(160, 160))

        self.policy_source = nn.Linear(160, 80)
        self.policy_destination = nn.Linear(160, 80)
        self.policy_edge = nn.Linear(80, 80)
        self.policy_head = nn.Sequential(nn.Linear(80, 48), nn.ReLU(), nn.Linear(48, 1))
        self.value_head = nn.Sequential(nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 64),
                                        nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def opportunities(self, marks, blockers, available):
        """Per square: complete a line, create a two-in-row, or keep a line open."""
        unblocked = F.relu(1 - blockers)
        patterns = torch.stack((F.relu(marks - 1) * unblocked,
                                F.relu(1 - (marks - 1).abs()) * unblocked,
                                unblocked), dim=-2)
        counts = patterns @ self.incidence.T
        return (counts[..., 0, :].clamp_max(1) * available,
                counts[..., 1, :] * available / 4, counts[..., 2, :] / 4)

    def encode(self, state):
        own = state[:, :81].reshape(-1, 9, 9)
        other = state[:, 81:162].reshape(-1, 9, 9)
        legal = state[:, 162:243].reshape(-1, 9, 9)
        won, lost, drawn = state[:, 243:252], state[:, 252:261], state[:, 261:270]
        open_board = 1 - won - lost - drawn
        available = (1 - own - other) * open_board.unsqueeze(2)
        own_lines, other_lines = own @ self.incidence, other @ self.incidence
        win, create, own_open = self.opportunities(own_lines, other_lines, available)
        block, prevent, other_open = self.opportunities(other_lines, own_lines, available)
        degree = self.degree / 4
        cell = torch.stack((own, other, available, legal, win, block, create, prevent,
                            own_open, other_open, degree.expand_as(own),
                            self.diagonal.expand_as(own)), dim=-1)

        won_lines, lost_lines, draws = won @ self.incidence, lost @ self.incidence, drawn @ self.incidence
        # A drawn board blocks a global winning line for both players.
        meta_win = (F.relu(won_lines - 1) * F.relu(1 - lost_lines - draws)) @ self.incidence.T
        meta_loss = (F.relu(lost_lines - 1) * F.relu(1 - won_lines - draws)) @ self.incidence.T
        meta_win = meta_win.clamp_max(1) * open_board
        meta_loss = meta_loss.clamp_max(1) * open_board
        forced = (state[:, 270:279] + state[:, 279:280]) * open_board
        threats = block.sum(2)
        board = torch.stack((won, lost, drawn, open_board, forced, own.mean(2), other.mean(2),
                             win.sum(2) / 9, threats / 9, meta_win, meta_loss,
                             degree.expand_as(won)), dim=-1)
        lines = torch.stack((won_lines, lost_lines, draws), dim=-1) / 3

        # If b == c, this move may close its own destination or occupy its last
        # threatening square. Empty squares on a previously closed board never count.
        closes = (win + F.relu(2 - available.sum(2, keepdim=True)) * available).clamp_max(1)
        target_open = open_board.unsqueeze(1) * (1 - self.diagonal * closes)
        free_reply = (1 - target_open) * available
        forced_threat = (threats.unsqueeze(1) - self.diagonal * block).clamp(0, 1) * target_open
        forced_loss = forced_threat * meta_loss.unsqueeze(1)
        old_danger = meta_loss * threats.clamp_max(1)
        other_danger = (old_danger.sum(1, keepdim=True) - old_danger).unsqueeze(2)
        remaining = (threats.unsqueeze(2) - block).clamp(0, 1) * (1 - closes)
        free_loss = free_reply * (other_danger + meta_loss.unsqueeze(2) * remaining).clamp_max(1)
        take_game = win * meta_win.unsqueeze(2)
        routing = torch.stack((take_game, block * meta_loss.unsqueeze(2), closes, free_reply,
                               forced_threat, forced_loss, free_loss,
                               (forced_loss + free_loss) * (1 - take_game)), dim=-1)
        return cell, board, lines, routing, available

    def forward(self, state):
        cell, board, lines, routing, available = self.encode(state)
        cell = self.cell_mlp(cell)
        board = F.relu(self.board_input(torch.cat((cell.mean(2), cell.amax(2), board), dim=-1)))
        # The same MLP acts on each of the eight lines. Sum its messages back
        # to their member squares with the corner/edge/center degree correction.
        line_context = (self.incidence.T @ board) / 3
        line_message = self.line_mlp(torch.cat((line_context, lines), dim=-1))
        context = (self.incidence @ line_message) / self.degree.unsqueeze(1)
        board = F.relu(board + self.board_mix(context))

        edge = self.edge_input(torch.cat((cell, routing), dim=-1))
        edge = F.relu(edge + self.edge_source(board).unsqueeze(2) + self.edge_destination(board).unsqueeze(1))
        edge = F.relu(edge + self.edge_mix(edge))
        playable = available.unsqueeze(-1)
        messages = edge * playable
        outgoing = messages.sum(2) / playable.sum(2).clamp_min(1)
        incoming = messages.sum(1) / playable.sum(1).clamp_min(1)
        board = F.relu(board + self.board_update(torch.cat((board, outgoing, incoming), dim=-1)))

        action = self.policy_edge(edge) + self.policy_source(board).unsqueeze(2)
        action = F.relu(action + self.policy_destination(board).unsqueeze(1))
        logits = self.policy_head(action).reshape(-1, 81)
        value = self.value_head(torch.cat((board.mean(1), board.amax(1)), dim=-1))
        return logits, value
