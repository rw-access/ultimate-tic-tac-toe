"""A browser policy/value model using geometry-biased attention on nine mini-boards.

The 2-D attention bias is inspired by the UTTT transformer described at
https://github.com/TilenG2/UTTT. This implementation uses a shared local CNN
and conditions each attention edge b -> c on playing cell c of board b.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BoardAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(96)
        self.qkv = nn.Linear(96, 3 * 96)
        self.relative_bias = nn.Parameter(torch.zeros(4, 25))
        self.move_bias = nn.Linear(8, 4, bias=False)
        self.out = nn.Linear(96, 96)
        self.ff = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 192), nn.ReLU(), nn.Linear(192, 96))

    def forward(self, boards, moves, displacement):
        batch = boards.size(0)
        qkv = self.qkv(self.norm(boards)).reshape(batch, 9, 3, 4, 24).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * (24 ** -0.5)
        # Keep all boards visible, including closed boards that affect the global outcome.
        bias = self.relative_bias[:, displacement] + self.move_bias(moves).permute(0, 3, 1, 2)
        weights = torch.softmax((scores + bias).float(), dim=-1).to(v.dtype)
        message = (weights @ v).transpose(1, 2).reshape(batch, 9, 96)
        boards = boards + self.out(message)
        return boards + self.ff(boards)


class RoutingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                 (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
        incidence = torch.zeros(9, 8)
        for index, squares in enumerate(lines):
            incidence[list(squares), index] = 1
        self.register_buffer("incidence", incidence)
        self.register_buffer("diagonal", torch.eye(9))
        i = torch.arange(9)
        row, col = i // 3, i % 3
        displacement = 5 * (row[None, :] - row[:, None] + 2) + (col[None, :] - col[:, None] + 2)
        self.register_buffer("displacement", displacement)

        self.stem = nn.Sequential(nn.Conv2d(6, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU())
        self.local = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48))
        self.token = nn.Linear(9 * 48 + 8, 96)
        self.board_position = nn.Parameter(torch.empty(1, 9, 96))
        self.cell_position = nn.Parameter(torch.empty(1, 1, 9, 96))
        nn.init.normal_(self.board_position, std=0.02)
        nn.init.normal_(self.cell_position, std=0.02)
        self.blocks = nn.ModuleList([BoardAttention(), BoardAttention()])
        self.final_norm = nn.LayerNorm(96)
        self.source = nn.Linear(96, 96)
        self.destination = nn.Linear(96, 96)
        self.cell = nn.Linear(48, 96)
        self.move = nn.Linear(8, 96)
        self.policy = nn.Sequential(nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, 64),
                                    nn.ReLU(), nn.Linear(64, 1))
        self.value = nn.Sequential(nn.Linear(9 * 96, 64), nn.ReLU(), nn.Linear(64, 32),
                                   nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def winning_cells(self, own, opponent, available):
        two = (own @ self.incidence - 1).clamp(0, 1)
        unblocked = (1 - opponent @ self.incidence).clamp_min(0)
        return ((two * unblocked) @ self.incidence.T).clamp_max(1) * available

    def encode(self, state):
        own = state[:, :81].reshape(-1, 9, 9)
        opponent = state[:, 81:162].reshape(-1, 9, 9)
        legal = state[:, 162:243].reshape(-1, 9, 9)
        won, lost, drawn = state[:, 243:252], state[:, 252:261], state[:, 261:270]
        opened = 1 - won - lost - drawn
        available = (1 - own - opponent) * opened.unsqueeze(2)
        take = self.winning_cells(own, opponent, available)
        block = self.winning_cells(opponent, own, available)
        global_win = self.winning_cells(won, lost + drawn, opened)
        global_loss = self.winning_cells(lost, won + drawn, opened)

        # A same-board move can remove a threat or close its own destination.
        last_empty = (2 - available.sum(2, keepdim=True)).clamp(0, 1) * available
        closes = (take + last_empty).clamp_max(1)
        destination_open = opened.unsqueeze(1) * (1 - self.diagonal * closes)
        redirected = available * (1 - destination_open)
        remaining = (block.sum(2).unsqueeze(1) - self.diagonal * block).clamp(0, 1)
        forced_threat = available * destination_open * remaining
        moves = torch.stack((available, legal, take, block, take * global_win.unsqueeze(2),
                             redirected, forced_threat, forced_threat * global_loss.unsqueeze(1)), dim=-1)
        forced = opened * (state[:, 270:279] + state[:, 279:280])
        status = torch.stack((won, lost, drawn, forced, own.mean(2), opponent.mean(2),
                              global_win, global_loss), dim=-1)
        local = torch.stack((own, opponent, available, legal, take, block), dim=2).reshape(-1, 6, 3, 3)
        return local, status, moves

    def forward(self, state):
        local, status, moves = self.encode(state)
        cells = self.stem(local)
        cells = F.relu(cells + self.local(cells)).reshape(-1, 9, 48, 9)
        boards = self.token(torch.cat((cells.flatten(2), status), dim=-1)) + self.board_position
        for layer in self.blocks:
            boards = layer(boards, moves, self.displacement)
        boards = self.final_norm(boards)

        destination = self.destination(boards)
        free_choice = moves[:, :, :, 5:6]
        next_board = ((1 - free_choice) * destination.unsqueeze(1)
                      + free_choice * destination.mean(1, keepdim=True).unsqueeze(1))
        actions = self.cell(cells.transpose(2, 3)) + self.cell_position + self.move(moves)
        actions = F.relu(actions + self.source(boards).unsqueeze(2) + next_board)
        return self.policy(actions).reshape(-1, 81), self.value(boards.flatten(1))
