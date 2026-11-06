import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class HARNetResBlock(nn.Module):
    def __init__(self, channels: int, depthwise: bool = False):
        super().__init__()
        if depthwise:
            self.conv1 = DepthwiseSeparableConv2d(channels, channels, 3, padding=1)
            self.conv2 = DepthwiseSeparableConv2d(channels, channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(residual + out)

class HARNet(nn.Module):
    """
    Hierarchical Action-Routing Network for Ultimate Tic-Tac-Toe.
    Micro-board encoder shared across all 9 sub-boards,
    Macro-board spatial reasoning over 3x3 meta-grid,
    Action-routing skip connection linking move (b, c) directly to target board c.
    """
    def __init__(
        self,
        micro_channels: int = 64,
        num_micro_blocks: int = 2,
        macro_channels: int = 192,
        num_macro_blocks: int = 4,
        action_hidden: int = 256,
        depthwise: bool = False
    ):
        super().__init__()
        self.micro_channels = micro_channels
        self.macro_channels = macro_channels
        
        # 1. Micro-Board Shared ConvNet (shared across 9 sub-boards)
        self.micro_in = nn.Conv2d(3, micro_channels, 3, padding=1)
        self.micro_bn = nn.BatchNorm2d(micro_channels)
        self.micro_blocks = nn.ModuleList([HARNetResBlock(micro_channels, depthwise=depthwise) for _ in range(num_micro_blocks)])
        
        # 2. Macro-Board Reasoning
        macro_in_dim = micro_channels + 4  # pooled micro + own_won + opp_won + drawn + forced
        self.macro_proj = nn.Linear(macro_in_dim, macro_channels)
        self.macro_blocks = nn.ModuleList([HARNetResBlock(macro_channels, depthwise=depthwise) for _ in range(num_macro_blocks)])
        
        # 3. Action-Routing Integration Head
        # move (b, c) combines: cell features at (b, c) + origin board b + target board c
        action_in_dim = micro_channels + 2 * macro_channels
        self.policy_mlp = nn.Sequential(
            nn.Linear(action_in_dim, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, action_hidden // 2),
            nn.ReLU(),
            nn.Linear(action_hidden // 2, 1)
        )
        
        # 4. Value Head (evaluates overall macro board state)
        self.value_mlp = nn.Sequential(
            nn.Linear(macro_channels * 9, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, action_hidden // 2),
            nn.ReLU(),
            nn.Linear(action_hidden // 2, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B = x.shape[0]
        
        # Slices from 280-feature encoding:
        own = x[:, 0:81].view(B, 9, 3, 3)
        opp = x[:, 81:162].view(B, 9, 3, 3)
        leg = x[:, 162:243].view(B, 9, 3, 3)
        
        own_won = x[:, 243:252]
        opp_won = x[:, 252:261]
        draw_won = x[:, 261:270]
        
        forced = x[:, 270:280]
        free_move = forced[:, 9:10]
        forced_boards = forced[:, 0:9] + free_move
        
        # Shared micro conv across all 9 sub-boards: [B * 9, 3, 3, 3]
        micro_in = torch.stack([own, opp, leg], dim=2).view(B * 9, 3, 3, 3)
        h_micro = F.relu(self.micro_bn(self.micro_in(micro_in)))
        for block in self.micro_blocks:
            h_micro = block(h_micro)
            
        # Cell features per candidate move: [B, 9, 9, micro_channels]
        cell_feats = h_micro.view(B, 9, self.micro_channels, 9).permute(0, 1, 3, 2)
        
        # Sub-board summary embeddings: [B, 9, micro_channels]
        board_pooled = F.adaptive_avg_pool2d(h_micro, (1, 1)).view(B, 9, self.micro_channels)
        
        # Macro context: [B, 9, micro_channels + 4]
        macro_in = torch.cat([
            board_pooled,
            own_won.unsqueeze(-1),
            opp_won.unsqueeze(-1),
            draw_won.unsqueeze(-1),
            forced_boards.unsqueeze(-1)
        ], dim=-1)
        
        # Macro 3x3 spatial ResBlocks: [B, macro_channels, 3, 3]
        h_macro = self.macro_proj(macro_in).view(B, 3, 3, self.macro_channels).permute(0, 3, 1, 2)
        for block in self.macro_blocks:
            h_macro = block(h_macro)
            
        macro_embed = h_macro.permute(0, 2, 3, 1).reshape(B, 9, self.macro_channels)
        
        # Action Routing: move at (b, c) connects to origin b and target c
        source_embed = macro_embed.unsqueeze(2).expand(-1, -1, 9, -1)  # [B, 9, 9, macro_channels]
        target_embed = macro_embed.unsqueeze(1).expand(-1, 9, -1, -1)  # [B, 9, 9, macro_channels]
        
        action_repr = torch.cat([cell_feats, source_embed, target_embed], dim=-1)
        policy_logits = self.policy_mlp(action_repr).view(B, 81)
        
        val_in = macro_embed.reshape(B, 9 * self.macro_channels)
        value = self.value_mlp(val_in)
        
        if return_features:
            return policy_logits, value, board_pooled, macro_embed
        return policy_logits, value

    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n, n * 4 / (1024 * 1024)

def build_harnet(tier: str):
    """Factory for HARNet tiers."""
    tier = tier.lower()
    if tier == "preview":
        return HARNet(micro_channels=64, num_micro_blocks=2, macro_channels=192, num_macro_blocks=3, action_hidden=256)
    elif tier == "teacher50m":
        # 13,476,674 params (~51.41 MB)
        return HARNet(micro_channels=96, num_micro_blocks=2, macro_channels=384, num_macro_blocks=4, action_hidden=512)
    elif tier == "challenger20m":
        # 4,966,274 params (~18.94 MB) - exact match to uttt.ai (4.99M params)
        return HARNet(micro_channels=64, num_micro_blocks=2, macro_channels=256, num_macro_blocks=3, action_hidden=384)
    elif tier == "student1_5m":
        # 369,042 params (1,476,168 bytes / 1.408 MB) - strictly <= 1,500,000 bytes
        return HARNet(micro_channels=32, num_micro_blocks=1, macro_channels=80, num_macro_blocks=2, action_hidden=112)
    elif tier == "fast_student1_5m":
        # Depthwise-separable convolutions for 1.6x+ CPU leaf evaluation speed in browser
        # Strictly <= 1,500,000 bytes budget (target ~1.46 MB ONNX)
        return HARNet(micro_channels=48, num_micro_blocks=1, macro_channels=128, num_macro_blocks=2, action_hidden=168, depthwise=True)
    else:
        raise ValueError(f"Unknown tier: {tier}")
