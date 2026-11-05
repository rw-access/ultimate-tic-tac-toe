"""Knowledge Distillation: Transfer Teacher Knowledge into Compact Browser Student.

Trains StudentNet on soft policy targets from Teacher50MNet and game outcome values,
leveraging 8-fold dihedral symmetries and legal move probability masking.
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.teacher_model import StudentNet, Teacher50MNet

def _board_to_row_col(cell: int) -> Tuple[int, int]:
    return cell // 3, cell % 3

def _cell_from_row_col(r: int, c: int) -> int:
    return r * 3 + c

def _symmetry_maps(device: torch.device):
    """Compute 8 dihedral symmetry transformation maps for 280-feature encoding."""
    maps = []
    for rot in range(4):
        for flip in (False, True):
            f_map = list(range(280))
            p_map = list(range(81))
            for b in range(9):
                br, bc = _board_to_row_col(b)
                for c in range(9):
                    cr, cc = _board_to_row_col(c)
                    r_b, c_b = br, bc
                    r_c, c_c = cr, cc
                    if flip:
                        c_b = 2 - c_b
                        c_c = 2 - c_c
                    for _ in range(rot):
                        r_b, c_b = c_b, 2 - r_b
                        r_c, c_c = c_c, 2 - r_c
                    new_b = _cell_from_row_col(r_b, c_b)
                    new_c = _cell_from_row_col(r_c, c_c)
                    orig_idx = b * 9 + c
                    new_idx = new_b * 9 + new_c
                    f_map[orig_idx] = new_idx
                    f_map[81 + orig_idx] = 81 + new_idx
                    f_map[162 + orig_idx] = 162 + new_idx
                    p_map[orig_idx] = new_idx

            for b in range(9):
                br, bc = _board_to_row_col(b)
                r_b, c_b = br, bc
                if flip:
                    c_b = 2 - c_b
                for _ in range(rot):
                    r_b, c_b = c_b, 2 - r_b
                new_b = _cell_from_row_col(r_b, c_b)
                f_map[243 + b] = 243 + new_b
                f_map[252 + b] = 252 + new_b
                f_map[261 + b] = 261 + new_b
                f_map[270 + b] = 270 + new_b

            maps.append((
                torch.tensor(f_map, dtype=torch.long, device=device),
                torch.tensor(p_map, dtype=torch.long, device=device),
            ))
    return maps

def load_dataset(paths: List[str], max_games_per_file: int = 35000) -> Tuple[np.ndarray, np.ndarray]:
    states, values = [], []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for step in record.get("steps", []):
                    states.append(step["features"])
                    values.append(step["value"])
                count += 1
                if count >= max_games_per_file:
                    break
    states_arr = np.array(states, dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)
    return states_arr, values_arr

def main():
    parser = argparse.ArgumentParser(description="Distill Teacher model into compact student network.")
    parser.add_argument("--teacher", required=True, help="Path to Teacher ONNX model")
    parser.add_argument("--data", nargs="+", required=True, help="Paths to self-play JSONL datasets")
    parser.add_argument("--output", required=True, help="Output path for distilled student ONNX")
    parser.add_argument("--initial-student", default=None, help="Optional initial student model to warm-start")
    parser.add_argument("--hidden-widths", nargs=2, type=int, default=[512, 256], help="Student hidden layer widths")
    parser.add_argument("--epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--temperature", type=float, default=1.5, help="Distillation softmax temperature")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=20261109, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distillation running on: {device}")

    import onnx
    from onnx import numpy_helper

    # Load frozen Teacher
    print(f"Loading Teacher from {args.teacher}...")
    teacher_onnx = onnx.load(args.teacher)
    teacher = Teacher50MNet().to(device)
    t_state = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy()).to(device) for init in teacher_onnx.graph.initializer}
    teacher.load_state_dict(t_state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize Student
    widths = tuple(args.hidden_widths)
    student = StudentNet(hidden_widths=widths).to(device)
    if args.initial_student and Path(args.initial_student).exists():
        try:
            print(f"Warm-starting student from {args.initial_student}...")
            s_onnx = onnx.load(args.initial_student)
            s_state = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy()).to(device) for init in s_onnx.graph.initializer}
            student.load_state_dict(s_state, strict=True)
            print("Student warm-started.")
        except Exception as e:
            print(f"Warm-start skipped: {e}")

    # Load self-play data
    states, values = load_dataset(args.data)
    n = len(states)
    val_size = min(20000, int(n * 0.08))
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[val_size:], perm[:val_size]

    train_states, train_vals = states[train_idx], values[train_idx]
    val_states, val_vals = states[val_idx], values[val_idx]
    print(f"Dataset: {len(train_states):,} train positions | {len(val_states):,} val positions")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    sym_maps = _symmetry_maps(device)
    T = args.temperature

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        student.train()
        p = np.random.permutation(len(train_states))
        epoch_loss = 0.0
        steps = 0

        for start in range(0, len(train_states) - args.batch_size + 1, args.batch_size):
            b_idx = p[start : start + args.batch_size]
            b_x = torch.from_numpy(train_states[b_idx]).to(device)
            b_v = torch.from_numpy(train_vals[b_idx]).to(device)

            sym_idx = np.random.randint(0, 8)
            f_map, _ = sym_maps[sym_idx]
            b_x = b_x[:, f_map]

            with torch.no_grad():
                t_logits, _ = teacher(b_x)
                t_logits = t_logits.masked_fill(b_x[:, 162:243] < 0.5, -1e9)
                t_soft = F.softmax(t_logits / T, dim=-1)

            optimizer.zero_grad(set_to_none=True)
            s_logits, s_v = student(b_x)
            s_logits = s_logits.masked_fill(b_x[:, 162:243] < 0.5, -1e9)
            s_log_soft = F.log_softmax(s_logits / T, dim=-1)

            kl_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (T * T)
            mse_loss = F.mse_loss(s_v.squeeze(-1), b_v)
            loss = kl_loss + mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:2d}/{args.epochs} | Avg Loss: {epoch_loss/max(steps,1):.4f} | Elapsed: {elapsed:.1f}s")

    # Export compact ONNX
    student.eval()
    dummy_input = torch.zeros(1, 280, device="cpu")
    student_cpu = student.to("cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        student_cpu, dummy_input, str(out_path),
        input_names=["state"], output_names=["policy", "value"],
        dynamic_axes={"state": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    size = out_path.stat().st_size
    print(f"Exported distilled model to {out_path} ({size:,} bytes | {size/1e6:.2f} MB)")
    assert size <= 1500000, f"Exceeded size limit: {size} bytes"

if __name__ == "__main__":
    main()
