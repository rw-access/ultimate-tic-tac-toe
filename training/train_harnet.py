#!/usr/bin/env python3
"""Train HARNet (Hierarchical Action-Routing Network) on RTX 5080."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_50m import _symmetry_maps, augment_batch
from train_hybrid_teacher import load_hybrid_data
from harnet import build_harnet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="preview", choices=["preview", "teacher50m", "challenger20m", "student1_5m"])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--utttai-weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    model = build_harnet(args.tier).to(device)
    num_params, size_mb = model.count_params()
    print(f"HARNet [{args.tier}] initialized: {num_params:,} parameters ({size_mb:.2f} MB ONNX)")

    train_data, val_data = load_hybrid_data(args.data, seed=args.seed)
    train_x, train_mp, train_mv, train_up, train_uv = train_data
    val_x, val_mp, val_mv, val_up, val_uv = val_data

    w_u = args.utttai_weight
    w_m = 1.0 - w_u
    print(f"Hybrid Blend: {w_m:.2f} MCTS 1024-Sim + {w_u:.2f} uttt.ai Soft Targets")

    train_p = w_m * train_mp + w_u * train_up
    train_v = w_m * train_mv + w_u * train_uv
    val_p = w_m * val_mp + w_u * val_up
    val_v = w_m * val_mv + w_u * val_uv

    print(f"Train: {len(train_x):,} positions | Val: {len(val_x):,} positions")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = (len(train_x) // args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-5)

    maps = _symmetry_maps()
    aug_rng = np.random.default_rng(args.seed)

    val_sub = min(25000, len(val_x))
    val_x_t = torch.from_numpy(val_x[:val_sub]).to(device)
    val_p_t = torch.from_numpy(val_p[:val_sub]).to(device)
    val_v_t = torch.from_numpy(val_v[:val_sub]).to(device)

    best_val_loss = float("inf")
    best_weights = None

    print(f"Starting {args.epochs} HARNet [{args.tier}] Training Epochs ({total_steps:,} updates)...")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(train_x))
        epoch_p_loss, epoch_v_loss = 0.0, 0.0
        n_batches = 0

        for start_idx in range(0, len(train_x) - args.batch_size + 1, args.batch_size):
            batch_idx = perm[start_idx : start_idx + args.batch_size]
            bx = train_x[batch_idx]
            bp = train_p[batch_idx]
            bv = train_v[batch_idx]

            syms = aug_rng.integers(0, 8, size=len(batch_idx))
            bx, bp = augment_batch(bx, bp, syms, maps)

            x = torch.from_numpy(bx).to(device)
            target_p = torch.from_numpy(bp).to(device)
            target_v = torch.from_numpy(bv).to(device)

            logits, pred_v = model(x)
            logits = logits.masked_fill(x[:, 162:243] < 0.5, -1e9)
            p_loss = -(target_p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = F.mse_loss(pred_v.squeeze(-1), target_v)
            loss = p_loss + 0.5 * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_p_loss += p_loss.item()
            epoch_v_loss += v_loss.item()
            n_batches += 1
            step += 1

            if step % 500 == 0:
                elapsed = time.time() - start_time
                ups = step / max(elapsed, 1e-3)
                print(f"  Step {step:5d}/{total_steps} | P-Loss: {p_loss.item():.4f} | V-Loss: {v_loss.item():.4f} | {ups:.1f} up/s")

        model.eval()
        with torch.no_grad():
            v_logits, v_pred_v = model(val_x_t)
            v_logits = v_logits.masked_fill(val_x_t[:, 162:243] < 0.5, -1e9)
            v_p_loss = -(val_p_t * F.log_softmax(v_logits, dim=1)).sum(dim=1).mean().item()
            v_v_loss = F.mse_loss(v_pred_v.squeeze(-1), val_v_t).item()
            val_total = v_p_loss + 0.5 * v_v_loss

        avg_p = epoch_p_loss / max(n_batches, 1)
        avg_v = epoch_v_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train P: {avg_p:.4f} V: {avg_v:.4f} | Val P: {v_p_loss:.4f} V: {v_v_loss:.4f} Total: {val_total:.4f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Training completed in {time.time() - start_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Export best ONNX and PyTorch weights
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_weights)
    pt_path = out_path.with_suffix(".pt")
    torch.save(best_weights, pt_path)
    print(f"Saved PyTorch weights to {pt_path}")

    model.eval()
    model_cpu = model.to("cpu")
    dummy = torch.randn(1, 280)

    torch.onnx.export(
        model_cpu, dummy, str(out_path),
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={"state": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=14,
        dynamo=False
    )
    print(f"Saved ONNX model to {out_path} ({out_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
