#!/usr/bin/env python3
"""Distill 50MB HARNet Teacher into 20MB Challenger and 1.46MB Browser Student."""
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
    parser.add_argument("--teacher-weights", required=True, help="Path to PyTorch .pt state dict of teacher")
    parser.add_argument("--student-tier", default="challenger20m", choices=["challenger20m", "student1_5m", "fast_student1_5m"])
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight on teacher soft distillation")
    parser.add_argument("--feature-weight", type=float, default=0.0, help="Weight for intermediate feature alignment")
    parser.add_argument("--utttai-weight", type=float, default=0.5, help="Weight for uttt.ai soft targets vs MCTS")
    parser.add_argument("--swa-start-epoch", type=int, default=0, help="Epoch to start SWA averaging (0 to disable)")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Load Teacher
    print(f"Loading 50MB HARNet Teacher from {args.teacher_weights}...")
    teacher = build_harnet("teacher50m").to(device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_teach, mb_teach = teacher.count_params()
    print(f"Teacher loaded: {n_teach:,} params ({mb_teach:.2f} MB)")

    # Init Student
    student = build_harnet(args.student_tier).to(device)
    n_stud, mb_stud = student.count_params()
    print(f"Student [{args.student_tier}] initialized: {n_stud:,} params ({mb_stud:.2f} MB, {n_stud*4:,} bytes)")

    # Data
    train_data, val_data = load_hybrid_data(args.data, seed=args.seed)
    train_x, train_mp, train_mv, train_up, train_uv = train_data
    val_x, val_mp, val_mv, val_up, val_uv = val_data

    # Combined target labels for regularization
    w_u = args.utttai_weight
    w_m = 1.0 - w_u
    print(f"Hybrid Blend: {w_m:.2f} MCTS + {w_u:.2f} uttt.ai Soft Targets")
    train_p = w_m * train_mp + w_u * train_up
    train_v = w_m * train_mv + w_u * train_uv
    val_p = w_m * val_mp + w_u * val_up
    val_v = w_m * val_mv + w_u * val_uv

    print(f"Train: {len(train_x):,} positions | Val: {len(val_x):,} positions")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = (len(train_x) // args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-5)

    if args.feature_weight > 0:
        proj_micro = nn.Linear(student.micro_channels, teacher.micro_channels).to(device)
        proj_macro = nn.Linear(student.macro_channels, teacher.macro_channels).to(device)
        feat_params = list(proj_micro.parameters()) + list(proj_macro.parameters())
        feat_optimizer = torch.optim.AdamW(feat_params, lr=args.lr, weight_decay=1e-4)
    else:
        proj_micro = None
        proj_macro = None
        feat_optimizer = None

    maps = _symmetry_maps()
    aug_rng = np.random.default_rng(args.seed)

    val_sub = min(25000, len(val_x))
    val_x_t = torch.from_numpy(val_x[:val_sub]).to(device)
    val_p_t = torch.from_numpy(val_p[:val_sub]).to(device)
    val_v_t = torch.from_numpy(val_v[:val_sub]).to(device)

    best_val_loss = float("inf")
    best_weights = None

    if args.swa_start_epoch > 0:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(student)
    else:
        swa_model = None

    T = args.temperature
    alpha = args.alpha

    print(f"Starting {args.epochs} Distillation Epochs into {args.student_tier} (T={T}, alpha={alpha})...")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        student.train()
        perm = np.random.permutation(len(train_x))
        epoch_loss = 0.0
        n_batches = 0

        for start_idx in range(0, len(train_x) - args.batch_size + 1, args.batch_size):
            batch_idx = perm[start_idx : start_idx + args.batch_size]
            bx = train_x[batch_idx]
            bp = train_p[batch_idx]
            bv = train_v[batch_idx]

            syms = aug_rng.integers(0, 8, size=len(batch_idx))
            bx, bp = augment_batch(bx, bp, syms, maps)

            x = torch.from_numpy(bx).to(device)
            hard_p = torch.from_numpy(bp).to(device)
            hard_v = torch.from_numpy(bv).to(device)
            legal_mask = x[:, 162:243] >= 0.5

            if args.feature_weight > 0:
                with torch.no_grad():
                    t_logits, t_val, t_micro, t_macro = teacher(x, return_features=True)
                    t_logits = t_logits.masked_fill(~legal_mask, -1e9)
                    t_soft_p = F.softmax(t_logits / T, dim=1)
                s_logits, s_val, s_micro, s_macro = student(x, return_features=True)
                feat_loss = (F.mse_loss(proj_micro(s_micro), t_micro) + F.mse_loss(proj_macro(s_macro), t_macro)) * args.feature_weight
            else:
                with torch.no_grad():
                    t_logits, t_val = teacher(x)
                    t_logits = t_logits.masked_fill(~legal_mask, -1e9)
                    t_soft_p = F.softmax(t_logits / T, dim=1)
                s_logits, s_val = student(x)
                feat_loss = 0.0

            s_logits_masked = s_logits.masked_fill(~legal_mask, -1e9)

            # Soft teacher distillation loss (KL / cross-entropy)
            s_log_soft_p = F.log_softmax(s_logits_masked / T, dim=1)
            distill_p_loss = -(t_soft_p * s_log_soft_p).sum(dim=1).mean() * (T * T)

            # Hard ground truth policy loss
            hard_p_loss = -(hard_p * F.log_softmax(s_logits_masked, dim=1)).sum(dim=1).mean()

            policy_loss = alpha * distill_p_loss + (1.0 - alpha) * hard_p_loss

            # Value loss: blend teacher value + ground truth
            v_teacher_loss = F.mse_loss(s_val.squeeze(-1), t_val.squeeze(-1))
            v_hard_loss = F.mse_loss(s_val.squeeze(-1), hard_v)
            value_loss = 0.5 * v_teacher_loss + 0.5 * v_hard_loss

            loss = policy_loss + 0.5 * value_loss + feat_loss

            optimizer.zero_grad(set_to_none=True)
            if feat_optimizer is not None:
                feat_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if feat_optimizer is not None:
                feat_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % 500 == 0:
                elapsed = time.time() - start_time
                ups = step / max(elapsed, 1e-3)
                feat_info = f", Feat: {feat_loss.item():.4f}" if isinstance(feat_loss, torch.Tensor) else ""
                print(f"  Step {step:5d}/{total_steps} | Loss: {loss.item():.4f} (P-Distill: {distill_p_loss.item():.4f}, V: {value_loss.item():.4f}{feat_info}) | {ups:.1f} up/s")

        student.eval()
        with torch.no_grad():
            v_logits, v_pred_v = student(val_x_t)
            v_logits = v_logits.masked_fill(val_x_t[:, 162:243] < 0.5, -1e9)
            v_p_loss = -(val_p_t * F.log_softmax(v_logits, dim=1)).sum(dim=1).mean().item()
            v_v_loss = F.mse_loss(v_pred_v.squeeze(-1), val_v_t).item()
            val_total = v_p_loss + 0.5 * v_v_loss

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val P: {v_p_loss:.4f} V: {v_v_loss:.4f} Total: {val_total:.4f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_weights = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        if swa_model is not None and (epoch + 1) >= args.swa_start_epoch:
            swa_model.update_parameters(student)
            print(f"  [SWA] Updated averaged model at epoch {epoch+1}")

    if swa_model is not None:
        swa_model.eval()
        with torch.no_grad():
            v_logits, v_pred_v = swa_model(val_x_t)
            v_logits = v_logits.masked_fill(val_x_t[:, 162:243] < 0.5, -1e9)
            v_p_loss = -(val_p_t * F.log_softmax(v_logits, dim=1)).sum(dim=1).mean().item()
            v_v_loss = F.mse_loss(v_pred_v.squeeze(-1), val_v_t).item()
            swa_val_total = v_p_loss + 0.5 * v_v_loss
        print(f"[SWA] Final SWA Val Loss: {swa_val_total:.4f} vs Best Epoch: {best_val_loss:.4f}")
        if swa_val_total <= best_val_loss:
            print("  --> Adopting SWA averaged weights for promotion!")
            best_weights = {k.replace("module.", ""): v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
            best_val_loss = swa_val_total

    print(f"Distillation completed in {time.time() - start_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Export ONNX
    student.load_state_dict(best_weights)
    student.eval()
    student_cpu = student.to("cpu")
    dummy = torch.randn(1, 280)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, out_path.with_suffix(".pt"))

    torch.onnx.export(
        student_cpu, dummy, str(out_path),
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={"state": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=14,
        dynamo=False
    )
    print(f"Saved distilled ONNX model to {out_path} ({out_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
