from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SequenceThermalDepthDataset
from .losses import combined_loss, temporal_consistency_loss
from .models import ThermalDepthNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thermal depth training entrypoint")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--recurrent", type=str, default="convgru")
    p.add_argument("--backbone", type=str, default="simple_cnn")
    p.add_argument("--temporal_lambda", type=float, default=0.1)
    p.add_argument("--save_dir", type=Path, default=Path("checkpoints"))
    return p.parse_args()


def _build_loader(args: argparse.Namespace, list_file: str, augment: bool) -> DataLoader:
    ds = SequenceThermalDepthDataset(
        root_dir=args.root,
        list_file=list_file,
        seq_len=args.seq_len,
        out_h=args.height,
        out_w=args.width,
        return_last_only=True,
        augment_flip=augment,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=augment, num_workers=args.workers, pin_memory=True)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = _build_loader(args, args.train_list, augment=True)
    val_loader = _build_loader(args, args.val_list, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalDepthNet(backbone=args.backbone, recurrent=args.recurrent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch["thermal_seq"].to(device)
            gt = batch["depth"].to(device)
            mask = batch["mask"].to(device)

            out = model(x, decode_all=args.temporal_lambda > 0)
            loss = combined_loss(out.depth, gt, mask=mask, image=out.refined_last)
            if args.temporal_lambda > 0 and out.depth_sequence is not None:
                m = mask.unsqueeze(1).repeat(1, out.depth_sequence.shape[1], 1, 1)
                loss = loss + args.temporal_lambda * temporal_consistency_loss(out.depth_sequence, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())

        train_loss = total / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for batch in val_loader:
                x = batch["thermal_seq"].to(device)
                gt = batch["depth"].to(device)
                mask = batch["mask"].to(device)
                out = model(x, decode_all=False)
                val_total += float(combined_loss(out.depth, gt, mask=mask, image=out.refined_last).cpu())
            val_loss = val_total / max(1, len(val_loader))

        print(f"epoch={epoch+1} train={train_loss:.4f} val={val_loss:.4f}")

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, args.save_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, args.save_dir / "best.pt")


if __name__ == "__main__":
    main()
