#!/usr/bin/env python
"""SmolVLA training stub for the reBot cloth-folding dataset.

SmolVLA is a Vision-Language-Action model based on SmolVLM.  This stub provides
the dataset wrapper and training loop scaffold.  It assumes the LeRobot dataset
has been converted by ``wam_fold.lerobot_dataset.LeRobotConverter``.

Requirements (install separately):
    pip install torch transformers lerobot

Usage:
    python train_smolvla.py --dataset data/lerobot_dataset --epochs 100
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from wam_fold.lerobot_dataset import LeRobotDatasetLoader


class FoldDataset(Dataset):
    """PyTorch dataset wrapping the LeRobot-format fold data.

    Each sample: (images_top, images_front, state, action)
    Images are (C, H, W) float32 in [0, 1].
    State/action are (14,) float32.
    """

    def __init__(self, dataset_dir: str, camera_names: tuple = ("top", "front")):
        self.loader = LeRobotDatasetLoader(dataset_dir)
        self.camera_names = camera_names
        self.samples = []
        for ep in self.loader.episodes:
            for step_idx in range(ep["n_steps"]):
                self.samples.append((ep["episode_id"], step_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_id, step_idx = self.samples[idx]
        ep_data = self.loader.get_episode(ep_id)
        state  = torch.from_numpy(ep_data["states"][step_idx]).float()
        action = torch.from_numpy(ep_data["actions"][step_idx]).float()

        images = {}
        for cam in self.camera_names:
            imgs = ep_data["images"].get(cam, [])
            if step_idx < len(imgs) and imgs[step_idx] is not None:
                img = imgs[step_idx].astype(np.float32) / 255.0
                # HWC -> CHW
                images[cam] = torch.from_numpy(img).permute(2, 0, 1)
            else:
                images[cam] = torch.zeros(3, 256, 256)

        return images, state, action


def train(
    dataset_dir: str,
    epochs:      int   = 100,
    batch_size:  int   = 16,
    lr:          float = 1e-4,
    device:      str   = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset = FoldDataset(dataset_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # --- model placeholder ---
    # In practice, load SmolVLA from HuggingFace:
    #   from transformers import AutoModelForVision2Seq
    #   model = AutoModelForVision2Seq.from_pretrained("lerobot/smolvla-base")
    # For this stub we use a simple MLP to verify the data pipeline.
    import torch.nn as nn

    class StubVLA(nn.Module):
        def __init__(self, state_dim=14, action_dim=14):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * 256 * 256 * 2 + state_dim, 512),  # 2 cameras + state
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )

        def forward(self, images_dict, state):
            # Flatten images
            imgs = []
            for cam in ("top", "front"):
                imgs.append(images_dict[cam].flatten(1))
            x = torch.cat(imgs + [state], dim=1)
            return self.net(x)

    model = StubVLA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[train] dataset: {len(dataset)} samples")
    print(f"[train] device: {device}")
    print(f"[train] epochs: {epochs}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for images_dict, state, action in loader:
            state  = state.to(device)
            action = action.to(device)
            images_dict = {k: v.to(device) for k, v in images_dict.items()}

            pred = model(images_dict, state)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[train] epoch {epoch+1}/{epochs}  loss={avg:.6f}")

    # Save model
    torch.save(model.state_dict(), "smolvla_fold.pth")
    print("[train] saved smolvla_fold.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lerobot_dataset")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
