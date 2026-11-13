#!/usr/bin/env python
"""ACT (Action Chunking with Transformers) training stub.

ACT predicts a chunk of future actions from the current observation.
Paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
(Zhao et al., 2023).

This stub provides the chunked dataset wrapper and a simplified transformer
encoder-decoder.  It uses the same LeRobot-format data.

Requirements:
    pip install torch

Usage:
    python train_act.py --dataset data/lerobot_dataset --epochs 100 \
        --chunk-size 20
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from wam_fold.lerobot_dataset import LeRobotDatasetLoader


class ChunkedFoldDataset(Dataset):
    """Dataset that returns (obs, action_chunk) pairs.

    Each obs: (images_top, images_front, state)
    Each action_chunk: (chunk_size, 14) future actions from current step.
    """

    def __init__(self, dataset_dir: str, chunk_size: int = 20, camera_names: tuple = ("top", "front")):
        self.loader = LeRobotDatasetLoader(dataset_dir)
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.samples = []
        for ep in self.loader.episodes:
            s, e = ep["start_idx"], ep["end_idx"]
            for i in range(s, e):
                # Need enough future steps for the chunk
                remaining = e - i
                if remaining >= 1:
                    self.samples.append((ep["episode_id"], i - s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_id, step_idx = self.samples[idx]
        ep_data = self.loader.get_episode(ep_id)

        state  = torch.from_numpy(ep_data["states"][step_idx]).float()
        actions = torch.from_numpy(ep_data["actions"][step_idx:]).float()

        # Pad or truncate to chunk_size
        chunk = torch.zeros(self.chunk_size, 14)
        n = min(self.chunk_size, len(actions))
        chunk[:n] = actions[:n]

        images = {}
        for cam in self.camera_names:
            imgs = ep_data["images"].get(cam, [])
            if step_idx < len(imgs) and imgs[step_idx] is not None:
                img = imgs[step_idx].astype(np.float32) / 255.0
                images[cam] = torch.from_numpy(img).permute(2, 0, 1)
            else:
                images[cam] = torch.zeros(3, 256, 256)

        return images, state, chunk


class SimpleACT(nn.Module):
    """Simplified ACT: CNN encoder + transformer encoder-decoder.

    This is a minimal stub to verify the data pipeline.  The full ACT uses
    a ResNet-18 backbone and a transformer with CVAE (conditional VAE) training.
    """

    def __init__(self, state_dim=14, action_dim=14, chunk_size=20,
                 hidden_dim=256, n_heads=4, n_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Simple CNN image encoder (replace with ResNet-18 in practice)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # 2 cameras
        img_feat_dim = 128 * 2
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.img_proj    = nn.Linear(img_feat_dim, hidden_dim)

        # Transformer encoder-decoder
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size + 1, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=512, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=512, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, images_dict, state):
        B = state.shape[0]
        # Encode images
        feats = []
        for cam in ("top", "front"):
            feat = self.img_encoder(images_dict[cam])
            feats.append(feat)
        img_feat = torch.cat(feats, dim=1)
        img_emb  = self.img_proj(img_feat).unsqueeze(1)   # (B, 1, H)
        state_emb = self.state_proj(state).unsqueeze(1)  # (B, 1, H)
        # Query embeddings for action chunk
        queries = torch.zeros(B, self.chunk_size, img_emb.shape[2], device=state.device)
        queries = torch.cat([state_emb, queries], dim=1)  # (B, chunk+1, H)
        queries = queries + self.pos_embed[:, :queries.shape[1]]
        # Encoder input: image + state
        enc_input = torch.cat([img_emb, state_emb], dim=1)
        enc_out = self.encoder(enc_input)
        # Decode
        dec_out = self.decoder(queries, enc_out)
        actions = self.action_head(dec_out)  # (B, chunk+1, action_dim)
        # Drop the first (state query), keep chunk_size action predictions
        return actions[:, 1:]  # (B, chunk_size, action_dim)


def train(
    dataset_dir:  str,
    epochs:       int   = 100,
    batch_size:   int   = 16,
    lr:           float = 1e-4,
    chunk_size:   int   = 20,
    device:       str   = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset = ChunkedFoldDataset(dataset_dir, chunk_size=chunk_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SimpleACT(chunk_size=chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[ACT] dataset: {len(dataset)} samples")
    print(f"[ACT] chunk_size: {chunk_size}")
    print(f"[ACT] device: {device}")
    print(f"[ACT] epochs: {epochs}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for images_dict, state, action_chunk in loader:
            state = state.to(device)
            action_chunk = action_chunk.to(device)
            images_dict = {k: v.to(device) for k, v in images_dict.items()}

            pred = model(images_dict, state)
            loss = criterion(pred, action_chunk)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[ACT] epoch {epoch+1}/{epochs}  loss={avg:.6f}")

    torch.save(model.state_dict(), "act_fold.pth")
    print("[ACT] saved act_fold.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lerobot_dataset")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk-size", type=int, default=20)
    args = ap.parse_args()
    train(args.dataset, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, chunk_size=args.chunk_size)
