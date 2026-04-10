from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from device_utils import get_device


class LandmarkForecaster(nn.Module):
    def __init__(self, num_points: int, hidden_dim: int = 128, num_layers: int = 2, horizon: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_points = int(num_points)
        self.horizon = int(horizon)
        input_dim = self.num_points * 3  # x, y, visible
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.horizon * self.num_points * 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]
        pred = self.head(final_hidden)
        return pred.view(x.shape[0], self.horizon, self.num_points, 2)


@dataclass
class DynamicsTrainingSummary:
    num_samples: int
    final_loss: float
    best_loss: float
    device: str


def _normalize_points(points: np.ndarray, image_hw):
    h, w = image_hw
    out = points.astype(np.float32).copy()
    out[..., 0] = out[..., 0] / max(w - 1, 1)
    out[..., 1] = out[..., 1] / max(h - 1, 1)
    return out


def prepare_dynamics_dataset(tracks: np.ndarray, occlusions: np.ndarray, history: int, horizon: int, image_hw):
    # tracks: K x T x 2, occlusions: K x T
    num_points, num_frames, _ = tracks.shape
    tracks_t = np.transpose(tracks, (1, 0, 2))
    visible_t = (~occlusions).astype(np.float32).T  # T x K
    tracks_norm = _normalize_points(tracks_t, image_hw=image_hw)

    inputs = []
    targets = []
    masks = []
    for start in range(0, num_frames - history - horizon + 1):
        past = tracks_norm[start : start + history]  # H x K x 2
        future = tracks_norm[start + history : start + history + horizon]
        future_visible = visible_t[start + history : start + history + horizon]
        if future_visible.mean() < 0.35:
            continue
        past_visible = visible_t[start : start + history]
        # Use the last valid point as a fill value where occluded.
        filled_past = past.copy()
        for point_idx in range(num_points):
            last_valid = filled_past[0, point_idx]
            for t in range(history):
                if past_visible[t, point_idx] > 0.5:
                    last_valid = filled_past[t, point_idx]
                else:
                    filled_past[t, point_idx] = last_valid
        past_features = np.concatenate([filled_past, past_visible[..., None]], axis=-1)  # H x K x 3
        inputs.append(past_features.reshape(history, num_points * 3))
        targets.append(future)
        masks.append(future_visible)

    if not inputs:
        raise RuntimeError("No usable training windows for landmark dynamics.")

    x = torch.from_numpy(np.stack(inputs, axis=0)).float()
    y = torch.from_numpy(np.stack(targets, axis=0)).float()
    m = torch.from_numpy(np.stack(masks, axis=0)).float()
    return TensorDataset(x, y, m)


def train_landmark_dynamics(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    history: int,
    horizon: int,
    image_hw,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    steps: int = 600,
    seed: int = 7,
):
    torch.manual_seed(seed)
    dataset = prepare_dynamics_dataset(
        tracks=tracks,
        occlusions=occlusions,
        history=history,
        horizon=horizon,
        image_hw=image_hw,
    )
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)
    device = get_device(log=True)
    model = LandmarkForecaster(
        num_points=tracks.shape[0],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    final_loss = float("inf")

    step = 0
    while step < steps:
        for x, y, m in loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            pred = model(x)
            coord_loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")
            coord_loss = coord_loss.mean(dim=-1) * m
            loss = coord_loss.sum() / m.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.item())
            best_loss = min(best_loss, final_loss)
            step += 1
            if step >= steps:
                break

    summary = DynamicsTrainingSummary(
        num_samples=len(dataset),
        final_loss=final_loss,
        best_loss=best_loss,
        device=str(device),
    )
    return model.cpu(), summary


@torch.no_grad()
def predict_future_points(
    model: nn.Module,
    tracks: np.ndarray,
    occlusions: np.ndarray,
    history: int,
    horizon: int,
    image_hw,
):
    num_points, num_frames, _ = tracks.shape
    if num_frames < history:
        raise ValueError("Not enough frames for the requested history window.")

    h, w = image_hw
    visible = (~occlusions).astype(np.float32).T
    tracks_t = np.transpose(tracks, (1, 0, 2))
    tracks_norm = _normalize_points(tracks_t, image_hw=image_hw)
    past = tracks_norm[-history:].copy()
    past_visible = visible[-history:]
    for point_idx in range(num_points):
        last_valid = past[0, point_idx]
        for t in range(history):
            if past_visible[t, point_idx] > 0.5:
                last_valid = past[t, point_idx]
            else:
                past[t, point_idx] = last_valid
    x = np.concatenate([past, past_visible[..., None]], axis=-1).reshape(1, history, num_points * 3)
    x = torch.from_numpy(x).float()

    device = get_device(log=False)
    model = model.to(device)
    pred = model(x.to(device)).cpu().numpy()[0]  # H x K x 2 normalized
    pred[..., 0] *= max(w - 1, 1)
    pred[..., 1] *= max(h - 1, 1)
    return pred.astype(np.float32)
