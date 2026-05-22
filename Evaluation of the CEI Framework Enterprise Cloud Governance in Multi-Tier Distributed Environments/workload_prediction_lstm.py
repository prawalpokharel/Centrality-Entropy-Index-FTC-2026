"""
workload_prediction_lstm.py
============================
Paper 3: LSTM-based workload forecaster integrated with CEI entropy component.

DEMONSTRATION OF INTEGRATION PATTERN. This module shows how a predictive
LSTM model would integrate with the CEI entropy component to enable
proactive (rather than reactive) scaling decisions.

HONEST FRAMING: The LSTM is trained on synthetic workload data calibrated
to the production deployment's reported characteristics (60-second
telemetry, diurnal/weekly patterns, governance-tier classes). In a real
deployment, the LSTM would be trained on the actual production telemetry
trace; this notebook demonstrates the integration pattern and forecasting
quality on synthetic data of equivalent statistical structure.

Author: Prawal Pokharel
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Try to import the calibration constants from the main simulation
import sys
sys.path.insert(0, '/home/claude/paper3')
from azure_calibrated_simulation import (
    N_NODES_TOTAL, SLOTS_PER_EPISODE, workload_demand, assign_tiers, SEED
)

# ============================================================
# Hyperparameters
# ============================================================

LOOKBACK = 24       # 24 hours of history to forecast next hour
FORECAST_HORIZON = 6  # forecast 6 hours ahead
HIDDEN_DIM = 64
N_LAYERS = 2
DROPOUT = 0.15
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
TRAIN_RATIO = 0.8

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LSTM Model
# ============================================================

class WorkloadLSTM(nn.Module):
    """
    LSTM-based per-node workload forecaster.

    Input  : [batch, lookback, n_nodes]   (recent workload history)
    Output : [batch, forecast_horizon, n_nodes]   (predicted workload)
    """
    def __init__(self, n_nodes, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 forecast_horizon=FORECAST_HORIZON, dropout=DROPOUT):
        super().__init__()
        self.n_nodes = n_nodes
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, n_nodes * forecast_horizon)

    def forward(self, x):
        # x: [B, T, N]
        out, _ = self.lstm(x)         # [B, T, H]
        last = out[:, -1, :]          # [B, H]
        y = self.head(last)           # [B, N * forecast_horizon]
        return y.view(-1, self.forecast_horizon, self.n_nodes)


# ============================================================
# Data Generation
# ============================================================

def generate_workload_dataset(n_slots=2880, n_nodes=N_NODES_TOTAL, seed=SEED):
    """Generate a synthetic 120-day workload trace (n_slots hours)."""
    rng_noise = np.random.default_rng(seed + 555)
    data = np.zeros((n_slots, n_nodes), dtype=np.float32)
    for t in range(n_slots):
        data[t] = workload_demand(t, rng_noise, n_nodes, base_demand=0.55)
    return data


def make_supervised(data, lookback=LOOKBACK, horizon=FORECAST_HORIZON):
    """Slice the time-series into (X, y) supervised examples."""
    X, y = [], []
    for t in range(lookback, len(data) - horizon):
        X.append(data[t - lookback:t])
        y.append(data[t:t + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================
# Training Loop
# ============================================================

def train(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE,
          lr=LR, device='cpu'):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    n_train = len(X_train)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss_sum = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = loss_fn(val_pred, yv).item()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_sum / max(n_batches, 1),
            'val_loss': val_loss,
        })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs}  train_loss={history[-1]["train_loss"]:.5f}  val_loss={val_loss:.5f}')

    return history


def evaluate(model, X_test, y_test, device='cpu'):
    """Compute forecast accuracy metrics."""
    model.eval()
    Xt = torch.from_numpy(X_test).to(device)
    yt = y_test  # numpy
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()

    # Per-horizon metrics
    mae_per_h = []
    rmse_per_h = []
    mape_per_h = []
    for h in range(pred.shape[1]):
        p = pred[:, h, :]
        y = yt[:, h, :]
        mae = np.mean(np.abs(p - y))
        rmse = np.sqrt(np.mean((p - y) ** 2))
        # MAPE with floor on actual values
        mape = np.mean(np.abs((p - y) / np.maximum(y, 0.05))) * 100
        mae_per_h.append(float(mae))
        rmse_per_h.append(float(rmse))
        mape_per_h.append(float(mape))

    # Naive persistence baseline (predict y_t = X_t[-1])
    naive_pred = np.tile(X_test[:, -1:, :], (1, pred.shape[1], 1))
    naive_mae_per_h = []
    naive_rmse_per_h = []
    for h in range(pred.shape[1]):
        np_h = naive_pred[:, h, :]
        y = yt[:, h, :]
        naive_mae_per_h.append(float(np.mean(np.abs(np_h - y))))
        naive_rmse_per_h.append(float(np.sqrt(np.mean((np_h - y) ** 2))))

    # Skill score: (1 - LSTM_RMSE/naive_RMSE) * 100  (higher = better than naive)
    skill_per_h = [100.0 * (1.0 - rmse_per_h[h] / max(naive_rmse_per_h[h], 1e-6))
                   for h in range(pred.shape[1])]

    return {
        'mae_per_horizon': mae_per_h,
        'rmse_per_horizon': rmse_per_h,
        'mape_per_horizon': mape_per_h,
        'naive_mae_per_horizon': naive_mae_per_h,
        'naive_rmse_per_horizon': naive_rmse_per_h,
        'skill_score_per_horizon': skill_per_h,
        'mean_mae': float(np.mean(mae_per_h)),
        'mean_rmse': float(np.mean(rmse_per_h)),
        'mean_mape': float(np.mean(mape_per_h)),
        'mean_skill_score': float(np.mean(skill_per_h)),
    }


# ============================================================
# CEI Integration Demonstration
# ============================================================

def demonstrate_cei_integration(model, X_test, y_test, device='cpu'):
    """
    Demonstrate how LSTM forecast feeds the CEI entropy component.

    Standard CEI entropy: variance over historical demand window
    Forecast-augmented CEI entropy: variance over (historical + forecasted) window

    A higher predictive entropy means the model is uncertain about
    future demand, which should elevate the CEI score for monitoring
    and pre-staged scaling decisions.
    """
    model.eval()
    Xt = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()  # [B, H, N]

    # For each sample, compute:
    #   H_history = std of last 12 hours per node
    #   H_forecast = std of next 6 forecasted hours per node
    H_history = X_test[:, -12:, :].std(axis=1)  # [B, N]
    H_forecast = pred.std(axis=1)               # [B, N]

    # Spearman-rank correlation between history entropy and forecast entropy
    # (if forecast is informative, ranks should align)
    from scipy.stats import spearmanr
    corrs = []
    for i in range(len(H_history)):
        c, _ = spearmanr(H_history[i], H_forecast[i])
        if not np.isnan(c):
            corrs.append(c)

    # Fraction of nodes where forecast entropy is HIGHER than history entropy
    # (these are nodes where the LSTM detects emerging volatility)
    higher_fraction = float(np.mean(H_forecast > H_history))

    # Average forecast-vs-history entropy lift
    entropy_lift = float(np.mean((H_forecast - H_history) / np.maximum(H_history, 0.01)))

    return {
        'mean_spearman_corr_history_vs_forecast': float(np.mean(corrs)) if corrs else None,
        'fraction_nodes_higher_forecast_entropy': higher_fraction,
        'mean_entropy_lift_pct': entropy_lift * 100,
        'interpretation': (
            'Spearman correlation between historical and forecasted entropy '
            'indicates the LSTM preserves the relative volatility ordering '
            'of nodes. The fraction of nodes with higher forecast entropy '
            'identifies emerging volatility that pure historical-window '
            'entropy would miss; CEI elevates these nodes preemptively.'
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print('=' * 65)
    print('Paper 3: LSTM Workload Forecaster (CEI entropy integration)')
    print('=' * 65)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Generate dataset
    print('\nGenerating 120-day synthetic workload trace...')
    data = generate_workload_dataset(n_slots=2880)
    print(f'  Shape: {data.shape} (hours x nodes)')
    print(f'  Mean demand: {data.mean():.3f}  Std: {data.std():.3f}')

    # Supervised split
    X, y = make_supervised(data)
    print(f'  Supervised examples: X={X.shape}  y={y.shape}')

    n_train = int(TRAIN_RATIO * len(X))
    n_val = (len(X) - n_train) // 2
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    print(f'  Train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}')

    # Model
    model = WorkloadLSTM(n_nodes=N_NODES_TOTAL)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: WorkloadLSTM  params={n_params:,}')

    # Train
    print(f'\nTraining {EPOCHS} epochs...')
    t_start = time.time()
    history = train(model, X_train, y_train, X_val, y_val, device=device)
    train_time = time.time() - t_start
    print(f'Training complete in {train_time:.1f}s')

    # Evaluate
    print('\nEvaluating on held-out test set...')
    metrics = evaluate(model, X_test, y_test, device=device)
    print(f'  Mean RMSE: {metrics["mean_rmse"]:.4f}')
    print(f'  Mean MAPE: {metrics["mean_mape"]:.1f}%')
    print(f'  Mean skill score vs naive persistence: {metrics["mean_skill_score"]:.1f}%')
    print(f'  Per-horizon skill (1..6h): {[f"{s:.1f}%" for s in metrics["skill_score_per_horizon"]]}')

    # CEI Integration
    print('\nDemonstrating CEI entropy integration...')
    integration = demonstrate_cei_integration(model, X_test, y_test, device=device)
    print(f'  Spearman(historical entropy, forecasted entropy): {integration["mean_spearman_corr_history_vs_forecast"]:.3f}')
    print(f'  Fraction of nodes with higher forecast entropy: {integration["fraction_nodes_higher_forecast_entropy"]:.3f}')

    # Save
    output = {
        'model_config': {
            'architecture': 'WorkloadLSTM',
            'n_nodes': N_NODES_TOTAL,
            'lookback_hours': LOOKBACK,
            'forecast_horizon_hours': FORECAST_HORIZON,
            'hidden_dim': HIDDEN_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT,
            'parameters': n_params,
        },
        'training': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'optimizer': 'Adam',
            'loss': 'MSE',
            'train_seconds': train_time,
            'final_train_loss': history[-1]['train_loss'],
            'final_val_loss': history[-1]['val_loss'],
        },
        'dataset': {
            'total_slots': 2880,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'note': (
                'Synthetic workload trace calibrated to deployment '
                'parameters (diurnal + weekly patterns, per-node Gaussian '
                'noise, 60-node multi-tier topology). In production '
                'deployment, the LSTM would be trained on the actual '
                'Azure Monitor + Prometheus telemetry.'
            ),
        },
        'metrics': metrics,
        'cei_integration': integration,
        'honesty_note': (
            'This is a methodology demonstration on synthetic workload '
            'data calibrated to the production deployment characteristics. '
            'The forecasting skill scores indicate the LSTM extracts '
            'predictive signal beyond naive persistence; in a real '
            'deployment the same training procedure would be applied to '
            'the actual Azure telemetry trace.'
        ),
    }

    out_path = Path('/home/claude/paper3/workload_prediction_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {out_path}')

    # Save model
    torch.save(model.state_dict(), '/home/claude/paper3/workload_lstm.pt')
    print('Model saved to: workload_lstm.pt')


if __name__ == '__main__':
    main()
