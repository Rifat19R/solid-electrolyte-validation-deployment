import numpy as np
import torch

from sse_transfer.models.torch_regressors import (
    CompositionAttentionRegressor,
    ShallowMLPRegressor,
    attention_forward,
    count_parameters,
    mlp_forward,
    to_tensor,
    train_fixed_epochs,
    train_with_early_stopping,
)


def _linear_toy_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.01, size=n)).astype(np.float32)
    return to_tensor(X), to_tensor(y)


def test_shallow_mlp_forward_output_shape() -> None:
    model = ShallowMLPRegressor(in_dim=3, hidden_dims=(8,), dropout=0.0)
    x = to_tensor(np.zeros((5, 3), dtype=np.float32))
    out = mlp_forward(model, (x,))
    assert out.shape == (5,)


def test_shallow_mlp_linear_head_has_no_hidden_layer_params_beyond_output() -> None:
    model = ShallowMLPRegressor(in_dim=3, hidden_dims=(), dropout=0.0)
    # A pure linear map: in_dim weights + 1 bias.
    assert count_parameters(model) == 3 + 1


def test_train_with_early_stopping_fits_a_linear_relationship() -> None:
    X_train, y_train = _linear_toy_data(60, seed=0)
    X_val, y_val = _linear_toy_data(20, seed=1)
    model = ShallowMLPRegressor(in_dim=3, hidden_dims=(16,), dropout=0.0)
    result = train_with_early_stopping(
        model,
        mlp_forward,
        (X_train,),
        y_train,
        (X_val,),
        y_val,
        lr=1e-2,
        weight_decay=0.0,
        max_epochs=300,
        patience=30,
        loss="mse",
        seed=17,
    )
    assert result.best_epoch >= 0
    assert result.val_mae_history[result.best_epoch] < 1.0
    assert result.n_parameters == count_parameters(model)


def test_train_with_early_stopping_is_deterministic_for_fixed_seed() -> None:
    X_train, y_train = _linear_toy_data(40, seed=2)
    X_val, y_val = _linear_toy_data(15, seed=3)

    def _run() -> list[float]:
        # Model construction (weight init) must be seeded by the caller, same
        # as scripts/phase1_9_composition_ood.py's build_model helpers do;
        # train_with_early_stopping's own internal seeding only covers
        # training-time stochasticity (e.g. dropout), not weight init.
        torch.manual_seed(101)
        model = ShallowMLPRegressor(in_dim=3, hidden_dims=(8,), dropout=0.0)
        result = train_with_early_stopping(
            model,
            mlp_forward,
            (X_train,),
            y_train,
            (X_val,),
            y_val,
            lr=1e-2,
            weight_decay=0.0,
            max_epochs=20,
            patience=20,
            loss="mse",
            seed=101,
        )
        return result.val_mae_history

    assert _run() == _run()


def test_train_with_early_stopping_stops_before_max_epochs_when_no_improvement() -> None:
    X_train, y_train = _linear_toy_data(30, seed=4)
    X_val, y_val = _linear_toy_data(10, seed=5)
    model = ShallowMLPRegressor(in_dim=3, hidden_dims=(4,), dropout=0.0)
    result = train_with_early_stopping(
        model,
        mlp_forward,
        (X_train,),
        y_train,
        (X_val,),
        y_val,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=500,
        patience=5,
        loss="mse",
        seed=7,
    )
    assert result.n_epochs_run <= 500
    if result.converged:
        assert result.n_epochs_run < 500


def test_train_fixed_epochs_runs_requested_number_of_steps() -> None:
    X_train, y_train = _linear_toy_data(25, seed=6)
    model = ShallowMLPRegressor(in_dim=3, hidden_dims=(4,), dropout=0.0)
    fitted = train_fixed_epochs(
        model,
        mlp_forward,
        (X_train,),
        y_train,
        lr=1e-2,
        weight_decay=0.0,
        n_epochs=10,
        loss="mse",
        seed=17,
    )
    assert fitted is model


def test_composition_attention_forward_output_shape_and_masking() -> None:
    model = CompositionAttentionRegressor(
        vocab_size=10, d_model=8, n_heads=2, n_layers=1, dropout=0.0
    )
    idx = torch.tensor([[2, 3, 0], [4, 0, 0]], dtype=torch.long)
    frac = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    out = attention_forward(model, (idx, frac, mask))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_composition_attention_param_count_matches_helper() -> None:
    model = CompositionAttentionRegressor(
        vocab_size=5, d_model=8, n_heads=2, n_layers=1, dropout=0.0
    )
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())
