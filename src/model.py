"""Model factory, hyperparameter tuning, and evaluation.

Three candidate regressors are tuned with RandomizedSearchCV on a
TimeSeriesSplit, scored by negative MAE. The best validation-MAE model wins.
Tree-based regressors are scale-invariant, so we don't apply a StandardScaler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]

log = logging.getLogger("model")


class EnsembleRegressor:
    """Averaging ensemble of fixed-hyperparameter models with different seeds.

    Wraps a list of pre-fitted regressors and exposes `predict()` (arithmetic
    mean) and `feature_importances_` (averaged). Picklable — safe for joblib.
    """

    def __init__(self, models: List[Any]) -> None:
        self.models = models

    def predict(self, X: Any) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        imps = [
            m.feature_importances_
            for m in self.models
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return None
        return np.stack(imps, axis=0).mean(axis=0)


@dataclass
class Candidate:
    name: str
    builder: Callable[[int], Any]
    param_distributions: Dict[str, Any] = field(default_factory=dict)


def _rf_builder(random_state: int):
    return RandomForestRegressor(
        n_jobs=-1,
        random_state=random_state,
    )


def _lgbm_builder(random_state: int):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")
    # MSE objective: empirically gave better val MAE 1.62 vs L1 loss's 1.67
    # on this dataset (commit history). The MSE gradient is smoother and
    # the booster finds better splits even when the reported metric is MAE.
    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        force_row_wise=True,
        verbose=-1,
    )


def _xgb_builder(random_state: int):
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    # Same MSE rationale as LightGBM above.
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def candidates(random_state: int = 42) -> Dict[str, Candidate]:
    cands: Dict[str, Candidate] = {
        "random_forest": Candidate(
            name="random_forest",
            builder=_rf_builder,
            param_distributions={
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [None, 8, 12, 16, 24],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.5, 0.75, 1.0],
            },
        ),
    }
    if LGBMRegressor is not None:
        cands["lightgbm"] = Candidate(
            name="lightgbm",
            builder=_lgbm_builder,
            param_distributions={
                "n_estimators": [300, 500, 700, 1000],
                "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
                "num_leaves": [31, 63, 127, 255],
                "max_depth": [-1, 6, 10, 15, 20],
                "min_child_samples": [10, 20, 30, 50, 100],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
                "reg_lambda": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
                "min_split_gain": [0.0, 0.01, 0.05, 0.1],
            },
        )
    if XGBRegressor is not None:
        cands["xgboost"] = Candidate(
            name="xgboost",
            builder=_xgb_builder,
            param_distributions={
                "n_estimators": [300, 500, 700, 1000],
                "max_depth": [4, 6, 8, 10, 12],
                "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 10, 20],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
                "gamma": [0.0, 0.05, 0.1, 0.3, 1.0],
            },
        )
    return cands


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> Dict[str, float]:
    """Regression metrics + the in-domain `accuracy within ±1 day`."""
    y_pred_clipped = np.clip(np.round(y_pred), 0, horizon)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc1 = float(np.mean(np.abs(y_pred_clipped - y_true) <= 1))
    return {
        "mae_days": round(mae, 4),
        "rmse_days": round(rmse, 4),
        "r2": round(r2, 4),
        "accuracy_within_1day": round(acc1, 4),
    }


def tune_candidate(
    cand: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: int = 0,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """Randomized search with TimeSeriesSplit, scored on neg-MAE."""
    estimator = cand.builder(random_state)
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=cand.param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=verbose,
        refit=True,
    )

    log.info("Tuning %s: %d iter × %d splits", cand.name, n_iter, n_splits)
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    search.fit(X, y, **fit_kwargs)
    best_cv_mae = -float(search.best_score_)
    log.info("%s best CV MAE = %.4f, params = %s", cand.name, best_cv_mae, search.best_params_)
    return search.best_estimator_, search.best_params_, best_cv_mae


def build_ensemble(
    cand: Candidate,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_ensemble: int = 5,
    base_seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> EnsembleRegressor:
    """Fit `n_ensemble` copies of `cand` at `best_params` with different random seeds.

    Each model sees the same training data but uses a different seed for
    feature/sample subsampling, so their errors are weakly correlated.
    Averaging reduces prediction variance without touching bias — typically
    yields 3–8% MAE improvement over a single model.
    """
    models: List[Any] = []
    for i in range(n_ensemble):
        seed = base_seed + i * 37
        m = cand.builder(seed)
        m.set_params(**best_params)
        if sample_weight is not None:
            m.fit(X, y, sample_weight=sample_weight)
        else:
            m.fit(X, y)
        models.append(m)
    log.info(
        "Ensemble: trained %d %s models (seeds %d…%d)",
        n_ensemble, cand.name, base_seed, base_seed + (n_ensemble - 1) * 37,
    )
    return EnsembleRegressor(models)


def select_best(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    only: Optional[Tuple[str, ...]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tune every candidate and return a result dict for the best validation-MAE model."""
    cands = candidates(random_state=random_state)
    if only:
        cands = {k: v for k, v in cands.items() if k in only}
    if not cands:
        raise RuntimeError("No candidates available — check installs (lightgbm/xgboost)")

    results: Dict[str, Dict[str, Any]] = {}
    fitted: Dict[str, Any] = {}

    for name, cand in cands.items():
        try:
            model, best_params, cv_mae = tune_candidate(
                cand,
                X_train,
                y_train,
                n_iter=n_iter,
                n_splits=n_splits,
                random_state=random_state,
                sample_weight=sample_weight,
            )
        except Exception as exc:
            log.exception("Tuning failed for %s: %s", name, exc)
            continue

        y_pred = model.predict(X_val)
        metrics = evaluate(y_val.to_numpy(), y_pred, horizon=horizon)
        metrics["cv_mae_days"] = round(cv_mae, 4)

        log.info(
            "%s — val MAE %.4f, RMSE %.4f, R² %.4f, acc±1 %.2f%%",
            name,
            metrics["mae_days"],
            metrics["rmse_days"],
            metrics["r2"],
            100 * metrics["accuracy_within_1day"],
        )

        fitted[name] = model
        results[name] = {
            "best_params": best_params,
            "metrics": metrics,
        }

    if not results:
        raise RuntimeError("All candidates failed to fit")

    best_name = min(results, key=lambda k: results[k]["metrics"]["mae_days"])
    log.info(
        "🏆 Best model: %s (val MAE = %.4f)",
        best_name,
        results[best_name]["metrics"]["mae_days"],
    )

    return {
        "best_name": best_name,
        "best_model": fitted[best_name],
        "all_results": results,
    }
