import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from const import *

np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
except Exception:
    TORCH_AVAILABLE = False


def ensure_2d(X):
    """Ensure input is a 2D array by reshaping 1D inputs into column vectors."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def ensure_2d_query(X):
    """Ensure query inputs are represented as 2D row vectors."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def softplus_np(z):
    """Numerically stable softplus implementation."""

    z = np.asarray(z)
    large_mask = z > 10
    small_mask = z < -10
    medium_mask = ~(large_mask | small_mask)

    result = np.zeros_like(z)
    result[large_mask] = z[large_mask]
    result[small_mask] = 0.0
    result[medium_mask] = np.log1p(np.exp(z[medium_mask]))
    return result


def infer_task_from_sources(sources) -> str:
    """Infer task type from a list of source models.

    We avoid relying on concrete Python classes so that NN-backed wrappers
    (and other custom sources) can plug into the MDCP pipeline.

    Convention:
    - classification sources implement joint_prob(_at_pairs)
    - regression sources implement joint_pdf(_at_pairs)
    """

    if not sources:
        raise ValueError("sources must be a non-empty list")
    src0 = sources[0]
    has_prob = hasattr(src0, 'joint_prob_at_pairs') or hasattr(src0, 'joint_prob')
    has_pdf = hasattr(src0, 'joint_pdf_at_pairs') or hasattr(src0, 'joint_pdf')

    # Prefer classification if both exist (rare; indicates aliasing)
    if has_prob and not has_pdf:
        return 'classification'
    if has_pdf and not has_prob:
        return 'regression'
    if has_prob and has_pdf:
        return 'classification'
    raise AttributeError(
        "Source models must implement either joint_prob(_at_pairs) for classification or joint_pdf(_at_pairs) for regression"
    )


class NNSourceModelClassification:
    """Classification source backed by a torch module mapping X -> logits.

    This is intended for per-source heads that operate on embeddings.
    """

    def __init__(
        self,
        model: 'torch.nn.Module',
        *,
        device: str = 'cpu',
        probability_floor: float = 1e-12,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available. Install torch to use NNSourceModelClassification.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._prob_floor = float(probability_floor)

    def f_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def predict_proba(self, x_query):
        Xq = ensure_2d(x_query).astype(np.float32)
        tx = torch.tensor(Xq, device=self.device)
        with torch.no_grad():
            logits = self.model(tx)
            probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
        probs_np = np.clip(probs_np, self._prob_floor, 1.0)
        probs_sum = np.maximum(probs_np.sum(axis=1, keepdims=True), 1e-12)
        return probs_np / probs_sum

    def joint_prob(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        probs = self.predict_proba(Xq)
        y_vals = np.asarray(y_vals, dtype=int)
        out = np.full((Xq.shape[0], len(y_vals)), self._prob_floor)
        for idx, y in enumerate(y_vals):
            if 0 <= int(y) < probs.shape[1]:
                out[:, idx] = probs[:, int(y)]
        return out

    def joint_prob_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        probs = self.predict_proba(Xpairs)
        Ypairs = np.asarray(Ypairs, dtype=int).reshape(-1)
        rows = np.arange(Ypairs.shape[0])
        valid = (Ypairs >= 0) & (Ypairs < probs.shape[1])
        result = np.full(Ypairs.shape[0], self._prob_floor)
        if np.any(valid):
            result[valid] = probs[rows[valid], Ypairs[valid]]
        return np.clip(result, self._prob_floor, 1.0)


class NNSourceModelRegressionGaussian:
    """Gaussian regression source backed by a torch module mapping X -> (mu, scale).

    The wrapped model may return:
    - tuple/list: (mu, scale)
    - object with attributes .mean and .scale
    - tensor with last-dim 2 representing (mu, raw_scale)
    """

    def __init__(
        self,
        model: 'torch.nn.Module',
        *,
        device: str = 'cpu',
        variance_floor: float = 1e-6,
        pdf_floor: float = 1e-12,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available. Install torch to use NNSourceModelRegressionGaussian.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._var_floor = float(variance_floor)
        self._pdf_floor = float(pdf_floor)
        self._softplus = nn.Softplus()

    def marginal_pdf_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def _predict_mu_sigma(self, x_query):
        Xq = ensure_2d(x_query).astype(np.float32)
        tx = torch.tensor(Xq, device=self.device)
        with torch.no_grad():
            out = self.model(tx)
            if hasattr(out, 'mean') and hasattr(out, 'scale'):
                mu_t = out.mean
                scale_t = out.scale
            elif isinstance(out, (tuple, list)) and len(out) >= 2:
                mu_t, scale_t = out[0], out[1]
            else:
                mu_t = out[:, 0]
                scale_t = out[:, 1]

            mu = mu_t.reshape(-1).detach().cpu().numpy()
            scale_raw = scale_t.reshape(-1)
            if torch.any(scale_raw <= 0):
                scale_pos = self._softplus(scale_raw) + float(np.sqrt(self._var_floor))
            else:
                scale_pos = scale_raw
            sigma = scale_pos.detach().cpu().numpy()
        sigma = np.maximum(sigma, float(np.sqrt(self._var_floor)))
        return mu, sigma

    def predict_mu(self, x_query):
        """Return mean prediction mu(x) for regression baselines."""

        mu, _ = self._predict_mu_sigma(x_query)
        return mu

    def predict_sigma(self, x_query):
        """Return scale prediction sigma(x) for regression baselines."""

        _, sigma = self._predict_mu_sigma(x_query)
        return sigma

    def joint_pdf(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        y_vals = np.asarray(y_vals, dtype=float)
        mu, sigma = self._predict_mu_sigma(Xq)
        mu = mu[:, None]
        sigma = np.maximum(sigma[:, None], 1e-12)
        z = (y_vals[None, :] - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self._pdf_floor, None)

    def joint_pdf_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        Ypairs = np.asarray(Ypairs, dtype=float).reshape(-1)
        mu, sigma = self._predict_mu_sigma(Xpairs)
        sigma = np.maximum(sigma, 1e-12)
        z = (Ypairs - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self._pdf_floor, None)


class SourceModelClassification:
    """Conditional probability estimator with cross-validated calibration."""

    def __init__(
        self,
        X,
        Y,
        learner='gbm',
        calibration_method='isotonic',
        n_splits=5,
        probability_floor=1e-6,
    ):
        self.X = ensure_2d(X)
        self.Y = np.asarray(Y).astype(int)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        self.classes, class_counts = np.unique(self.Y, return_counts=True)
        if self.classes.size < 2:
            raise ValueError("Classification source requires at least two classes.")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.learner = learner
        self.calibration_method = calibration_method
        self.prob_floor = probability_floor
        self.n_splits = self._resolve_cv_splits(class_counts, n_splits)

        base_estimator = self._build_base_estimator()
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        try:
            self.model = CalibratedClassifierCV(
                base_estimator=base_estimator,
                method=self.calibration_method,
                cv=cv,
            )
        except TypeError:
            # scikit-learn >=1.4 renamed argument to ``estimator``
            self.model = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=cv,
            )
        self.model.fit(self.X, self.Y)
        self.classes_ = self.model.classes_

    def _resolve_cv_splits(self, class_counts, requested):
        min_per_class = int(np.min(class_counts))
        return max(2, min(requested, min_per_class))

    def _build_base_estimator(self):
        learner = (self.learner or 'gbm').lower()
        if learner == 'rf':
            return RandomForestClassifier(
                n_estimators=500,
                max_features='sqrt',
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        return HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            learning_rate=0.1,
        )

    def f_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def predict_proba(self, x_query):
        Xq = ensure_2d(x_query)
        probs = self.model.predict_proba(Xq)
        probs = np.clip(probs, self.prob_floor, 1.0)
        probs_sum = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs / probs_sum

    def f_y_given_x(self, x_query):
        return self.predict_proba(x_query)

    def joint_prob(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        probs = self.predict_proba(Xq)
        y_vals = np.asarray(y_vals)
        out = np.full((Xq.shape[0], len(y_vals)), self.prob_floor)

        for idx, y in enumerate(y_vals):
            if y in self.class_to_idx:
                out[:, idx] = probs[:, self.class_to_idx[y]]
        return out

    def joint_prob_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        probs = self.predict_proba(Xpairs)
        Ypairs = np.asarray(Ypairs).reshape(-1)
        result = np.full(Ypairs.shape[0], self.prob_floor)
        idx = np.arange(Ypairs.shape[0])
        y_indices = np.array([self.class_to_idx.get(y, -1) for y in Ypairs])
        mask = y_indices >= 0
        if np.any(mask):
            result[mask] = probs[idx[mask], y_indices[mask]]
        return result


class SourceModelRegressionGaussian:
    """Gaussian plug-in conditional density obtained via OOF mean and log-variance models."""

    def __init__(
        self,
        X,
        Y,
        learner='gbm',
        n_splits=5,
        variance_floor=1e-6,
        pdf_floor=1e-12,
    ):
        self.X = ensure_2d(X)
        self.Y = np.asarray(Y).reshape(-1)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")
        if self.X.shape[0] < 5:
            raise ValueError("Regression source requires at least five samples.")

        self.learner = learner
        self.n_splits = max(2, min(n_splits, self.X.shape[0] - 1))
        self.variance_floor = variance_floor
        self.pdf_floor = pdf_floor
        self.log_variance_model = None
        self._log_var_constant = None

        self._fit_models()

    def _build_regressor(self):
        learner = (self.learner or 'gbm').lower()
        if learner == 'rf':
            return RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        return HistGradientBoostingRegressor(random_state=RANDOM_SEED)

    def _fit_models(self):
        base_mean = self._build_regressor()
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        oof_mu = np.zeros(self.X.shape[0])

        for train_idx, val_idx in cv.split(self.X):
            model = clone(base_mean)
            model.fit(self.X[train_idx], self.Y[train_idx])
            oof_mu[val_idx] = model.predict(self.X[val_idx])

        self.mean_model = clone(base_mean)
        self.mean_model.fit(self.X, self.Y)

        resid_sq = (self.Y - oof_mu) ** 2
        resid_sq = np.maximum(resid_sq, self.variance_floor)
        self._log_var_constant = float(np.log(np.mean(resid_sq)))

        if np.allclose(resid_sq, resid_sq[0], atol=1e-10):
            self.log_variance_model = None
        else:
            base_var = self._build_regressor()
            log_resid = np.log(resid_sq)
            self.log_variance_model = clone(base_var)
            self.log_variance_model.fit(self.X, log_resid)

    def _predict_log_variance(self, Xq):
        if self.log_variance_model is None:
            return np.full(Xq.shape[0], self._log_var_constant)
        log_var = self.log_variance_model.predict(Xq)
        min_log = np.log(self.variance_floor)
        return np.maximum(log_var, min_log)

    def predict_mu(self, x_query):
        Xq = ensure_2d(x_query)
        return self.mean_model.predict(Xq)

    def predict_sigma(self, x_query):
        Xq = ensure_2d(x_query)
        log_var = self._predict_log_variance(Xq)
        var = np.maximum(np.exp(log_var), self.variance_floor)
        return np.sqrt(var)

    def marginal_pdf_x(self, x_query):
        Xq = ensure_2d(x_query)
        return np.ones(Xq.shape[0])

    def conditional_pdf_y_given_x(self, x_query, y_vals):
        Xq = ensure_2d(x_query)
        y_vals = np.asarray(y_vals)
        mu = self.predict_mu(Xq)[:, None]
        sigma = self.predict_sigma(Xq)[:, None]
        sigma = np.maximum(sigma, 1e-12)
        z = (y_vals[None, :] - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self.pdf_floor, None)

    def joint_pdf(self, x_query, y_vals):
        return self.conditional_pdf_y_given_x(x_query, y_vals)

    def joint_pdf_xy(self, x_query, y_vals):
        return self.joint_pdf(x_query, y_vals)

    def joint_pdf_at_pairs(self, Xpairs, Ypairs):
        Xpairs = ensure_2d(Xpairs)
        Ypairs = np.asarray(Ypairs).reshape(-1)
        mu = self.predict_mu(Xpairs)
        sigma = self.predict_sigma(Xpairs)
        sigma = np.maximum(sigma, 1e-12)
        z = (Ypairs - mu) / sigma
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
        return np.clip(pdf, self.pdf_floor, None)


# --------------------------
# Pooled conditional estimators (pooled \hat p_data(y|x))
# --------------------------
class PooledConditionalClassifier:
    """Pooled estimator wrapper for $\hat p_{data}(y\mid x)$ (classification).

    This wrapper is used to compute $\hat p_{data}(y_i\mid x_i)$ at *paired* inputs.

    Supported usage patterns:
    1) Wrap a pretrained estimator via ``model=...``.
       - Torch head: a torch module mapping embeddings -> logits.
       - Scikit-learn estimator: any object exposing ``predict_proba`` and ``classes_``.
    2) Train a GBM-based pooled conditional model from data via ``(X, Y)``.
       - Training mirrors `SourceModelClassification` (GBM/RF + calibrated probabilities).

    Public API:
        - conditional_at_pairs(X, Y) -> p_data(y_i|x_i)
    """

    def __init__(
        self,
        X=None,
        Y=None,
        *,
        model=None,
        learner: str = 'gbm',
        calibration_method: str = 'isotonic',
        n_splits: int = 5,
        probability_floor: float = 1e-12,
        device: str = 'cpu',
        **_unused,
    ):
        self._prob_floor = float(probability_floor)

        self._backend = None
        self._torch_model = None
        self._sk_model = None
        self._class_to_col = None
        self._gbm_source = None

        # Case 1: wrap an existing estimator.
        if model is not None:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.device = torch.device(device)
                self._torch_model = model.to(self.device)
                self._torch_model.eval()
                self._backend = 'torch'
                return

            if hasattr(model, 'predict_proba'):
                self._sk_model = model
                classes = getattr(model, 'classes_', None)
                if classes is None:
                    raise ValueError("Sklearn pooled classifier must expose 'classes_' to map labels -> columns")
                classes = np.asarray(classes)
                self._class_to_col = {int(c): int(i) for i, c in enumerate(classes.tolist())}
                self._backend = 'sklearn'
                return

            raise ValueError(
                "Unsupported pooled classifier model; expected a torch.nn.Module or an sklearn-like estimator with predict_proba"
            )

        # Case 2: train a pooled GBM model from data.
        if X is None or Y is None:
            raise ValueError("Provide (X, Y) to train pooled GBM, or pass an existing model.")

        self._gbm_source = SourceModelClassification(
            X,
            Y,
            learner=learner,
            calibration_method=calibration_method,
            n_splits=int(n_splits),
            probability_floor=float(probability_floor),
        )
        self._backend = 'gbm'

    def _predict_proba_torch(self, X):
        if self._torch_model is None:
            raise RuntimeError("Torch pooled model is not initialized")
        Xq = ensure_2d(X).astype(np.float32)
        tx = torch.tensor(Xq, device=self.device)
        with torch.no_grad():
            out = self._torch_model(tx)
            if out.ndim != 2:
                raise ValueError("Torch pooled classifier model must return a 2D tensor (n, C)")
            probs = torch.softmax(out, dim=1)
        probs_np = probs.detach().cpu().numpy()
        probs_np = np.clip(probs_np, self._prob_floor, 1.0)
        probs_sum = np.maximum(probs_np.sum(axis=1, keepdims=True), 1e-12)
        return probs_np / probs_sum

    def conditional_at_pairs(self, X, Y):
        Xq = ensure_2d(X)
        Y_int = np.asarray(Y, dtype=int).reshape(-1)
        if Xq.shape[0] != Y_int.shape[0]:
            raise ValueError("X and Y must have the same number of rows for conditional_at_pairs")

        if self._backend == 'torch':
            probs = self._predict_proba_torch(Xq)
            rows = np.arange(Y_int.shape[0])
            valid = (Y_int >= 0) & (Y_int < probs.shape[1])
            out = np.full(Y_int.shape[0], self._prob_floor, dtype=float)
            if np.any(valid):
                out[valid] = probs[rows[valid], Y_int[valid]]
            return np.clip(out, self._prob_floor, 1.0)

        if self._backend == 'gbm':
            if self._gbm_source is None:
                raise RuntimeError("GBM pooled classifier is not initialized")
            return np.clip(self._gbm_source.joint_prob_at_pairs(Xq, Y_int), self._prob_floor, 1.0)

        if self._backend == 'sklearn':
            if self._sk_model is None or self._class_to_col is None:
                raise RuntimeError("Sklearn pooled classifier is not initialized")
            probs = np.asarray(self._sk_model.predict_proba(Xq), dtype=float)
            probs = np.clip(probs, self._prob_floor, 1.0)
            probs_sum = np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            probs = probs / probs_sum
            out = np.full(Y_int.shape[0], self._prob_floor, dtype=float)
            rows = np.arange(Y_int.shape[0])
            cols = np.array([self._class_to_col.get(int(y), -1) for y in Y_int], dtype=int)
            mask = cols >= 0
            if np.any(mask):
                out[mask] = probs[rows[mask], cols[mask]]
            return np.clip(out, self._prob_floor, 1.0)

        raise RuntimeError(f"Unknown pooled classifier backend: {self._backend}")


class PooledConditionalRegressionGaussian:
    """Pooled estimator wrapper for $\hat p_{data}(y\mid x)$ (Gaussian regression).

    Supported usage patterns:
    1) Wrap a pretrained estimator via ``model=...``.
       - Torch head: a torch module mapping embeddings -> (mu, scale).
       - Any object exposing ``joint_pdf_at_pairs(X, Y)`` (e.g. `SourceModelRegressionGaussian`).
    2) Train a GBM-based Gaussian conditional density model from data via ``(X, Y)``.
       - Training mirrors `SourceModelRegressionGaussian` (mean + heteroskedastic variance).

    Public API:
        - conditional_at_pairs(X, Y) -> p_data(y_i|x_i)
    """

    def __init__(
        self,
        X=None,
        Y=None,
        *,
        model=None,
        learner: str = 'gbm',
        n_splits: int = 5,
        variance_floor: float = 1e-6,
        pdf_floor: float = 1e-12,
        device: str = 'cpu',
        **_unused,
    ):
        self._var_floor = float(variance_floor)
        self._pdf_floor = float(pdf_floor)

        self._backend = None
        self._torch_model = None
        self._softplus = nn.Softplus() if TORCH_AVAILABLE else None
        self._gbm_source = None
        self._pair_pdf_model = None

        # Case 1: wrap an existing estimator.
        if model is not None:
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.device = torch.device(device)
                self._torch_model = model.to(self.device)
                self._torch_model.eval()
                self._backend = 'torch'
                return

            if hasattr(model, 'joint_pdf_at_pairs'):
                self._pair_pdf_model = model
                self._backend = 'pair_pdf'
                return

            raise ValueError(
                "Unsupported pooled regression model; expected a torch.nn.Module or an object with joint_pdf_at_pairs(X, Y)"
            )

        # Case 2: train a pooled GBM model from data.
        if X is None or Y is None:
            raise ValueError("Provide (X, Y) to train pooled GBM, or pass an existing model.")

        self._gbm_source = SourceModelRegressionGaussian(
            X,
            Y,
            learner=learner,
            n_splits=int(n_splits),
            variance_floor=float(variance_floor),
            pdf_floor=float(pdf_floor),
        )
        self._backend = 'gbm'

    def _predict_mu_sigma_torch(self, X):
        if self._torch_model is None or not TORCH_AVAILABLE:
            raise RuntimeError("Torch pooled regression model is not initialized")
        Xq = ensure_2d(X).astype(np.float32)
        tx = torch.tensor(Xq, device=self.device)
        with torch.no_grad():
            out = self._torch_model(tx)
            if hasattr(out, 'mean') and hasattr(out, 'scale'):
                mu_t = out.mean
                scale_t = out.scale
            elif isinstance(out, (tuple, list)) and len(out) >= 2:
                mu_t, scale_t = out[0], out[1]
            else:
                mu_t = out[:, 0]
                scale_t = out[:, 1]

            mu = mu_t.reshape(-1).detach().cpu().numpy()
            scale_raw = scale_t.reshape(-1)
            # If scale is already positive, keep it; otherwise treat as raw and apply softplus.
            if torch.any(scale_raw <= 0):
                if self._softplus is None:
                    raise RuntimeError("Torch softplus unavailable")
                scale_pos = self._softplus(scale_raw) + float(np.sqrt(self._var_floor))
            else:
                scale_pos = scale_raw
            sigma = scale_pos.detach().cpu().numpy()
        sigma = np.maximum(sigma, float(np.sqrt(self._var_floor)))
        return mu, sigma

    def conditional_at_pairs(self, X, Y):
        Xq = ensure_2d(X)
        Yq = np.asarray(Y, dtype=float).reshape(-1)
        if Xq.shape[0] != Yq.shape[0]:
            raise ValueError("X and Y must have the same number of rows for conditional_at_pairs")

        if self._backend == 'torch':
            mu, sigma = self._predict_mu_sigma_torch(Xq)
            sigma = np.maximum(sigma, float(np.sqrt(self._var_floor)))
            z = (Yq - mu) / sigma
            pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * z ** 2)
            return np.clip(pdf, self._pdf_floor, None)

        if self._backend == 'gbm':
            if self._gbm_source is None:
                raise RuntimeError("GBM pooled regression model is not initialized")
            return np.clip(self._gbm_source.joint_pdf_at_pairs(Xq, Yq), self._pdf_floor, None)

        if self._backend == 'pair_pdf':
            if self._pair_pdf_model is None:
                raise RuntimeError("Pair-pdf pooled regression model is not initialized")
            return np.clip(self._pair_pdf_model.joint_pdf_at_pairs(Xq, Yq), self._pdf_floor, None)

        raise RuntimeError(f"Unknown pooled regression backend: {self._backend}")


def build_pooled_conditional_model(task: str, X, Y, backend: str = 'gbm', **kwargs):
    """Factory for pooled conditional estimators.

    Parameters
    ----------
    task:
        'classification' or 'regression'
    backend:
        'gbm' (default).
        Note: evaluation scripts typically wrap pretrained Torch heads directly instead of calling this factory.
    kwargs:
        Forwarded to the underlying estimator constructors.
    """

    backend = (backend or 'gbm').lower()
    if backend != 'gbm':
        raise ValueError("Only backend='gbm' is supported for pooled conditional training.")

    if task == 'classification':
        return PooledConditionalClassifier(X, Y, **kwargs)
    elif task == 'regression':
        return PooledConditionalRegressionGaussian(X, Y, **kwargs)
    else:
        raise ValueError(f"Unsupported task for pooled conditional model: {task}")


# ---------------------------------------------------------------------
# Backward-compatible aliases (legacy names used by older evaluation code)
# ---------------------------------------------------------------------
PooledConditionalClassifierNN = PooledConditionalClassifier
PooledConditionalRegressionGaussianNN = PooledConditionalRegressionGaussian


# --------------------------
# Lambda parameterizations
# --------------------------
class LambdaSpline:
    r"""
    lambda_j(x) = softplus( Phi(x) @ Theta.T ), Theta shape (K, m)
    Fit Theta by minimizing empirical marginal objective with scipy.minimize.
    
    The marginal loss function is:
    Phi_marg^(n)(lambda(·)) := (1-alpha) ∫_X Σ_j lambda_j(x) d_nv(x) + ∫_X ∫_Y (1 - h_lambda^(n)(x,y))_- d_mv(y) d_nv(x)
    where h_lambda^(n)(x,y) = Σ_j lambda_j(x) \hat{f}_j^(n)(x,y)
    
    We approximate the integrals using the empirical distribution on the training set.
    """
    def __init__(self, X_train, K, n_splines=5, degree=3, include_bias=True, diff_order=2,
                 gamma1=0.01, gamma2=0.01, gamma3=0):
        self.X_train = ensure_2d(X_train)
        self.K = K  # number of sources
        self.spline = SplineTransformer(n_knots=n_splines, degree=degree, include_bias=include_bias)
        self.Phi = self.spline.fit_transform(self.X_train)  # (n, m), m depends on n_knots and degree
        self.n, self.m = self.Phi.shape
        
        # Initialize Theta
        rng = np.random.default_rng(RANDOM_SEED)
        self.Theta = rng.standard_normal((self.K, self.m))

        diff_order = diff_order  # difference order for smoothness penalty
        m = self.m
        if diff_order == 1:
            # D shape (m-1, m)
            D = np.zeros((m-1, m))
            for i in range(m-1):
                D[i, i] = -1.0
                D[i, i+1] = 1.0
        else:
            # second difference: D shape (m-2, m)
            D = np.zeros((m-2, m))
            for i in range(m-2):
                D[i, i] = 1.0
                D[i, i+1] = -2.0
                D[i, i+2] = 1.0
        self._D = D
        self._gamma1 = gamma1  # L2 penalty on lambda
        self._gamma2 = gamma2  # smoothness penalty on Theta
        self._gamma3 = gamma3  # penalty on h_lambda values

    def lambda_at_Phi(self, Phi):
        A = Phi @ self.Theta.T  # (q, K)
        return softplus_np(A)   # (q, K)

    def lambda_at_x(self, x_query):
        Xq = ensure_2d(x_query)
        Phi_q = self.spline.transform(Xq)
        return self.lambda_at_Phi(Phi_q)

    def objective_flat_marginal(self, theta_flat, F_train_joint, X_train, Y_train, alpha, task='regression', 
                                source_weights=None):
        r"""
        Empirical marginal objective function:
        \hat{Phi}_marg(λ(·)) := (1/N) Σ_i [(1 - h_λ(X_i,Y_i))_- / \hat{p}_data(Y_i|X_i)] + (1-alpha) (1/N) Σ_i Σ_j λ_j(X_i)
        
        where \hat{p}_data(y|x) = Σ_k \hat{p}_k \hat{f}_k(x,y) / Σ_k \hat{p}_k \hat{f}_k(x) is the conditional density under the mixture
        
        :param F_train_joint: (n, K) with F_{i,j} = f_j(x_i, y_i) (joint densities)
        :param X_train,Y_train: training data
        :param source_weights: (K,) array of \hat{p}_k = n_k/N weights, if None assumes equal weights
        """
        K, m = self.K, self.m
        n = len(X_train)
        Theta = theta_flat.reshape((K, m))
        A = self.Phi @ Theta.T
        Lambda = softplus_np(A)  # (n,K)
        
        # Set default equal weights if not provided
        if source_weights is None:
            source_weights = np.ones(K) / K
        
        # Compute mixture densities for conditional p_data(y|x)
        # \hat{p}_data(x_i, y_i) = Σ_k \hat{p}_k \hat{f}_k(x_i, y_i)
        p_data_joint = np.sum(source_weights[None, :] * F_train_joint, axis=1)  # (n,)
        
        # For \hat{p}_data(x_i), we need marginal densities from each source
        # This requires computing \hat{f}_k(x_i) for each source k and point i
        # We'll compute this by integrating over y or using the marginal density methods
        p_data_marginal = np.zeros(n)
        for i in range(n):
            x_i = X_train[i:i+1]  # reshape to (1, d)
            marginal_i = 0.0
            for k in range(K):
                # Get marginal density f_k(x_i) from source k
                if hasattr(self, '_sources') and self._sources is not None:
                    src = self._sources[k]
                    if task == 'classification':
                        # For classification: use f_x() method from new SourceModelClassification
                        marginal_k = src.f_x(x_i)[0]
                    else:
                        # For regression: use the marginal PDF method
                        marginal_k = src.marginal_pdf_x(x_i)[0]
                else:
                    # Fallback: estimate marginal from available joint densities
                    # This is an approximation, if we'd have access to source models
                    marginal_k = F_train_joint[i, k] / max(1e-6, np.mean(F_train_joint[:, k]))
                
                marginal_i += source_weights[k] * marginal_k
            p_data_marginal[i] = max(marginal_i, 1e-6)
        
        # Conditional density \hat{p}_data(y_i|x_i) = \hat{p}_data(x_i, y_i) / \hat{p}_data(x_i)
        p_data_conditional = p_data_joint / p_data_marginal
        p_data_conditional = np.maximum(p_data_conditional, 1e-6)  # numerical stability
        
        # Compute h_λ(x_i, y_i) = Σ_j λ_j(x_i) * \hat{f}_j(x_i, y_i)
        h_values = np.sum(Lambda * F_train_joint, axis=1)  # (n,)
        
        # First term: (1/N) Σ_i [(1 - h_λ(X_i,Y_i))_- / \hat{p}_data(Y_i|X_i)]
        hinge_loss = np.minimum(1.0 - h_values, 0.0)   # corrected sign
        term1 = np.mean(hinge_loss / p_data_conditional)
        
        # Second term: (1-alpha) (1/N) Σ_i Σ_j λ_j(X_i)
        term2 = (1.0 - alpha) * np.mean(np.sum(Lambda, axis=1))
        
        # Penalty term
        l2_pen = self._gamma1 * np.mean(np.sum(Lambda**2, axis=1))
        smooth_pen = self._gamma2 * sum(np.sum((self._D @ Theta[j])**2) for j in range(K))
        h_pen = self._gamma3 * np.mean(h_values**2)

        # minus penalty, since we maximize the dual (what we return)
        return term1 + term2 - l2_pen - smooth_pen - h_pen

    def fit_torch(self, sources, X_pool, Y_pool, alpha=0.1,
                epochs=10000, batch_size=256, lr=1e-3,
                device='cpu', source_weights=None, verbose=False,
                tol=1e-4, p_data_conditional=None, p_data_floor=1e-8):
        """
        PyTorch-based vectorized fit for LambdaSpline.
        Replaces the scipy.minimize pipeline with minibatch Adam + autograd.
        Updates self.Theta (shape K x m) in-place.

        Relies on:
        - self.Phi (n x m) precomputed in __init__
        - self._D (r x m) difference matrix for smoothness
        - self._gamma1, _gamma2, _gamma3 penalties
        - sources: list of source models (used only to compute marginals)
        - build_F_train_joint_from_sources(...) existing helper to build joint densities.

        Early stopping:
        - Stops when relative change in the full objective falls below ``tol`` (default 1e-4).
        """
        import torch
        import torch.nn.functional as F
        device = torch.device(device)

        # store sources for marginal density computation
        self._sources = sources
        task = infer_task_from_sources(sources)
        n, m = self.Phi.shape
        K = self.K

        # Build F_train_joint like the original fit (n x K numpy)
        F_train_joint = build_F_train_joint_from_sources(sources, X_pool, Y_pool, task)  # numpy
        F_np = np.asarray(F_train_joint, dtype=np.float32)  # (n, K)

        # Source weights (default equal) used only when pooled conditionals are not provided
        if source_weights is None:
            source_weights = np.ones(K, dtype=np.float32) / float(K)
            if verbose:
                print("Using equal source weights. Consider providing source_weights for better accuracy.")
        source_weights = np.asarray(source_weights, dtype=np.float32)

        # Precompute p_data(y|x)
        if p_data_conditional is not None:
            p_data_conditional_np = np.asarray(p_data_conditional, dtype=np.float32).reshape(-1)
            if p_data_conditional_np.shape[0] != n:
                raise ValueError("p_data_conditional length does not match X_pool")
            p_data_conditional_np = np.maximum(p_data_conditional_np, p_data_floor)
        else:
            raise ValueError("Please compute p_data(y|x) over pooled data externally and provide it to fit().")
            # # fallback: mixture over sources
            # p_data_joint_np = (F_np * source_weights[None, :]).sum(axis=1)  # (n,)
            # p_data_marginal_np = np.zeros(n, dtype=np.float32)
            # for k, src in enumerate(sources):
            #     if task == 'classification':
            #         marg_k = src.f_x(X_pool).astype(np.float32)
            #     else:
            #         marg_k = src.marginal_pdf_x(X_pool).astype(np.float32)
            #     p_data_marginal_np += source_weights[k] * marg_k
            # p_data_marginal_np = np.maximum(p_data_marginal_np, p_data_floor)
            # p_data_conditional_np = np.maximum(p_data_joint_np / p_data_marginal_np, p_data_floor)

        # Move tensors to torch
        Phi_t = torch.tensor(self.Phi.astype(np.float32), device=device)           # (n, m)
        F_t = torch.tensor(F_np, device=device)                                    # (n, K)
        p_data_cond_t = torch.tensor(p_data_conditional_np.astype(np.float32), device=device)  # (n,)
        DT = torch.tensor(self._D.astype(np.float32), device=device)               # (r, m)

        # Initialize Theta as a torch parameter (start from existing self.Theta if available)
        theta_init = np.array(self.Theta, dtype=np.float32) if hasattr(self, 'Theta') else np.zeros((K, m), dtype=np.float32)
        Theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)  # (K, m)

        # Optimizer
        optimizer = torch.optim.Adam([Theta], lr=lr)

        # Training loop with minibatches
        n_idx = n
        indices = np.arange(n_idx)
        steps_per_epoch = max(1, int(np.ceil(n_idx / batch_size)))

        prev_obj = None
        for epoch in range(epochs):
            # shuffle each epoch
            np.random.shuffle(indices)
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min((step + 1) * batch_size, n_idx)
                if start >= end:
                    break
                batch_idx = indices[start:end]
                b = len(batch_idx)

                # batch tensors
                Phi_b = Phi_t[batch_idx, :]            # (b, m)
                F_b = F_t[batch_idx, :]                # (b, K)
                pcond_b = p_data_cond_t[batch_idx]     # (b,)

                # compute A = Phi_b @ Theta.T  -> (b, K)
                A = Phi_b @ Theta.T
                # nonnegativity via softplus (matches original param)
                Lambda = F.softplus(A)                 # (b, K)

                # compute h = sum_j lambda_j(x_i) * f_j(x_i, y_i)
                h = (Lambda * F_b).sum(dim=1)          # (b,)

                # hinge: (1 - h)_- == min(1-h, 0) == -relu(h-1)
                hinge = -F.relu(h - 1.0)               # (b,)

                # term1: mean over batch of hinge / p_data_conditional
                term1 = (hinge / pcond_b).mean()

                # term2: (1-alpha) * mean_i sum_j lambda_j(x_i)
                term2 = (1.0 - alpha) * Lambda.sum(dim=1).mean()

                # penalties
                l2_pen = self._gamma1 * (Lambda**2).sum(dim=1).mean()

                # smooth penalty: sum_j ||D @ Theta_j||^2
                # DT @ Theta.T -> (r, K), then square and sum
                DT_Theta = DT @ Theta.T                 # (r, K)
                smooth_pen = self._gamma2 * (DT_Theta**2).sum()

                # penalty on h
                h_pen = self._gamma3 * (h**2).mean()

                # objective (original returns term1 + term2 - penalties) -> we want to maximize that,
                # so we minimize negative of it
                obj = term1 + term2 - l2_pen - smooth_pen - h_pen
                loss = -obj

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().numpy()) * b

            # average loss for epoch
            epoch_loss = epoch_loss / n_idx
            with torch.no_grad():
                A_full = Phi_t @ Theta.T
                Lambda_full = F.softplus(A_full)
                h_full = (Lambda_full * F_t).sum(dim=1)
                hinge_full = -F.relu(h_full - 1.0)
                term1_full = (hinge_full / p_data_cond_t).mean()
                term2_full = (1.0 - alpha) * Lambda_full.sum(dim=1).mean()
                DT_Theta_full = DT @ Theta.T
                smooth_full = self._gamma2 * (DT_Theta_full**2).sum()
                l2_full = self._gamma1 * (Lambda_full**2).sum(dim=1).mean()
                h_pen_full = self._gamma3 * (h_full**2).mean()
                obj_full = term1_full + term2_full - l2_full - smooth_full - h_pen_full
                obj_value = obj_full.item()

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"[Torch fit] epoch {epoch+1}/{epochs}  loss={epoch_loss:.6g}  obj={obj_value:.6g}")

            if tol is not None and prev_obj is not None:
                denom = max(1e-8, abs(prev_obj))  # turn to abs_tol = 1e-12 when objective value is small
                rel_change = abs(obj_value - prev_obj) / denom
                if rel_change < tol:
                    if verbose:
                        print(f"[Torch fit] early stopping at epoch {epoch+1} (rel_change={rel_change:.3g})")
                    break

            prev_obj = obj_value

        # write Theta back to numpy storage like original class expects
        self.Theta = Theta.detach().cpu().numpy().reshape((K, m))

        res = {
            'success': True,
            'message': 'fit_torch completed',
            'theta': self.Theta
        }
        return res


class LambdaNN:
    """
    Neural net parameterization for lambda(x). Uses PyTorch.
    """
    def __init__(
        self,
        X_train,
        K,
        hidden_sizes=(8, 8),
        activation='relu',
        device='cpu',
        *,
        pca_dim=None,
        pca_whiten: bool = False,
        pca_random_state: int = RANDOM_SEED,
        pca_state=None,
    ):
        '''
        :param X_train: (n, d) array of training features
        :param K: number of sources (distributions)
        :param hidden_sizes: tuple of hidden layer sizes
        '''
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available. Install torch to use LambdaNN.")
        self.device = device
        self.X_train = ensure_2d(X_train)
        d = self.X_train.shape[1]

        # Optional PCA projection (used only by the LambdaNN network).
        # Important: this DOES NOT change the feature space used by the source models / pooled head.
        self._use_pca = False
        self._pca_dim = None
        self._pca_mean = None
        self._pca_components = None
        self._pca_whiten = bool(pca_whiten)
        self._pca_scale = None

        if pca_state is not None:
            mean = np.asarray(pca_state.get('mean'), dtype=np.float32).reshape(-1)
            components = np.asarray(pca_state.get('components'), dtype=np.float32)
            whiten_flag = bool(pca_state.get('whiten', self._pca_whiten))
            explained_variance = pca_state.get('explained_variance')
            if components.ndim != 2:
                raise ValueError("pca_state['components'] must be a 2D array")
            if mean.shape[0] != components.shape[1]:
                raise ValueError(
                    "pca_state mean/components dimension mismatch: "
                    f"mean={mean.shape}, components={components.shape}"
                )
            self._use_pca = True
            self._pca_dim = int(components.shape[0])
            self._pca_mean = mean
            self._pca_components = components
            self._pca_whiten = whiten_flag
            if self._pca_whiten:
                if explained_variance is None:
                    raise ValueError("pca_state must include 'explained_variance' when whiten=True")
                ev = np.asarray(explained_variance, dtype=np.float32).reshape(-1)
                if ev.shape[0] != self._pca_dim:
                    raise ValueError(
                        "pca_state explained_variance length mismatch: "
                        f"expected {self._pca_dim}, got {ev.shape[0]}"
                    )
                self._pca_scale = np.sqrt(np.maximum(ev, 1e-12))
        elif pca_dim is not None and int(pca_dim) > 0:
            k = int(pca_dim)
            n = int(self.X_train.shape[0])
            max_k = int(min(n, d))
            if k > max_k:
                raise ValueError(
                    f"pca_dim={k} is too large for training data (n={n}, d={d}); "
                    f"must be <= min(n, d) = {max_k}."
                )
            if k == d:
                # Explicitly avoid fitting PCA when it would be an identity transform.
                self._use_pca = False
            else:
                pca = PCA(
                    n_components=k,
                    whiten=bool(pca_whiten),
                    random_state=int(pca_random_state),
                    svd_solver='auto',
                )
                pca.fit(self.X_train.astype(np.float32))
                self._use_pca = True
                self._pca_dim = int(k)
                self._pca_mean = pca.mean_.astype(np.float32)
                self._pca_components = pca.components_.astype(np.float32)
                self._pca_whiten = bool(pca_whiten)
                if self._pca_whiten:
                    self._pca_scale = np.sqrt(np.maximum(pca.explained_variance_.astype(np.float32), 1e-12))

        X_train_nn = self._transform_features(self.X_train)
        d_nn = int(X_train_nn.shape[1])
        layers = []
        in_dim = d_nn
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU() if activation=='relu' else nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, K))
        self.model = nn.Sequential(*layers).to(device)
        self.K = K

    def _transform_features(self, X):
        X2 = ensure_2d(X).astype(np.float32)
        if not self._use_pca:
            return X2
        if self._pca_mean is None or self._pca_components is None:
            raise RuntimeError("PCA is enabled but PCA parameters are missing")
        Z = (X2 - self._pca_mean[None, :]) @ self._pca_components.T
        if self._pca_whiten:
            if self._pca_scale is None:
                raise RuntimeError("PCA whitening requested but scale parameters are missing")
            Z = Z / self._pca_scale[None, :]
        return Z.astype(np.float32)

    def lambda_at_x(self, x_query):
        '''
        Dimensions:
        - x_query: (q, d) array of query points
        - Returns (q, K) array of lambda_j(x_query) values
        '''
        self.model.eval()
        Xq = self._transform_features(x_query)
        tx = torch.tensor(Xq, device=self.device)
        with torch.no_grad():
            out = self.model(tx).cpu().numpy()
        return softplus_np(out)  # (q,K)

    def fit(
        self,
        sources,
        X_pool,
        Y_pool,
        alpha=0.1,
        epochs=200,
        lr=1e-3,
        batch_size=None,
        *,
        gamma1: float = 0.0,
        verbose=False,
        source_weights=None,
        p_data_model=None,
        p_data_conditional=None,
        p_data_floor=1e-8,
    ):
        """
        Fit using the marginal objective

        \hat{\Phi}_{\mathrm{marg}}(\lambda(\cdot)) =
            E\big[(1-h_\lambda)_- / \hat p_{data}(y|x)\big]
            + (1-\alpha) E[\sum_j \lambda_j(x)]
            - \gamma_1 E[\sum_j \lambda_j(x)^2]

        If ``p_data_model`` or ``p_data_conditional`` is provided, they are used to compute
        \hat p_{data}(y|x); otherwise the previous mixture-from-sources approximation is used.
        """
        task = infer_task_from_sources(sources)
        F_train_joint = build_F_train_joint_from_sources(sources, X_pool, Y_pool, task)

        # # Source weights only needed for the mixture fallback
        # if source_weights is None:
        #     source_weights = np.ones(self.K) / self.K
        #     if verbose:
        #         print("Using equal source weights (mixture fallback). Provide source_weights or p_data_model for custom weighting.")
        # source_weights = np.asarray(source_weights, dtype=np.float32)

        # Compute p_data(y|x)
        if p_data_model is not None:
            p_data_conditional_np = np.asarray(p_data_model.conditional_at_pairs(X_pool, Y_pool), dtype=np.float32)
        elif p_data_conditional is not None:
            p_data_conditional_np = np.asarray(p_data_conditional, dtype=np.float32).reshape(-1)
            if p_data_conditional_np.shape[0] != len(X_pool):
                raise ValueError("p_data_conditional length does not match X_pool")
        else:
            raise ValueError("Please compute p_data(y|x) over pooled data externally and provide it to fit().")
            # # fallback: mixture over sources
            # if task == 'classification':
            #     p_data_joint = (F_train_joint * source_weights[None, :]).sum(axis=1)
            #     p_data_marginal = np.zeros(len(X_pool), dtype=np.float32)
            #     for k, src in enumerate(sources):
            #         p_data_marginal += source_weights[k] * src.f_x(X_pool)
            #     p_data_conditional_np = p_data_joint / np.maximum(p_data_marginal, p_data_floor)
            # else:
            #     p_data_joint = (F_train_joint * source_weights[None, :]).sum(axis=1)
            #     p_data_marginal = np.zeros(len(X_pool), dtype=np.float32)
            #     for k, src in enumerate(sources):
            #         p_data_marginal += source_weights[k] * src.marginal_pdf_x(X_pool)
            #     p_data_conditional_np = p_data_joint / np.maximum(p_data_marginal, p_data_floor)

        p_data_conditional_np = np.maximum(p_data_conditional_np, p_data_floor).astype(np.float32)

        X_train_nn = self._transform_features(self.X_train)
        X = torch.tensor(X_train_nn, device=self.device)
        F = torch.tensor(F_train_joint.astype(np.float32), device=self.device)
        p_data_conditional_t = torch.tensor(p_data_conditional_np, device=self.device)

        gamma1 = float(gamma1)

        n = X.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        bsize = batch_size or n

        for ep in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            for start in range(0, n, bsize):
                idx = perm[start:start + bsize]
                xb = X[idx]
                fb = F[idx]
                p_cond_b = p_data_conditional_t[idx]

                optimizer.zero_grad()
                out = self.model(xb)
                lam = torch.nn.functional.softplus(out)

                h_vals = torch.sum(lam * fb, dim=1)
                hinge = -torch.relu(h_vals - 1.0)  # (1 - h)_-
                term1 = torch.mean(hinge / torch.clamp(p_cond_b, min=p_data_floor))
                term2 = (1.0 - alpha) * torch.mean(torch.sum(lam, dim=1))

                l2_pen = gamma1 * torch.mean(torch.sum(lam ** 2, dim=1))

                obj = term1 + term2 - l2_pen
                loss = -obj
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)

            if verbose and ep % max(1, epochs // 10) == 0:
                print(f"Epoch {ep}/{epochs}, avg loss {total_loss / n:.6f}")
        return

    def _compute_marginal_densities(self, X_pool, sources, source_weights, task):
        r"""
        Compute marginal densities \hat{p}_data(x_i) = Σ_k \hat{p}_k \hat{f}_k(x_i) for all training points.
        """
        n = len(X_pool)
        p_data_marginal = np.zeros(n)
        
        for i in range(n):
            x_i = ensure_2d_query(X_pool[i])  # reshape to (1, d)
            marginal_i = 0.0
            for k, src in enumerate(sources):
                if task == 'classification':
                    # For classification: use f_x() method from new SourceModelClassification
                    marginal_k = src.f_x(x_i)[0]
                else:
                    # For regression: use the marginal PDF method
                    marginal_k = src.marginal_pdf_x(x_i)[0]
                
                marginal_i += source_weights[k] * marginal_k
            p_data_marginal[i] = max(marginal_i, 1e-6)
        
        return p_data_marginal


# --------------------------
# Training pipeline for multi-source aggregation
# --------------------------
def compute_source_weights_from_sizes(source_sizes):
    r"""
    Compute source weights \hat{p}_k = n_k/N from individual source sample sizes.
    
    :param source_sizes: list or array of sample sizes for each source
    :Returns array of normalized weights summing to 1
    """
    source_sizes = np.asarray(source_sizes)
    total_size = np.sum(source_sizes)
    return source_sizes / total_size

def build_F_train_joint_from_sources(sources, X_pool, Y_pool, task):
    """
    Build training matrix with joint densities f_j(x_i, y_i).
    
    :param sources: list of source-model objects (length K)
    :param X_pool,Y_pool: pooled dataset (n, d) and (n,)
    :param task: 'classification' or 'regression'

    :return F_train_joint: shape (n, K) with F_{i,j} = f_j(x_i, y_i)
    """
    n = X_pool.shape[0]
    K = len(sources)
    F = np.zeros((n, K))
    for j, src in enumerate(sources):
        if task == 'classification':
            F[:, j] = src.joint_prob_at_pairs(X_pool, Y_pool)
        else:
            F[:, j] = src.joint_pdf_at_pairs(X_pool, Y_pool)
    return F

def fit_lambda_from_sources(sources, lambda_mode, X_pool, Y_pool,
                            alpha=0.1, spline_kwargs=None, nn_kwargs=None,
                            solver_kwargs=None, verbose=False, source_weights=None,
                            use_torch=True, p_data_model=None, p_data_conditional=None,
                            p_data_floor=1e-8):
    r"""
    Fit lambda(x) parameterization using pooled training data from all sources with marginal loss.

    :param sources: list of per-source SourceModelClassification or SourceModelRegressionGaussian
    :param lambda_mode: 'spline' or 'nn'
    :param source_weights: (K,) array of \hat{p}_k = n_k/N weights for mixture density
    :param spline_kwargs: dict of kwargs for LambdaSpline

    :Returns an object with interface: lambda_at_x(x) -> (q, K) array

    spline_kwargs keys:
    - n_splines: int = 5,
    - degree: int = 3,
    - include_bias: bool = True,
    - diff_order: int = 2,
    - gamma1: float = 0.01, # L2 penalty on lambda
    - gamma2: float = 0.01, # smoothness penalty on Theta
    - gamma3: float = 0, # penalty on h_lambda values
    """
    X_pool = ensure_2d(X_pool)
    n = X_pool.shape[0]
    K = len(sources)

    if lambda_mode == 'spline':
        spline_kwargs = spline_kwargs or {}
        lam = LambdaSpline(X_pool, K, **spline_kwargs)
        solver_kwargs = solver_kwargs or {}
        solver_kwargs['source_weights'] = source_weights

        # if user asked for torch and torch is available, call the PyTorch fit
        if use_torch:
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch requested (use_torch=True) but TORCH_AVAILABLE is False")
            # fit_torch updates self.Theta in-place and returns a dict-like result;
            # keep the function behavior of returning the lambda object.
            lam.fit_torch(
                sources,
                X_pool,
                Y_pool,
                alpha=alpha,
                verbose=verbose,
                p_data_conditional=p_data_conditional,
                p_data_floor=p_data_floor,
                **solver_kwargs,
            )
            return lam
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("Torch not available. Install torch or set use_torch=False to use scipy minimize.")
            else:
                raise RuntimeError("use_torch=False is not supported when torch is available. Set use_torch=True.")
    elif lambda_mode == 'nn':
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required for 'nn' lambda_mode")
        nn_kwargs = nn_kwargs or {}
        lam = LambdaNN(X_pool, K, **nn_kwargs)
        solver_kwargs = solver_kwargs or {}
        solver_kwargs['source_weights'] = source_weights
        lam.fit(
            sources,
            X_pool,
            Y_pool,
            alpha=alpha,
            verbose=verbose,
            p_data_model=p_data_model,
            p_data_conditional=p_data_conditional,
            p_data_floor=p_data_floor,
            **solver_kwargs,
        )
        return lam
    else:
        raise ValueError("lambda_mode must be 'spline' or 'nn'")


# --------------------------
# Construct aggregated conformal set for a new x
# --------------------------
def precompute_calibration_cache(
    lam_model,
    sources,
    X_cal_list,
    Y_cal_list,
):
    """Precompute per-source log h values on calibration data for reuse.

    Parameters
    ----------
    lam_model : callable
        Function mapping an array of inputs to lambda weights of shape (n_samples, K).
    sources : list
        List of source models compatible with MDCP.
    X_cal_list, Y_cal_list : list
        Calibration covariates/targets per source.

    Returns
    -------
    dict
        Mapping ``source_index -> {"log_h_cal": np.ndarray, "log_h_cal_sorted": np.ndarray, "n_j": int}``.
    """

    if lam_model is None:
        raise ValueError("lam_model must be provided to precompute calibration cache.")

    K = len(sources)
    calibration_cache = {}

    for j in range(K):
        X_cal_j = np.asarray(X_cal_list[j])
        Y_cal_j = np.asarray(Y_cal_list[j])

        if X_cal_j.ndim == 1:
            X_cal_j = X_cal_j.reshape(-1, X_cal_j.size)

        n_j = X_cal_j.shape[0]
        if n_j <= 0:
            raise ValueError(f"Calibration set for source {j} is empty.")

        lam_cal_all = lam_model(X_cal_j)
        if lam_cal_all.ndim == 1:
            lam_cal_all = lam_cal_all.reshape(n_j, -1)
        if lam_cal_all.shape[0] == 1 and n_j > 1:
            lam_cal_all = np.tile(lam_cal_all, (n_j, 1))
        if lam_cal_all.shape[0] != n_j:
            raise ValueError("lam_model returned unexpected number of rows for calibration cache.")
        if lam_cal_all.shape[1] != K:
            raise ValueError("lam_model returned unexpected number of columns for calibration cache.")

        joint_calib = np.zeros((n_j, K), dtype=float)
        for k, src_k in enumerate(sources):
            if hasattr(src_k, 'joint_pdf_at_pairs'):
                joint_calib[:, k] = src_k.joint_pdf_at_pairs(X_cal_j, Y_cal_j)
            elif hasattr(src_k, 'joint_prob_at_pairs'):
                joint_calib[:, k] = src_k.joint_prob_at_pairs(X_cal_j, Y_cal_j)
            else:
                raise AttributeError(
                    f"Source {k} must have either joint_pdf_at_pairs or joint_prob_at_pairs method"
                )

        h_cal = np.sum(lam_cal_all * joint_calib, axis=1)
        log_h_cal = np.log(np.maximum(h_cal, 1e-100))
        log_h_cal_sorted = np.sort(log_h_cal)

        calibration_cache[j] = {
            "log_h_cal": log_h_cal,
            "log_h_cal_sorted": log_h_cal_sorted,
            "n_j": n_j,
        }

    return calibration_cache


def aggregated_conformal_set_multi(
    lam_model,               # callable: lam_model(X) -> array (n_samples, K) OR None
    sources,                 # list of source objects; each must implement joint_pdf(X, y_grid) for regression or joint_prob(X, y_grid) for classification -> shape (n, m)
    X_cal_list,              # list of np.arrays, one per source j: X_cal_list[j].shape = (n_j, d)
    Y_cal_list,              # list of np.arrays, one per source j: Y_cal_list[j].shape = (n_j,)
    X_test,                  # single test covariate (shape (d,) or (1,d))
    Y_test,                  # true label for test point (for coverage)
    y_grid,                  # 1D array-like of candidate y values (length m)
    alpha=0.1,               # target miscoverage level
    randomize_ties=True,     # whether to randomized-break ties in empirical p-value
    calibration_cache=None,  # optional dict from precompute_calibration_cache
    lam_x=None,              # optional precomputed lambda values at X_test (shape (K,))
    rng=None,                # optional np.random.Generator for tie-breaking
):
    """
    Compute aggregated conformal set via per-source p-values that use the FULL combined score h(y).
    
    Returns
    -------
    p_values_y_grid : np.array shape (K, m) with p_j(y) computed using calibration samples from source j
    
    union_mask_y_grid : boolean array shape (m,) where union_mask_y_grid[idx] == True iff max_j p_j(y) > alpha

    h_y_grid : np.array shape (m,) = h(y) evaluated at X_test

    h_true : float = h(X_test, Y_test) value for true label (scalar for single test point)

    p_values_true_y : np.array shape (K,) with p_j(Y_test) for true label Y_test (single test point)

    union_mask_true_y : boolean, whether true label is in the aggregated conformal set (scalar for single test point)

    Notes
    -----
    - Each source object in `sources` must provide joint_pdf(X, y_grid) for regression or joint_prob(X, y_grid) for classification, returning shape (n, m).
    
    - If lam_model is provided, it should be callable: lam_model(X) -> (n_samples, K).
        If lam_model is None, `lam` must be provided and is used as a constant vector across x.

        - Supplying ``calibration_cache`` (from :func:`precompute_calibration_cache`) skips recomputation of
            the calibration statistics for every test point.

        - ``lam_x`` can be used to pass precomputed lambda values for the current test point to avoid an
            extra call to ``lam_model``.

        - ``rng`` allows sharing a random generator across calls for reproducible tie-breaking without
            repeatedly reinitializing the default generator.
    
    - p_j(y) = EmpiricalProb_{(X,Y) in cal_j} [ h(X, Y) <= h(X_test, y) ]
        with optional randomized tie-breaking for equality.
    """
    K = len(sources)
    y_grid = np.asarray(y_grid)
    m = y_grid.size

    if lam_model is None and lam_x is None:
        raise ValueError("Either lam_model or lam_x must be provided.")

    # Make sure X_test is 2D for calling joint_pdf uniformly (single test point)
    X_test_arr = np.asarray(X_test)
    Y_test_scalar = np.asarray(Y_test)
    
    # Ensure X_test is single test point reshaped to (1, d)
    if X_test_arr.ndim == 1:
        X_test_2d = X_test_arr.reshape(1, -1)
    elif X_test_arr.ndim == 2 and X_test_arr.shape[0] == 1:
        X_test_2d = X_test_arr
    else:
        raise ValueError(f"X_test must be a single test point, got shape {X_test_arr.shape}")
    
    # Ensure Y_test is scalar
    if Y_test_scalar.ndim > 0 and Y_test_scalar.size == 1:
        Y_test_scalar = Y_test_scalar.item()
    elif Y_test_scalar.ndim == 0:
        Y_test_scalar = Y_test_scalar.item()
    else:
        raise ValueError(f"Y_test must be a single scalar value, got shape {Y_test_scalar.shape}")

    # -- Compute f_grid_test: per-source densities at (X_test, y_grid)
    # f_grid_test shape: (K, m)
    f_grid_test = np.zeros((K, m), dtype=float)
    joint_true = np.zeros(K, dtype=float)  # Single test point, shape (K,)
    y_grid_with_true = np.concatenate([y_grid, np.array([Y_test_scalar])])
    for k, src in enumerate(sources):
        # Handle both classification (joint_prob) and regression (joint_pdf) sources
        if hasattr(src, 'joint_pdf'):
            vals = src.joint_pdf(X_test_2d, y_grid_with_true)  # shape (1, m+1) expected
        elif hasattr(src, 'joint_prob'):
            vals = src.joint_prob(X_test_2d, y_grid_with_true)  # shape (1, m+1) expected
        else:
            raise AttributeError(f"Source {k} must have either joint_pdf or joint_prob method")

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        f_grid_test[k, :] = vals[0, :m]
        joint_true[k] = vals[0, m]

    # -- Compute lam at X_test (1, K) then h_y_grid (m,)
    if lam_x is not None:
        lam_test = np.asarray(lam_x)
        if lam_test.ndim > 1:
            lam_test = lam_test.reshape(-1)
        if lam_test.shape[0] != K:
            raise ValueError("lam_x must have length equal to number of sources K.")
    else:
        lam_test = lam_model(X_test_2d)  # shape (1, K)
        if lam_test.ndim == 1:
            lam_test = lam_test.reshape(1, -1)
        lam_test = lam_test[0]  # Extract to shape (K,)
    
    # h at X_test: h(y) = sum_k lam_k(X_test) * f_k(X_test, y)
    h_y_grid = np.dot(lam_test, f_grid_test)  # shape (m,)
    h_true = np.sum(lam_test * joint_true)    # scalar for single test point

    # Prepare output container: p-values per source and per y
    p_values_y_grid = np.zeros((K, m), dtype=float)
    p_values_true_y = np.zeros(K, dtype=float)  # Single test point, so shape (K,)

    # Precompute log h values once
    log_h_test = np.log(np.maximum(h_y_grid, 1e-100))
    log_h_true_scalar = np.log(np.maximum(h_true, 1e-100))

    # RNG for randomized ties
    rng_local = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
    atol = 1e-6
    rtol = 1e-4

    # -- For each source j, compute p_j(y) using h evaluated on source j calibration samples
    for j in range(K):
        X_cal_j = np.asarray(X_cal_list[j])
        Y_cal_j = np.asarray(Y_cal_list[j])
        if X_cal_j.ndim == 1:
            X_cal_j = X_cal_j.reshape(-1, X_cal_j.size)  # attempt to fix shape
        n_j = X_cal_j.shape[0]
        if n_j <= 0:
            raise ValueError(f"Calibration set for source {j} is empty.")

        cache_entry = None
        if calibration_cache is not None:
            cache_entry = calibration_cache.get(j)
            if cache_entry is not None:
                log_h_cal = np.asarray(cache_entry.get("log_h_cal"))
                cache_n_j = int(cache_entry.get("n_j", log_h_cal.shape[0]))
                if log_h_cal.shape[0] != cache_n_j or cache_n_j != n_j:
                    cache_entry = None

        if cache_entry is None:
            # lam evaluated at calibration X_cal_j: shape (n_j, K)
            if lam_model is None:
                raise ValueError(
                    "lam_model must be provided when calibration_cache lacks precomputed values."
                )
            lam_cal_all = lam_model(X_cal_j)
            if lam_cal_all.ndim == 1:
                lam_cal_all = lam_cal_all.reshape(n_j, -1)
            if lam_cal_all.shape[0] == 1 and n_j > 1:
                lam_cal_all = np.tile(lam_cal_all, (n_j, 1))
            if lam_cal_all.shape[0] != n_j or lam_cal_all.shape[1] != K:
                raise ValueError("lam_model returned unexpected shape for calibration X.")

            joint_calib = np.zeros((n_j, K), dtype=float)
            for k, src_k in enumerate(sources):
                if hasattr(src_k, 'joint_pdf_at_pairs'):
                    joint_calib[:, k] = src_k.joint_pdf_at_pairs(X_cal_j, Y_cal_j)
                elif hasattr(src_k, 'joint_prob_at_pairs'):
                    joint_calib[:, k] = src_k.joint_prob_at_pairs(X_cal_j, Y_cal_j)
                else:
                    raise AttributeError(
                        f"Source {k} must have either joint_pdf_at_pairs or joint_prob_at_pairs method"
                    )

            # now compute h_cal (n_j,) = sum_k lam_cal_all[i,k] * joint_calib[i,k]
            h_cal = np.sum(lam_cal_all * joint_calib, axis=1)   # shape (n_j,)
            log_h_cal = np.log(np.maximum(h_cal, 1e-100))  # avoid -inf

            log_h_cal_sorted = np.sort(log_h_cal)
            if calibration_cache is not None:
                calibration_cache[j] = {
                    "log_h_cal": log_h_cal,
                    "log_h_cal_sorted": log_h_cal_sorted,
                    "n_j": n_j,
                }
        else:
            log_h_cal = np.asarray(cache_entry["log_h_cal"])
            log_h_cal_sorted = np.asarray(cache_entry.get("log_h_cal_sorted"))
            if log_h_cal_sorted is None or log_h_cal_sorted.size != log_h_cal.size:
                log_h_cal_sorted = np.sort(log_h_cal)
                if calibration_cache is not None:
                    cache_entry["log_h_cal_sorted"] = log_h_cal_sorted
            n_j = int(cache_entry.get("n_j", log_h_cal.shape[0]))

        inv_denom = 1.0 / (n_j + 1)

        # Vectorized counts for less-than and tie handling via tolerance bands
        if randomize_ties:
            u = rng_local.random(size=m)
        else:
            u = np.ones(m)

        less_counts = np.searchsorted(log_h_cal_sorted, log_h_test, side='left')
        tol = atol + rtol * np.abs(log_h_test)
        lower_bounds = log_h_test - tol
        upper_bounds = log_h_test + tol
        lower_idx = np.searchsorted(log_h_cal_sorted, lower_bounds, side='left')
        upper_idx = np.searchsorted(log_h_cal_sorted, upper_bounds, side='right')
        eq_counts = np.maximum(0, upper_idx - lower_idx)
        p_values_y_grid[j, :] = (1.0 + less_counts + u * eq_counts) * inv_denom

        # Compute p-value for true label (single test point)
        if randomize_ties:
            u_true = rng_local.random()
        else:
            u_true = 1.0

        tol_true = atol + rtol * abs(log_h_true_scalar)
        less_true = np.searchsorted(log_h_cal_sorted, log_h_true_scalar, side='left')
        lower_true = log_h_true_scalar - tol_true
        upper_true = log_h_true_scalar + tol_true
        lower_true_idx = np.searchsorted(log_h_cal_sorted, lower_true, side='left')
        upper_true_idx = np.searchsorted(log_h_cal_sorted, upper_true, side='right')
        eq_true = max(0, upper_true_idx - lower_true_idx)
        p_values_true_y[j] = (1.0 + less_true + u_true * eq_true) * inv_denom


    # Aggregated set via max-p (union), represented by boolean masks
    union_mask_y_grid = np.any(p_values_y_grid > alpha, axis=0)   # shape (m,)
    union_mask_true_y = np.any(p_values_true_y > alpha)           # scalar boolean for single test point

    return {
        "p_values_y_grid": p_values_y_grid,               # shape (K, m)
        "union_mask_y_grid": union_mask_y_grid,           # shape (m,) boolean, True if included
        "h_y_grid": h_y_grid,                             # shape (m,)
        "y_grid": y_grid,                                 # shape (m,), full grid of candidate y values
        "h_true": h_true,                                 # scalar for single test point
        "p_values_true_y": p_values_true_y,               # shape (K,) for single test point
        "union_mask_true_y": union_mask_true_y,           # scalar boolean for single test point
    }
