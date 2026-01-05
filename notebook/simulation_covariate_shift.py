"""Covariate-shift simulation suite (linear-only).

This module implements the simulation DGP described in `AGENTS_CovariateShift.md`.

Key properties:
- K=3 sources with shared covariance Sigma (equicorrelated, off-diagonal=0.2).
- Informative coordinates I of size 4 sampled uniformly at random per call.
- Covariate shift via mean shifts supported only on I:
    mu_1 = 0, mu_2 = +delta_x * v, mu_3 = -delta_x * v,
  where v is a random unit vector supported on I.
- Two settings:
    * x_shift_only: P(Y|X) shared across sources.
    * x_and_y_shift: concept shift as in the existing linear DGP, with fixed tau=2.5.
- Linear-only: no nonlinear terms.

Important: Standardization is disabled by default to preserve mean shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _format_setting(setting: str) -> str:
    if setting not in {"x_shift_only", "x_and_y_shift"}:
        raise ValueError("setting must be 'x_shift_only' or 'x_and_y_shift'")
    return setting


@dataclass(frozen=True)
class CovariateShiftConfig:
    """Lightweight container for run-level covariate shift components."""

    informative_idx: np.ndarray
    shift_direction_v: np.ndarray
    means: List[np.ndarray]
    covariance: np.ndarray


class MultiSourceCovariateShiftSimulator:
    """Linear-only multi-source simulator with optional concept shift and covariate shift."""

    def __init__(self, random_seed: int = 42, temperature: float = 2.5) -> None:
        # Keep compatibility with existing scripts which call np.random.seed.
        np.random.seed(int(random_seed))
        self.random_seed = int(random_seed)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(int(random_seed))

    def _resolve_sample_counts(self, n_sources: int, n_samples_per_source: Any) -> List[int]:
        if isinstance(n_samples_per_source, int):
            return [int(n_samples_per_source)] * int(n_sources)
        if len(n_samples_per_source) != n_sources:
            raise ValueError("n_samples_per_source must have length equal to n_sources")
        return [int(v) for v in n_samples_per_source]

    def _build_covariance(self, n_features: int, correlation: float) -> np.ndarray:
        correlation = float(correlation)
        if correlation <= 0:
            return np.eye(int(n_features))
        rho = min(max(correlation, -0.95), 0.95)
        cov = np.full((int(n_features), int(n_features)), rho, dtype=float)
        np.fill_diagonal(cov, 1.0)
        return cov

    def _select_informative(self, n_features: int, n_informative: Optional[int]) -> np.ndarray:
        if n_informative is None:
            n_informative = min(int(n_features), 4)
            if n_informative < 1:
                n_informative = 1
        if n_informative > n_features:
            n_informative = int(n_features)
        idx = self.rng.choice(int(n_features), size=int(n_informative), replace=False)
        return np.sort(idx.astype(int))

    def _sample_shift_direction(self, n_features: int, informative_idx: np.ndarray) -> np.ndarray:
        v = np.zeros(int(n_features), dtype=float)
        if informative_idx.size == 0:
            # Degenerate but keep stable behavior.
            v[0] = 1.0
            return v
        v_I = self.rng.normal(0.0, 1.0, size=int(informative_idx.size))
        v[informative_idx] = v_I
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            # Extremely unlikely; fall back to a canonical axis within I.
            v[informative_idx[0]] = 1.0
            return v
        return v / norm

    def _build_covariate_shift_config(
        self,
        *,
        n_sources: int,
        n_features: int,
        correlation: float,
        n_informative: Optional[int],
        delta_x: float,
    ) -> CovariateShiftConfig:
        informative_idx = self._select_informative(n_features, n_informative)
        cov = self._build_covariance(n_features, correlation)
        v = self._sample_shift_direction(n_features, informative_idx)

        if n_sources != 3:
            raise ValueError("This covariate shift DGP is currently implemented for n_sources=3 only.")

        mu1 = np.zeros(int(n_features), dtype=float)
        mu2 = float(delta_x) * v
        mu3 = -float(delta_x) * v
        means = [mu1, mu2, mu3]

        return CovariateShiftConfig(
            informative_idx=informative_idx,
            shift_direction_v=v,
            means=means,
            covariance=cov,
        )

    def generate_multisource_classification(
        self,
        *,
        setting: str = "x_shift_only",
        delta_x: float = 0.0,
        n_sources: int = 3,
        n_samples_per_source: Any = (2000, 2000, 2000),
        n_features: int = 10,
        n_classes: int = 6,
        n_informative: Optional[int] = None,
        correlation: float = 0.2,
        standardize_features: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate linear-only multi-source classification data."""

        setting = _format_setting(setting)
        n_sources = int(n_sources)
        n_features = int(n_features)
        n_classes = int(n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)

        cov_cfg = self._build_covariate_shift_config(
            n_sources=n_sources,
            n_features=n_features,
            correlation=correlation,
            n_informative=n_informative,
            delta_x=float(delta_x),
        )
        informative_idx = cov_cfg.informative_idx
        cov = cov_cfg.covariance

        # Sample base slopes supported on I.
        base_betas = np.zeros((n_classes, n_features), dtype=float)
        for c in range(n_classes):
            base_betas[c, informative_idx] = self.rng.normal(0.0, 1.0, size=int(informative_idx.size))

        tau = float(self.temperature)  # fixed at 2.5 for this suite

        X_sources: List[np.ndarray] = []
        Y_sources: List[np.ndarray] = []
        params: List[Dict[str, Any]] = []

        if setting == "x_shift_only":
            # Shared intercepts and shared slopes.
            shared_biases = self.rng.normal(0.0, 0.4 * tau, size=n_classes)
            shared_lambda = tau

            for k in range(n_sources):
                X_k = self.rng.multivariate_normal(mean=cov_cfg.means[k], cov=cov, size=n_samples_per_source[k])
                if standardize_features:
                    # Explicitly not recommended for mean-shift experiments; kept for debugging.
                    X_k = (X_k - np.mean(X_k, axis=0, keepdims=True)) / (np.std(X_k, axis=0, keepdims=True) + 1e-8)

                logits = (X_k @ base_betas.T) + shared_biases
                logits = shared_lambda * logits
                probs = _softmax_stable(logits)
                y_k = np.array([self.rng.choice(n_classes, p=row) for row in probs], dtype=int)

                X_sources.append(X_k)
                Y_sources.append(y_k)
                params.append(
                    {
                        'source_id': k,
                        'setting': setting,
                        'delta_x': float(delta_x),
                        'temperature': tau,
                        'lambda_k': float(shared_lambda),
                        'informative_idx': informative_idx.copy(),
                        'shift_direction_v': cov_cfg.shift_direction_v.copy(),
                        'mu_k': cov_cfg.means[k].copy(),
                        'covariance': cov,
                        'betas': base_betas.copy(),
                        'biases': shared_biases.copy(),
                        'standardize_features': bool(standardize_features),
                    }
                )

            return X_sources, Y_sources, params

        # setting == "x_and_y_shift" (concept shift + covariate shift)
        u = self.rng.uniform(-1.0, 1.0, size=n_sources)
        lambdas = tau * (1.0 + 0.25 * tau * u)

        for k in range(n_sources):
            X_k = self.rng.multivariate_normal(mean=cov_cfg.means[k], cov=cov, size=n_samples_per_source[k])
            if standardize_features:
                X_k = (X_k - np.mean(X_k, axis=0, keepdims=True)) / (np.std(X_k, axis=0, keepdims=True) + 1e-8)

            # Independent intercepts per (k,c).
            biases_k = self.rng.normal(0.0, 0.4 * tau, size=n_classes)

            # beta_{kc} = \bar beta_c + tau * Delta_{kc} with Delta supported on I.
            betas_k = base_betas.copy()
            perturb = np.zeros((n_classes, n_features), dtype=float)
            if informative_idx.size:
                perturb[:, informative_idx] = self.rng.normal(
                    0.0,
                    0.15,
                    size=(n_classes, int(informative_idx.size)),
                )
            betas_k = betas_k + tau * perturb

            logits = (X_k @ betas_k.T) + biases_k
            logits = float(lambdas[k]) * logits
            probs = _softmax_stable(logits)
            y_k = np.array([self.rng.choice(n_classes, p=row) for row in probs], dtype=int)

            X_sources.append(X_k)
            Y_sources.append(y_k)
            params.append(
                {
                    'source_id': k,
                    'setting': setting,
                    'delta_x': float(delta_x),
                    'temperature': tau,
                    'u_k': float(u[k]),
                    'lambda_k': float(lambdas[k]),
                    'informative_idx': informative_idx.copy(),
                    'shift_direction_v': cov_cfg.shift_direction_v.copy(),
                    'mu_k': cov_cfg.means[k].copy(),
                    'covariance': cov,
                    'betas': betas_k.copy(),
                    'biases': biases_k.copy(),
                    'standardize_features': bool(standardize_features),
                }
            )

        return X_sources, Y_sources, params

    def generate_multisource_regression(
        self,
        *,
        setting: str = "x_shift_only",
        delta_x: float = 0.0,
        n_sources: int = 3,
        n_samples_per_source: Any = (2000, 2000, 2000),
        n_features: int = 10,
        n_informative: Optional[int] = None,
        correlation: float = 0.2,
        snr_range: Tuple[float, float] = (5.0, 10.0),
        standardize_features: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Generate linear-only multi-source regression data."""

        setting = _format_setting(setting)
        n_sources = int(n_sources)
        n_features = int(n_features)
        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)

        cov_cfg = self._build_covariate_shift_config(
            n_sources=n_sources,
            n_features=n_features,
            correlation=correlation,
            n_informative=n_informative,
            delta_x=float(delta_x),
        )
        informative_idx = cov_cfg.informative_idx
        cov = cov_cfg.covariance

        X_sources: List[np.ndarray] = []
        Y_sources: List[np.ndarray] = []
        params: List[Dict[str, Any]] = []

        tau = float(self.temperature)

        # Draw features first so shared-noise computations can refer to source 1.
        for k in range(n_sources):
            X_k = self.rng.multivariate_normal(mean=cov_cfg.means[k], cov=cov, size=n_samples_per_source[k])
            if standardize_features:
                X_k = (X_k - np.mean(X_k, axis=0, keepdims=True)) / (np.std(X_k, axis=0, keepdims=True) + 1e-8)
            X_sources.append(X_k)

        snr_target = float(self.rng.uniform(float(snr_range[0]), float(snr_range[1])))

        if setting == "x_shift_only":
            base_beta = np.zeros(n_features, dtype=float)
            if informative_idx.size:
                base_beta[informative_idx] = self.rng.normal(0.0, 1.0, size=int(informative_idx.size))
            base_bias = float(self.rng.normal(0.0, 0.5))

            # Sample-variance style, but keep sigma shared across sources.
            signal_ref = X_sources[0] @ base_beta + base_bias
            signal_var_ref = float(np.var(signal_ref))
            noise_std = float(np.sqrt(max(signal_var_ref, 1e-6) / max(snr_target, 1e-6)))
            noise_std = max(noise_std, 1e-3)

            for k in range(n_sources):
                signal = X_sources[k] @ base_beta + base_bias
                noise = self.rng.normal(0.0, noise_std, size=n_samples_per_source[k])
                y_k = signal + noise
                X_k = X_sources[k]

                Y_sources.append(y_k)
                params.append(
                    {
                        'source_id': k,
                        'setting': setting,
                        'delta_x': float(delta_x),
                        'temperature': tau,
                        'snr_target': snr_target,
                        'informative_idx': informative_idx.copy(),
                        'shift_direction_v': cov_cfg.shift_direction_v.copy(),
                        'mu_k': cov_cfg.means[k].copy(),
                        'covariance': cov,
                        'beta': base_beta.copy(),
                        'bias': base_bias,
                        'noise_std': noise_std,
                        'standardize_features': bool(standardize_features),
                        'shared_noise': True,
                    }
                )

            return X_sources, Y_sources, params

        # setting == "x_and_y_shift"
        base_beta = np.zeros(n_features, dtype=float)
        if informative_idx.size:
            base_beta[informative_idx] = self.rng.normal(0.0, 1.0, size=int(informative_idx.size))

        base_bias = float(self.rng.normal(0.0, 0.5))

        for k in range(n_sources):
            delta_k = np.zeros(n_features, dtype=float)
            if informative_idx.size:
                delta_k[informative_idx] = self.rng.normal(0.0, 1.0, size=int(informative_idx.size))
            beta_k = base_beta + 0.2 * tau * delta_k

            v_k = float(self.rng.normal(0.0, 0.5))
            bias_k = base_bias + tau * v_k

            signal = X_sources[k] @ beta_k + bias_k
            signal_var = float(np.var(signal))
            noise_std = float(np.sqrt(max(signal_var, 1e-6) / max(snr_target, 1e-6)))
            noise_std = max(noise_std, 1e-3)

            noise = self.rng.normal(0.0, noise_std, size=n_samples_per_source[k])
            y_k = signal + noise
            Y_sources.append(y_k)

            params.append(
                {
                    'source_id': k,
                    'setting': setting,
                    'delta_x': float(delta_x),
                    'temperature': tau,
                    'snr_target': snr_target,
                    'informative_idx': informative_idx.copy(),
                    'shift_direction_v': cov_cfg.shift_direction_v.copy(),
                    'mu_k': cov_cfg.means[k].copy(),
                    'covariance': cov,
                    'beta': beta_k.copy(),
                    'base_beta': base_beta.copy(),
                    'bias': bias_k,
                    'base_bias': base_bias,
                    'v_k': v_k,
                    'noise_std': noise_std,
                    'standardize_features': bool(standardize_features),
                    'shared_noise': False,
                }
            )

        return X_sources, Y_sources, params

    def get_data_summary(
        self,
        X_sources: Sequence[np.ndarray],
        Y_sources: Sequence[np.ndarray],
        task: str = 'classification',
    ) -> Dict[str, Any]:
        summary = {
            'n_sources': len(X_sources),
            'n_features': int(X_sources[0].shape[1]) if X_sources else 0,
            'total_samples': int(sum(len(X) for X in X_sources)),
            'samples_per_source': [int(len(X)) for X in X_sources],
            'task': task,
        }
        if task == 'classification':
            if Y_sources:
                summary['class_distributions'] = [np.bincount(np.asarray(Y, dtype=int)) for Y in Y_sources]
                summary['unique_classes'] = list(np.unique(np.concatenate(Y_sources).astype(int, copy=False)))
            else:
                summary['class_distributions'] = []
                summary['unique_classes'] = []
        elif task == 'regression':
            summary['target_stats'] = [
                {
                    'mean': float(np.mean(Y)) if len(Y) else float('nan'),
                    'std': float(np.std(Y)) if len(Y) else float('nan'),
                    'min': float(np.min(Y)) if len(Y) else float('nan'),
                    'max': float(np.max(Y)) if len(Y) else float('nan'),
                }
                for Y in Y_sources
            ]
        else:
            raise ValueError("task must be 'classification' or 'regression'")
        return summary
