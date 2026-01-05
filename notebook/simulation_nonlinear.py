import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple


def softplus_np(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    large = z > 10
    small = z < -10
    out = np.zeros_like(z)
    out[large] = z[large]
    out[small] = 0.0
    mid = ~(large | small)
    out[mid] = np.log1p(np.exp(z[mid]))
    return out


class MultiSourceSimulator:
    """High-dimensional simulator supporting sparse nonlinear effects."""

    MAX_INTERACTION_PAIRS = 6
    MAX_SOFTPLUS_UNITS = 3
    MAX_SINUSOID_UNITS = 3
    MAX_QUADRATIC_TERMS = 4
    PROJECTION_SUPPORT = 3

    def __init__(self, random_seed: int = 42, temperature: float = 1.0) -> None:
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.temperature = float(temperature)

    def _resolve_sample_counts(self, n_sources: int, n_samples_per_source: Any) -> List[int]:
        if isinstance(n_samples_per_source, int):
            return [n_samples_per_source] * n_sources
        if len(n_samples_per_source) != n_sources:
            raise ValueError("n_samples_per_source must have length equal to n_sources")
        return list(n_samples_per_source)

    def _build_covariance(self, n_features: int, correlation: float) -> np.ndarray:
        correlation = float(correlation)
        if correlation <= 0:
            return np.eye(n_features)
        rho = min(max(correlation, -0.95), 0.95)
        cov = np.full((n_features, n_features), rho)
        np.fill_diagonal(cov, 1.0)
        return cov

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        mu = np.mean(X, axis=0, keepdims=True)
        sigma = np.std(X, axis=0, keepdims=True)
        return (X - mu) / (sigma + 1e-8)

    def _select_informative(self, n_features: int, n_informative: Optional[int]) -> np.ndarray:
        if n_informative is None:
            n_informative = min(n_features, 4)
            if n_informative >= 3:
                n_informative = max(3, n_informative)
        if n_informative > n_features:
            n_informative = n_features
        if n_informative < 1:
            n_informative = min(1, n_features)
        idx = self.rng.choice(n_features, size=n_informative, replace=False)
        return np.sort(idx)

    def _choose_pairs(self, n_features: int, informative_idx: np.ndarray, max_pairs: int) -> np.ndarray:
        if max_pairs <= 0 or informative_idx.size < 2:
            return np.zeros((0, 2), dtype=int)
        candidate_pairs = list(combinations(informative_idx.tolist(), 2))
        if len(candidate_pairs) < max_pairs:
            supplemental = [pair for pair in combinations(range(n_features), 2) if pair not in candidate_pairs]
            candidate_pairs.extend(supplemental)
        if not candidate_pairs:
            return np.zeros((0, 2), dtype=int)
        count = min(max_pairs, len(candidate_pairs))
        indices = self.rng.choice(len(candidate_pairs), size=count, replace=False)
        return np.asarray([candidate_pairs[i] for i in indices], dtype=int)

    def _build_sparse_projections(
        self,
        n_features: int,
        informative_idx: np.ndarray,
        n_terms: int,
        support: int,
        scale_bounds: Tuple[float, float],
        bias_bounds: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_terms <= 0 or informative_idx.size == 0:
            return np.zeros((0, n_features)), np.zeros(0), np.zeros(0)
        projections: List[np.ndarray] = []
        scales: List[float] = []
        biases: List[float] = []
        support = max(1, min(support, informative_idx.size))
        for _ in range(n_terms):
            chosen = self.rng.choice(informative_idx, size=support, replace=False)
            weights = self.rng.normal(0.0, 1.0, size=support)
            norm = max(np.linalg.norm(weights), 1e-8)
            weights = weights / norm
            weights *= self.rng.uniform(0.3, 0.7) * (1.0 + 0.1 * self.temperature)
            proj = np.zeros(n_features)
            proj[chosen] = weights
            projections.append(proj)
            scales.append(self.rng.uniform(*scale_bounds) * (1.0 + 0.1 * self.temperature))
            biases.append(self.rng.uniform(*bias_bounds))
        return np.vstack(projections), np.asarray(biases), np.asarray(scales)

    def _build_quadratics(self, informative_idx: np.ndarray, max_terms: int) -> Tuple[np.ndarray, np.ndarray]:
        if max_terms <= 0 or informative_idx.size == 0:
            return np.zeros(0, dtype=int), np.zeros(0)
        count = min(max_terms, informative_idx.size)
        chosen = np.sort(self.rng.choice(informative_idx, size=count, replace=False))
        weights = self.rng.normal(0.0, 0.5 + 0.3 * self.temperature, size=count)
        return chosen, weights

    def _resolve_multipliers(self, provided: Optional[Dict[str, float]], keys: List[str], defaults: Dict[str, float]) -> Dict[str, float]:
        resolved = {k: float(defaults.get(k, 0.0)) for k in keys}
        if provided:
            for key, value in provided.items():
                if key in resolved:
                    resolved[key] = float(value)
        return resolved

    def _merge_params(self, defaults: Dict[str, Any], override: Optional[Dict[str, Any]], dtype_map: Dict[str, Any]) -> Dict[str, Any]:
        if not override:
            return {k: np.array(v, copy=True) for k, v in defaults.items()}
        merged: Dict[str, Any] = {}
        for key, default_value in defaults.items():
            if key in override:
                arr = np.asarray(override[key])
                if key in dtype_map:
                    arr = arr.astype(dtype_map[key], copy=False)
                merged[key] = np.array(arr, copy=True)
            else:
                merged[key] = np.array(default_value, copy=True)
        return merged

    def _default_classification_params(self, n_features: int, informative_idx: np.ndarray) -> Dict[str, Any]:
        interaction_pairs = self._choose_pairs(n_features, informative_idx, self.MAX_INTERACTION_PAIRS)
        if interaction_pairs.size:
            interaction_weights = self.rng.normal(0.0, 0.6 + 0.2 * self.temperature, size=interaction_pairs.shape[0])
        else:
            interaction_weights = np.zeros(0)
        soft_proj, soft_bias, soft_scale = self._build_sparse_projections(
            n_features,
            informative_idx,
            self.MAX_SOFTPLUS_UNITS,
            self.PROJECTION_SUPPORT,
            (0.6, 1.6),
            (-0.5, 0.5),
        )
        sin_proj, sin_bias, sin_scale = self._build_sparse_projections(
            n_features,
            informative_idx,
            self.MAX_SINUSOID_UNITS,
            self.PROJECTION_SUPPORT,
            (0.4, 1.2),
            (-np.pi / 3.0, np.pi / 3.0),
        )
        return {
            'interaction_pairs': interaction_pairs,
            'interaction_weights': interaction_weights,
            'softplus_proj': soft_proj,
            'softplus_bias': soft_bias,
            'softplus_scale': soft_scale,
            'sinusoid_proj': sin_proj,
            'sinusoid_bias': sin_bias,
            'sinusoid_scale': sin_scale,
        }

    def _default_regression_params(self, n_features: int, informative_idx: np.ndarray) -> Dict[str, Any]:
        base = self._default_classification_params(n_features, informative_idx)
        quad_idx, quad_weights = self._build_quadratics(informative_idx, self.MAX_QUADRATIC_TERMS)
        base.update({'quadratic_indices': quad_idx, 'quadratic_weights': quad_weights})
        return base

    def _interaction_effect(self, X: np.ndarray, pairs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if pairs.size == 0 or weights.size == 0:
            return np.zeros(X.shape[0])
        interactions = X[:, pairs[:, 0]] * X[:, pairs[:, 1]]
        return interactions @ weights

    def _softplus_effect(self, X: np.ndarray, proj: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if proj.size == 0 or scale.size == 0:
            return np.zeros(X.shape[0])
        z = X @ proj.T + bias
        vals = softplus_np(z)
        return vals @ scale

    def _sinusoid_effect(self, X: np.ndarray, proj: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> np.ndarray:
        if proj.size == 0 or scale.size == 0:
            return np.zeros(X.shape[0])
        z = X @ proj.T + bias
        vals = np.sin(z)
        return vals @ scale

    def _quadratic_effect(self, X: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if indices.size == 0 or weights.size == 0:
            return np.zeros(X.shape[0])
        values = X[:, indices] ** 2
        return values @ weights

    def generate_multisource_classification(
        self,
        n_sources: int = 3,
        n_samples_per_source: Any = (2000, 2000, 2000),
        n_features: int = 10,
        n_classes: int = 2,
        min_total_samples: int = 1000,
        use_nonlinear: bool = False,
        standardize_features: bool = True,
        nonlinear_config: Optional[Dict[str, Any]] = None,
        n_informative: Optional[int] = None,
        correlation: float = 0.2,
        signal_scale: float = 2.5,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)
        if sum(n_samples_per_source) < min_total_samples:
            raise ValueError("Total samples across sources below the required minimum.")

        informative_idx = self._select_informative(n_features, n_informative)
        cov_base = self._build_covariance(n_features, correlation)

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        # Multinomial-softmax parameterization
        base_betas = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            base_betas[c, informative_idx] = self.rng.normal(0, 1.0, size=informative_idx.size)

        if use_nonlinear:
            defaults = self._default_classification_params(n_features, informative_idx)
            params = self._merge_params(
                defaults,
                nonlinear_config.get('params') if nonlinear_config else None,
                {
                    'interaction_pairs': int,
                    'interaction_weights': float,
                    'softplus_proj': float,
                    'softplus_bias': float,
                    'softplus_scale': float,
                    'sinusoid_proj': float,
                    'sinusoid_bias': float,
                    'sinusoid_scale': float,
                },
            )
            multipliers = self._resolve_multipliers(
                nonlinear_config.get('multipliers') if nonlinear_config else None,
                ['interaction', 'softplus', 'sinusoid'],
                {'interaction': 2.0, 'softplus': 0.0, 'sinusoid': 0.0},
            )
        else:
            params = None
            multipliers = {'interaction': 0.0, 'softplus': 0.0, 'sinusoid': 0.0}

        X_sources: List[np.ndarray] = []
        Y_sources: List[np.ndarray] = []
        source_params: List[Dict[str, Any]] = []

        for j in range(n_sources):
            X_j = self.rng.multivariate_normal(np.zeros(n_features), cov_base, n_samples_per_source[j])
            X_std = self._standardize(X_j) if standardize_features else X_j

            signal_multiplier = signal_scale * (1.0 + 0.25 * self.temperature * self.rng.uniform(-1, 1))

            betas_j = base_betas.copy()
            betas_j[:, informative_idx] += self.temperature * self.rng.normal(
                0, 0.15, size=(n_classes, informative_idx.size)
            )
            # b_{kc} ~ N(0, (0.4*tau)^2) independently across (k,c).
            biases_j = self.rng.normal(0, 0.4 * self.temperature, size=n_classes)

            logits = (X_std @ betas_j.T) + biases_j
            logits *= signal_multiplier

            nonlinear_effects = np.zeros(X_std.shape[0])
            if use_nonlinear and params is not None:
                nonlinear_effects += multipliers['interaction'] * self._interaction_effect(
                    X_std,
                    params['interaction_pairs'],
                    params['interaction_weights'],
                )
                nonlinear_effects += multipliers['softplus'] * self._softplus_effect(
                    X_std,
                    params['softplus_proj'],
                    params['softplus_bias'],
                    params['softplus_scale'],
                )
                nonlinear_effects += multipliers['sinusoid'] * self._sinusoid_effect(
                    X_std,
                    params['sinusoid_proj'],
                    params['sinusoid_bias'],
                    params['sinusoid_scale'],
                )

            # Apply nonlinear effects as a common shift to non-reference classes.
            logits[:, 1:] += nonlinear_effects[:, None]
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            Y_j = np.array([self.rng.choice(n_classes, p=row) for row in probs])
            stored_params = {
                'betas': betas_j,
                'biases': biases_j,
            }

            X_sources.append(X_std if standardize_features else X_j)
            Y_sources.append(Y_j)

            source_entry: Dict[str, Any] = {
                'source_id': j,
                'n_classes': n_classes,
                'informative_idx': informative_idx.copy(),
                'signal_scale': signal_multiplier,
                'cov_scale': 1.0,
                'use_nonlinear': use_nonlinear,
                'nonlinear_multipliers': multipliers.copy(),
                **stored_params,
            }
            if use_nonlinear and params is not None:
                source_entry['nonlinear_params'] = {k: np.array(v, copy=True) for k, v in params.items()}
            source_params.append(source_entry)

        return X_sources, Y_sources, source_params

    def generate_multisource_regression(
        self,
        n_sources: int = 3,
        n_samples_per_source: Any = (2000, 2000, 2000),
        n_features: int = 10,
        use_nonlinear: bool = False,
        standardize_features: bool = True,
        heteroskedastic: bool = False,
        nonlinear_config: Optional[Dict[str, Any]] = None,
        n_informative: Optional[int] = None,
        correlation: float = 0.2,
        snr_range: Tuple[float, float] = (5.0, 10.0),
        apply_temperature_noise: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)
        informative_idx = self._select_informative(n_features, n_informative)
        cov_base = self._build_covariance(n_features, correlation)

        base_beta = np.zeros(n_features)
        base_beta[informative_idx] = self.rng.normal(0, 1.0, size=informative_idx.size)
        base_bias = self.rng.normal(0, 0.5)

        if use_nonlinear:
            defaults = self._default_regression_params(n_features, informative_idx)
            params = self._merge_params(
                defaults,
                nonlinear_config.get('params') if nonlinear_config else None,
                {
                    'interaction_pairs': int,
                    'interaction_weights': float,
                    'softplus_proj': float,
                    'softplus_bias': float,
                    'softplus_scale': float,
                    'sinusoid_proj': float,
                    'sinusoid_bias': float,
                    'sinusoid_scale': float,
                    'quadratic_indices': int,
                    'quadratic_weights': float,
                },
            )
            multipliers = self._resolve_multipliers(
                nonlinear_config.get('multipliers') if nonlinear_config else None,
                ['interaction', 'softplus', 'sinusoid', 'quadratic'],
                {'interaction': 2.0, 'softplus': 0.0, 'sinusoid': 0.0, 'quadratic': 0.0},
            )
        else:
            params = None
            multipliers = {'interaction': 0.0, 'softplus': 0.0, 'sinusoid': 0.0, 'quadratic': 0.0}

        X_sources: List[np.ndarray] = []
        Y_sources: List[np.ndarray] = []
        source_params: List[Dict[str, Any]] = []

        for j in range(n_sources):
            X_j = self.rng.multivariate_normal(np.zeros(n_features), cov_base, n_samples_per_source[j])
            X_std = self._standardize(X_j) if standardize_features else X_j

            beta_j = base_beta.copy()
            beta_j[informative_idx] += self.temperature * self.rng.normal(0, 0.2, size=informative_idx.size)
            bias_j = base_bias + self.rng.normal(0, 0.5 * self.temperature)

            signal = X_std @ beta_j + bias_j

            nonlinear_effects = np.zeros(X_std.shape[0])
            if use_nonlinear and params is not None:
                nonlinear_effects += multipliers['interaction'] * self._interaction_effect(
                    X_std,
                    params['interaction_pairs'],
                    params['interaction_weights'],
                )
                nonlinear_effects += multipliers['softplus'] * self._softplus_effect(
                    X_std,
                    params['softplus_proj'],
                    params['softplus_bias'],
                    params['softplus_scale'],
                )
                nonlinear_effects += multipliers['sinusoid'] * self._sinusoid_effect(
                    X_std,
                    params['sinusoid_proj'],
                    params['sinusoid_bias'],
                    params['sinusoid_scale'],
                )
                nonlinear_effects += multipliers['quadratic'] * self._quadratic_effect(
                    X_std,
                    params['quadratic_indices'],
                    params['quadratic_weights'],
                )

            signal += nonlinear_effects
            signal_var = np.var(signal)

            snr_target = self.rng.uniform(*snr_range)
            noise_std = np.sqrt(max(signal_var, 1e-6) / max(snr_target, 1e-6))
            noise_multiplier = 1.0
            if apply_temperature_noise:
                low = max(0.0, 1.0 - self.temperature / 4.0)
                high = 1.0 + self.temperature / 4.0
                noise_multiplier = float(self.rng.uniform(low, high))
                noise_std *= noise_multiplier
            noise_std = max(noise_std, 1e-3)

            if heteroskedastic:
                ref_feature = X_std[:, 0] if standardize_features else X_j[:, 0]
                noise = self.rng.normal(0, noise_std * (1 + 0.5 * np.abs(ref_feature)))
            else:
                noise = self.rng.normal(0, noise_std, n_samples_per_source[j])

            Y_j = signal + noise

            X_sources.append(X_std if standardize_features else X_j)
            Y_sources.append(Y_j)
            source_entry: Dict[str, Any] = {
                'source_id': j,
                'beta': beta_j,
                'bias': bias_j,
                'noise_std': noise_std,
                'noise_multiplier': noise_multiplier,
                'snr_target': snr_target,
                'informative_idx': informative_idx.copy(),
                'cov_scale': 1.0,
                'use_nonlinear': use_nonlinear,
                'heteroskedastic': heteroskedastic,
                'nonlinear_multipliers': multipliers.copy(),
            }
            if use_nonlinear and params is not None:
                source_entry['nonlinear_params'] = {k: np.array(v, copy=True) for k, v in params.items()}
            source_params.append(source_entry)

        return X_sources, Y_sources, source_params

    @staticmethod
    def create_classification_config(
        interaction: float = 0.0,
        sinusoid: float = 0.0,
        softplus: float = 0.0,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = {
            'multipliers': {
                'interaction': interaction,
                'sinusoid': sinusoid,
                'softplus': softplus,
            }
        }
        if custom_params is not None:
            config['params'] = custom_params
        return config

    @staticmethod
    def create_regression_config(
        quadratic: float = 0.0,
        interaction: float = 0.0,
        sinusoid: float = 0.0,
        softplus: float = 0.0,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = {
            'multipliers': {
                'quadratic': quadratic,
                'interaction': interaction,
                'sinusoid': sinusoid,
                'softplus': softplus,
            }
        }
        if custom_params is not None:
            config['params'] = custom_params
        return config

    @staticmethod
    def get_available_terms(task: str = 'classification') -> List[str]:
        if task == 'classification':
            return ['interaction', 'sinusoid', 'softplus']
        if task == 'regression':
            return ['quadratic', 'interaction', 'sinusoid', 'softplus']
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'regression'.")

    def visualize_multisource_data(
        self,
        X_sources: List[np.ndarray],
        Y_sources: List[np.ndarray],
        task: str = 'classification',
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        n_sources = len(X_sources)
        colors = plt.cm.Set1(np.linspace(0, 1, n_sources))

        if task == 'classification':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            Y_all = np.hstack(Y_sources)
            unique_classes = np.unique(Y_all)
            n_classes = len(unique_classes)
            class_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x'][:n_classes]
            class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

            ax = axes[0]
            for j, (X_j, Y_j) in enumerate(zip(X_sources, Y_sources)):
                for i, class_val in enumerate(unique_classes):
                    mask = Y_j == class_val
                    if np.any(mask):
                        ax.scatter(
                            X_j[mask, 0],
                            X_j[mask, 1],
                            c=[colors[j]],
                            alpha=0.7,
                            marker=class_markers[i],
                            s=60,
                            label=f'Source {j + 1}, Class {class_val}',
                        )
            ax.set_title('Multi-Source Classification Data')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            ax = axes[1]
            X_all = np.vstack(X_sources)
            for i, class_val in enumerate(unique_classes):
                mask = Y_all == class_val
                ax.scatter(
                    X_all[mask, 0],
                    X_all[mask, 1],
                    c=[class_colors[i]],
                    alpha=0.7,
                    marker=class_markers[i],
                    s=60,
                    label=f'Class {class_val}',
                )
            ax.set_title('Combined Classification Data')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()

        elif task == 'regression':
            fig, axes = plt.subplots(2, 2, figsize=figsize)

            ax = axes[0, 0]
            for j, (X_j, Y_j) in enumerate(zip(X_sources, Y_sources)):
                scatter = ax.scatter(X_j[:, 0], X_j[:, 1], c=Y_j, alpha=0.7, cmap='viridis')
                ax.set_title(f'Source {j + 1} (colored by target)')
                plt.colorbar(scatter, ax=ax)
                break

            ax = axes[0, 1]
            for j, Y_j in enumerate(Y_sources):
                ax.hist(Y_j, bins=20, alpha=0.6, label=f'Source {j + 1}', color=colors[j])
            ax.set_title('Target Distribution by Source')
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Frequency')
            ax.legend()

            ax = axes[1, 0]
            for j, X_j in enumerate(X_sources):
                ax.scatter(X_j[:, 0], X_j[:, 1], c=colors[j], alpha=0.7, label=f'Source {j + 1}')
            ax.set_title('Feature Space by Source')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()

            ax = axes[1, 1]
            means = [np.mean(Y_j) for Y_j in Y_sources]
            stds = [np.std(Y_j) for Y_j in Y_sources]
            x_pos = range(len(means))
            ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors)
            ax.set_title('Target Mean +/- Std by Source')
            ax.set_xlabel('Source')
            ax.set_ylabel('Target Value')
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels([f'Source {j + 1}' for j in range(len(means))])

        else:
            raise ValueError("task must be 'classification' or 'regression'")

        plt.tight_layout()
        return fig

    def get_data_summary(
        self,
        X_sources: List[np.ndarray],
        Y_sources: List[np.ndarray],
        task: str = 'classification',
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'n_sources': len(X_sources),
            'n_features': X_sources[0].shape[1] if X_sources else 0,
            'total_samples': sum(len(X) for X in X_sources),
            'samples_per_source': [len(X) for X in X_sources],
            'task': task,
        }

        if task == 'classification':
            summary['class_distributions'] = [np.bincount(Y) for Y in Y_sources]
            summary['unique_classes'] = list(np.unique(np.hstack(Y_sources)))
        elif task == 'regression':
            summary['target_stats'] = [
                {
                    'mean': float(np.mean(Y)),
                    'std': float(np.std(Y)),
                    'min': float(np.min(Y)),
                    'max': float(np.max(Y)),
                }
                for Y in Y_sources
            ]
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        return summary
