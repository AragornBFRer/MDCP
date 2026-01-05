import numpy as np
import matplotlib.pyplot as plt

class MultiSourceSimulator:
    """
    Core simulator class for generating multi-source data for MDCP experiments.
    Focuses only on data generation and visualization.
    """
    
    def __init__(self, random_seed=42, temperature=1.0):
        """
        Initialize the simulator with a random seed and temperature.
        
        :param random_seed: int
            Random seed for reproducibility
        :param temperature: float
            Controls variation between sources and classes (default: 1.0)
            - Higher values (>1.0) increase variation between sources
            - Lower values (<1.0) decrease variation, making sources more similar
            - 0.0 would make all sources identical (not recommended)
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.temperature = temperature
        
    def _resolve_sample_counts(self, n_sources, n_samples_per_source):
        """Normalize sample count specification across sources."""
        if isinstance(n_samples_per_source, int):
            return [n_samples_per_source] * n_sources
        if len(n_samples_per_source) != n_sources:
            raise ValueError("n_samples_per_source must have length equal to n_sources")
        return list(n_samples_per_source)

    def _build_covariance(self, n_features, correlation):
        """Create an equicorrelated covariance matrix."""
        correlation = float(correlation)
        if correlation <= 0:
            return np.eye(n_features)
        rho = min(max(correlation, -0.95), 0.95)
        cov = np.full((n_features, n_features), rho)
        np.fill_diagonal(cov, 1.0)
        return cov

    def _standardize(self, X):
        """Standardize each feature to zero mean and unit variance."""
        mu = np.mean(X, axis=0, keepdims=True)
        sigma = np.std(X, axis=0, keepdims=True)
        return (X - mu) / (sigma + 1e-8)

    def _select_informative(self, n_features, n_informative=None):
        """Select informative feature indices respecting sparsity guidance."""
        if n_informative is None:
            n_informative = min(n_features, 4)
            if n_informative >= 3:
                n_informative = max(3, n_informative)
        if n_informative > n_features:
            n_informative = n_features
        if n_informative < 1:
            n_informative = min(1, n_features)
        return np.sort(np.random.choice(n_features, size=n_informative, replace=False))
        
    def generate_multisource_classification(self, n_sources=3, n_samples_per_source=[2000, 2000, 2000], 
                                          n_features=10, n_classes=2, min_total_samples=1000,
                                          n_informative=None, correlation=0.2, signal_scale=2.5,
                                          standardize_features=True):
        """
        Generate multi-source classification data with calibrated high-dimensional signals.

        All sources share the same informative coordinates but exhibit source-specific
        parameter perturbations controlled by ``temperature``.
        """
        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)
        if sum(n_samples_per_source) < min_total_samples:
            raise ValueError("Total samples across sources below the required minimum.")

        informative_idx = self._select_informative(n_features, n_informative)
        cov_base = self._build_covariance(n_features, correlation)

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        # Multinomial-softmax parameterization.
        base_betas = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            base_betas[c, informative_idx] = np.random.normal(0, 1.0, size=informative_idx.size)

        X_sources = []
        Y_sources = []
        source_params = []

        for j in range(n_sources):
            X_j = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=cov_base,
                size=n_samples_per_source[j]
            )

            if standardize_features:
                X_std = self._standardize(X_j)
            else:
                X_std = X_j

            signal_multiplier = signal_scale * (1.0 + 0.25 * self.temperature * np.random.uniform(-1, 1))

            betas_j = base_betas.copy()
            # b_{kc} ~ N(0, (0.4*tau)^2) independently across (k,c).
            biases_j = np.random.normal(0, 0.4 * self.temperature, size=n_classes)
            betas_j[:, informative_idx] += self.temperature * np.random.normal(
                0, 0.15, size=(n_classes, informative_idx.size)
            )

            # Scale both slopes and intercepts by the source signal strength.
            logits = (X_std @ betas_j.T) + biases_j
            logits *= signal_multiplier
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            Y_j = np.array([np.random.choice(n_classes, p=row) for row in probs])

            source_params.append({
                'source_id': j,
                'n_classes': n_classes,
                'betas': betas_j,
                'biases': biases_j,
                'informative_idx': informative_idx.copy(),
                'signal_scale': signal_multiplier,
                'cov_scale': 1.0,
            })

            X_sources.append(X_std)
            Y_sources.append(Y_j)

        return X_sources, Y_sources, source_params
    
    def generate_multisource_regression(self, n_sources=3, n_samples_per_source=[1000, 1000, 1000], 
                                       n_features=10, n_informative=None, correlation=0.2,
                                       snr_range=(5.0, 10.0), standardize_features=True,
                                       apply_temperature_noise=False):
        """
        Generate multi-source regression data with controlled signal-to-noise ratios.

        Each source shares informative coordinates but receives temperature-controlled
        perturbations in coefficients and noise scales.
        """
        n_samples_per_source = self._resolve_sample_counts(n_sources, n_samples_per_source)
        informative_idx = self._select_informative(n_features, n_informative)
        cov_base = self._build_covariance(n_features, correlation)

        base_beta = np.zeros(n_features)
        base_beta[informative_idx] = np.random.normal(0, 1.0, size=informative_idx.size)
        base_bias = np.random.normal(0, 0.5)

        X_sources = []
        Y_sources = []
        source_params = []

        for j in range(n_sources):
            X_j = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=cov_base,
                size=n_samples_per_source[j]
            )

            if standardize_features:
                X_std = self._standardize(X_j)
            else:
                X_std = X_j

            beta_j = base_beta.copy()
            beta_j[informative_idx] += self.temperature * np.random.normal(0, 0.2, size=informative_idx.size)
            bias_j = base_bias + np.random.normal(0, 0.5 * self.temperature)

            signal = X_std @ beta_j + bias_j
            signal_var = np.var(signal)

            snr_target = np.random.uniform(*snr_range)
            noise_std = np.sqrt(max(signal_var, 1e-6) / max(snr_target, 1e-6))
            noise_multiplier = 1.0
            if apply_temperature_noise:
                low = max(0.0, 1.0 - self.temperature / 4.0)
                high = 1.0 + self.temperature / 4.0
                noise_multiplier = float(np.random.uniform(low, high))
                noise_std *= noise_multiplier
            noise_std = max(noise_std, 1e-3)

            noise = np.random.normal(0, noise_std, size=n_samples_per_source[j])
            Y_j = signal + noise

            snr_empirical = np.var(signal) / max(np.var(noise), 1e-6)

            X_sources.append(X_std)
            Y_sources.append(Y_j)
            source_params.append({
                'source_id': j,
                'beta': beta_j,
                'bias': bias_j,
                'noise_std': noise_std,
                'noise_multiplier': noise_multiplier,
                'snr_target': snr_target,
                'snr_empirical': snr_empirical,
                'informative_idx': informative_idx.copy(),
                'cov_scale': 1.0,
            })
            
        return X_sources, Y_sources, source_params
    
    def visualize_multisource_data(self, X_sources, Y_sources, task='classification', 
                                  figsize=(12, 8)):
        """
        Visualize multi-source data.

        :param X_sources: list
            List of feature arrays
        :param Y_sources: list
            List of target/label arrays
        :param task: str
            Either 'classification' or 'regression'
        :param figsize: tuple
            Figure size for plots
        """
        n_sources = len(X_sources)
        colors = plt.cm.Set1(np.linspace(0, 1, n_sources))
        
        if task == 'classification':
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Determine unique classes and create markers/colors
            Y_all = np.hstack(Y_sources)
            unique_classes = np.unique(Y_all)
            n_classes = len(unique_classes)
            class_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x'][:n_classes]
            class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
            
            # Plot by source and class
            ax = axes[0]
            for j, (X_j, Y_j) in enumerate(zip(X_sources, Y_sources)):
                for i, class_val in enumerate(unique_classes):
                    mask = Y_j == class_val
                    if np.any(mask):
                        ax.scatter(X_j[mask, 0], X_j[mask, 1], 
                                  c=[colors[j]], alpha=0.7,
                                  marker=class_markers[i],
                                  s=60,
                                  label=f'Source {j+1}, Class {class_val}')
            ax.set_title('Multi-Source Classification Data')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Combined view - color by class only
            ax = axes[1]
            X_all = np.vstack(X_sources)
            for i, class_val in enumerate(unique_classes):
                mask = Y_all == class_val
                ax.scatter(X_all[mask, 0], X_all[mask, 1],
                          c=[class_colors[i]], alpha=0.7,
                          marker=class_markers[i],
                          s=60,
                          label=f'Class {class_val}')
            ax.set_title('Combined Classification Data')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            
        elif task == 'regression':
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Scatter plot colored by target
            ax = axes[0, 0]
            for j, (X_j, Y_j) in enumerate(zip(X_sources, Y_sources)):
                scatter = ax.scatter(X_j[:, 0], X_j[:, 1], c=Y_j, 
                                   alpha=0.7, cmap='viridis')
                ax.set_title(f'Source {j+1} (colored by target)')
                plt.colorbar(scatter, ax=ax)
                break  # Show only first source for clarity
                
            # Target distributions
            ax = axes[0, 1]
            for j, Y_j in enumerate(Y_sources):
                ax.hist(Y_j, bins=20, alpha=0.6, 
                       label=f'Source {j+1}', color=colors[j])
            ax.set_title('Target Distribution by Source')
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # Feature distributions
            ax = axes[1, 0]
            for j, X_j in enumerate(X_sources):
                ax.scatter(X_j[:, 0], X_j[:, 1], 
                          c=colors[j], alpha=0.7, label=f'Source {j+1}')
            ax.set_title('Feature Space by Source')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            
            # Source-wise target means
            ax = axes[1, 1]
            means = [np.mean(Y_j) for Y_j in Y_sources]
            stds = [np.std(Y_j) for Y_j in Y_sources]
            x_pos = range(len(means))
            ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors)
            ax.set_title('Target Mean ± Std by Source')
            ax.set_xlabel('Source')
            ax.set_ylabel('Target Value')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Source {j+1}' for j in range(len(means))])
        
        plt.tight_layout()
        return fig
    
    def get_data_summary(self, X_sources, Y_sources, task='classification'):
        """
        Get a summary of the generated multi-source data.

        :param X_sources: list
            List of feature arrays
        :param Y_sources: list
            List of target/label arrays
        :param task: str
            Either 'classification' or 'regression'

        :returns summary: dict
            Summary statistics of the data
        """
        summary = {
            'n_sources': len(X_sources),
            'n_features': X_sources[0].shape[1] if X_sources else 0,
            'total_samples': sum(len(X) for X in X_sources),
            'samples_per_source': [len(X) for X in X_sources],
            'task': task
        }
        
        if task == 'classification':
            summary['class_distributions'] = [np.bincount(Y) for Y in Y_sources]
            summary['unique_classes'] = list(np.unique(np.hstack(Y_sources)))
        elif task == 'regression':
            summary['target_stats'] = [
                {'mean': np.mean(Y), 'std': np.std(Y), 'min': np.min(Y), 'max': np.max(Y)}
                for Y in Y_sources
            ]
            
        return summary
