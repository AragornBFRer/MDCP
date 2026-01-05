"""
Baseline Conformal Prediction Methods

This module implements baseline conformal prediction methods for comparison with the MDCP framework,
focusing on providing prediction sets with coverage guarantees.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import sys
import os

try:  # Prefer optional dependency; only required when CQR is enabled.
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover - defer import error until CQR is requested
    HistGradientBoostingRegressor = None  # type: ignore[assignment]

sys.path.append('../model')
try:
    # using the source distribution modeling as in the MDCP framework
    from model.MDCP import SourceModelClassification, SourceModelRegressionGaussian
except ImportError as e:
    print(f"Warning: Could not import MDCP models: {e}")
    print("Make sure the model/ directory contains the required modules")


@dataclass
class _CQRSourceModel:
    """Container for per-source CQR components."""

    q_lo_model: Optional[Any]
    q_hi_model: Optional[Any]
    tau: Optional[float] = None

    def predict_quantiles(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_lo_model is None or self.q_hi_model is None:
            raise RuntimeError("CQR models not trained for this source.")
        lo = self.q_lo_model.predict(X)
        hi = self.q_hi_model.predict(X)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> Optional[float]:
        if self.q_lo_model is None or self.q_hi_model is None:
            return None
        if X_cal.size == 0 or y_cal.size == 0:
            return None
        q_lo, q_hi = self.predict_quantiles(X_cal)
        y_flat = np.asarray(y_cal, dtype=float).reshape(-1)
        scores = np.maximum(q_lo - y_flat, y_flat - q_hi)
        n = len(scores)
        if n == 0:
            return None
        k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
        k = int(np.clip(k, 1, n))
        tau = float(np.partition(scores, k - 1)[k - 1])
        self.tau = tau
        return tau

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.tau is None:
            raise RuntimeError("CQR source model not calibrated. Call calibrate() first.")
        q_lo, q_hi = self.predict_quantiles(X)
        tau = float(self.tau)
        return q_lo - tau, q_hi + tau


class BaselineConformalPredictor:
    """
    Baseline conformal predictor for multi-source scenarios.
    
    This class implements basic conformal prediction baselines:
    1. Single Source: Use one source's conformal score directly
    2. Max-P Aggregation: Calculate p-value for each source distribution separately,
       and include the data point as long as there is one source's p-value above the desired alpha threshold
    """
    
    def __init__(self, random_seed=42, enable_cqr: bool = False,
                 cqr_quantiles: Optional[Tuple[float, float]] = None):
        """Initialize the predictor with a random seed.

        Args:
            random_seed: Seed controlling stochastic components.
            enable_cqr: When True, fit Conformalized Quantile Regression (CQR)
                per source in addition to the default Gaussian residual baseline.
            cqr_quantiles: Optional (q_lo, q_hi) used when fitting quantile
                regressors. Defaults to (0.1, 0.9) if unspecified.
        """
        self.random_seed = random_seed
        self._rng = np.random.default_rng(self.random_seed)
        self.source_models = None
        self.calibration_scores = {}
        self.calibration_labels = {}
        self.thresholds = {}
        self.task = None
        self._enable_cqr = bool(enable_cqr)
        self._cqr_q_lo = None if cqr_quantiles is None else float(cqr_quantiles[0])
        self._cqr_q_hi = None if cqr_quantiles is None else float(cqr_quantiles[1])
        self._cqr_models: List[_CQRSourceModel] = []
        self._cqr_ready = False
        self._cqr_quantiles: Optional[Tuple[float, float]] = None
        if cqr_quantiles is not None:
            self._cqr_quantiles = (self._cqr_q_lo, self._cqr_q_hi)
    
    def train_source_models(
        self,
        X_sources,
        Y_sources,
        task='classification',
        cqr_alpha: Optional[float] = None,
    ):
        """
        Train individual source models using MDCP framework.
        
        :param X_sources: list
            List of feature arrays
        :param Y_sources: list
            List of target/label arrays
        :param task: str
            Either 'classification' or 'regression'
        :param cqr_alpha: Optional[float]
            Desired miscoverage level for CQR quantile models. When provided
            (and regression CQR is enabled) the lower/upper quantile models are
            fit at (alpha/2, 1-alpha/2). If omitted the fallback quantiles are
            used or overridden by ``cqr_quantiles`` passed at construction.
        
        :returns source_models: list
            List of trained source models
        """
        self.task = task
        self._cqr_models = []  # reset any pre-existing CQR models on re-train
        self._cqr_ready = False
        source_models = []
        
        for j, (X_j, Y_j) in enumerate(zip(X_sources, Y_sources)):
            if task == 'classification':
                model = SourceModelClassification(X_j, Y_j)
            elif task == 'regression':
                model = SourceModelRegressionGaussian(X_j, Y_j)
            else:
                raise ValueError("task must be 'classification' or 'regression'")
                
            source_models.append(model)
            
        self.source_models = source_models

        if task == 'regression' and self._enable_cqr:
            if HistGradientBoostingRegressor is None:
                raise ImportError(
                    "scikit-learn is required to enable the CQR baseline. "
                    "Install scikit-learn or disable enable_cqr."
                )

            if self._cqr_q_lo is not None and self._cqr_q_hi is not None:
                q_lo = float(self._cqr_q_lo)
                q_hi = float(self._cqr_q_hi)
            elif cqr_alpha is not None:
                alpha = float(cqr_alpha)
                if not (0.0 < alpha < 1.0):
                    raise ValueError("cqr_alpha must lie in (0, 1)")
                q_lo = alpha / 2.0
                q_hi = 1.0 - alpha / 2.0
            else:
                q_lo = 0.1
                q_hi = 0.9
            if not (0.0 < q_lo < q_hi < 1.0):
                raise ValueError("cqr_quantiles must satisfy 0 < q_lo < q_hi < 1")
            self._cqr_quantiles = (q_lo, q_hi)

            for X_j, Y_j in zip(X_sources, Y_sources):
                if len(X_j) == 0:
                    self._cqr_models.append(_CQRSourceModel(None, None, None))
                    continue
                q_lo_model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q_lo,
                    random_state=self.random_seed,
                )
                q_hi_model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q_hi,
                    random_state=self.random_seed,
                )
                q_lo_model.fit(X_j, Y_j)
                q_hi_model.fit(X_j, Y_j)
                self._cqr_models.append(_CQRSourceModel(q_lo_model, q_hi_model, None))

        return source_models
    
    def _aps_score(self, y_probs, y_true, classes):
        """
        Compute Adaptive Prediction Set (APS) score for classification.

        Reference paper: https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf

        The implementation is a realization of APS score, 
        with simplified expression but identical value.
        
        :param y_probs: array of shape (n_samples, n_classes)
            Predicted class probabilities
        :param y_true: array of shape (n_samples,)
            True class labels
        :param classes: list
            List of class labels corresponding to y_probs columns
        """
        n_samples, n_classes = y_probs.shape
        scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Sort probabilities in descending order
            sorted_probs = np.sort(y_probs[i])[::-1]
            sorted_indices = np.argsort(y_probs[i])[::-1]
            
            # Find position of true class in sorted order
            true_class_pos_matches = np.where(sorted_indices == y_true[i])[0]
            if len(true_class_pos_matches) == 0:
                scores[i] = 1.0
                continue
            
            true_class_pos = true_class_pos_matches[0]
            
            # APS score: sum of probabilities up to (but not including) true class
            # plus a random uniform component for the true class probability
            if true_class_pos == 0:
                scores[i] = self._rng.uniform(0, sorted_probs[0])
            else:
                scores[i] = np.sum(sorted_probs[:true_class_pos]) + self._rng.uniform(0, sorted_probs[true_class_pos])
        
        return scores
    
    def _regression_score(self, y_pred, y_std, y_true):
        """Compute normalized residual score for regression."""
        return np.abs(y_true - y_pred) / np.maximum(y_std, 1e-6)
    
    def _compute_p_value(self, test_score, calibration_scores):
        """
        :returns p_value: float
            P-value (proportion of calibration scores >= test score)
        """
        n = len(calibration_scores)
        return (1 + np.sum(calibration_scores >= test_score)) / (n + 1)
    
    def _compute_scores(self, X, Y, source_id):
        """
        Compute conformal scores for a given source and data.
        
        :param X: array
            Feature data
        :param Y: array
            Target data
        :param source_id: int
            ID of the source model to use
            
        :returns scores: array
            Conformal scores
        """
        if self.source_models is None:
            raise ValueError("Source models not trained. Call train_source_models first.")
        
        model = self.source_models[source_id]
        
        if self.task == 'classification':
            if len(np.unique(Y)) <= 1:
                return None  # Skip sources with only one class
                
            # Get all possible classes from the model
            if hasattr(model, 'classes'):
                classes = sorted(model.classes)
            else:
                # Fallback: infer from data
                classes = sorted(np.unique(Y))
            
            # Get predicted probabilities for all classes
            joint_probs = model.joint_prob(X, classes)  # (n, n_classes)
            # Normalize to get conditional probabilities p(y|x)
            marginal = joint_probs.sum(axis=1, keepdims=True)
            cond_probs = joint_probs / np.maximum(marginal, 1e-12)
            
            # Compute APS scores
            scores = self._aps_score(cond_probs, Y, classes)
            
        elif self.task == 'regression':
            # Get predictions
            mu = model.predict_mu(X)
            sigma = model.predict_sigma(X)
            
            # Compute normalized residual scores
            scores = self._regression_score(mu, sigma, Y)
            
        return scores
    
    def calibrate(self, X_sources, Y_sources, alpha=0.1):
        """
        Calibrate the conformal predictors using source data.
        """
        if self.source_models is None:
            raise ValueError("Source models not trained. Call train_source_models first.")
        
        n_sources = len(X_sources)

        # Reset calibration artifacts before recomputing
        self.calibration_scores = {}
        self.calibration_labels = {}
        self._cqr_ready = False

        # Compute conformal scores for each source
        for j in range(n_sources):
            scores = self._compute_scores(X_sources[j], Y_sources[j], j)
            if scores is not None:
                self.calibration_scores[j] = scores
                self.calibration_labels[j] = Y_sources[j]
        
        valid_sources = list(self.calibration_scores.keys())
        if not valid_sources:
            raise ValueError("No valid sources for calibration")
        
        # Compute conformal thresholds per source
        self.thresholds = {}
        for j in valid_sources:
            scores_j = np.asarray(self.calibration_scores[j])
            n_j = len(scores_j)
            if n_j == 0:
                continue

            # Conformal index k_j = ceil((n_j + 1) * (1 - alpha))
            k_j = int(np.ceil((n_j + 1) * (1 - alpha)))
            # Translate to quantile level in [0, 1]
            q_level_j = (max(k_j - 1, 0)) / n_j
            q_level_j = min(max(q_level_j, 0.0), 1.0)

            try:
                threshold_j = np.quantile(scores_j, q_level_j, method='higher')
            except TypeError:
                # Fallback for older NumPy versions
                threshold_j = np.quantile(scores_j, q_level_j, interpolation='higher')

            self.thresholds[f'source_{j}'] = threshold_j

        # Fallback threshold for legacy single-source lookups
        first_source = valid_sources[0]
        first_key = f'source_{first_source}'
        if first_key not in self.thresholds:
            raise ValueError(f"No calibration threshold computed for source {first_source}.")
        self.thresholds['single_source'] = self.thresholds[first_key]

        self.alpha = alpha

        if self.task == 'regression' and self._enable_cqr and self._cqr_models:
            any_calibrated = False
            for model, X_cal, Y_cal in zip(self._cqr_models, X_sources, Y_sources):
                if model.q_lo_model is None or model.q_hi_model is None:
                    continue
                tau = model.calibrate(np.asarray(X_cal), np.asarray(Y_cal), alpha)
                if tau is not None:
                    any_calibrated = True
            self._cqr_ready = any_calibrated
    
    def predict_set_single_source(self, X_test, source_id=0):
        """
        Generate prediction sets using single source method.
        
        :param X_test: array
            Test features
        :param source_id: int
            Source ID to use (default: 0 for first valid source)
            Used for single source prediction
            
        :returns prediction_sets: array
            Boolean array indicating which points are in prediction set
        """
        if not self.calibration_scores:
            raise ValueError("Predictor not calibrated. Call calibrate first.")
        
        valid_sources = [j for j in self.calibration_scores.keys()]
        if source_id not in valid_sources:
            source_id = valid_sources[0]  # Use first valid source as fallback
        
        # Compute test scores using specified source
        if self.task == 'classification':
            model = self.source_models[source_id]
            
            # Get all possible classes from the model
            if hasattr(model, 'classes'):
                classes = sorted(model.classes)
            else:
                # fallback: use calibration labels for that source (must have been stored in calibrate)
                classes = sorted(np.unique(self.calibration_labels[source_id]))

            joint_probs = model.joint_prob(X_test, classes)
            marginal = joint_probs.sum(axis=1, keepdims=True)
            cond_probs = joint_probs / np.maximum(marginal, 1e-12)
            
            # Filter out classes with zero probability for all test points
            max_probs_per_class = np.max(cond_probs, axis=0)
            valid_class_mask = max_probs_per_class > 1e-10
            valid_classes = [classes[i] for i in range(len(classes)) if valid_class_mask[i]]
            valid_cond_probs = cond_probs[:, valid_class_mask]
            
            n_test = len(X_test)
            n_classes = len(valid_classes)
            prediction_set = np.zeros((n_test, max(valid_classes) + 1))  # Size based on max class label
            
            for i in range(n_test):
                for c_idx, c in enumerate(valid_classes):
                    test_score = self._aps_score(valid_cond_probs[i:i+1], np.array([c_idx]), valid_classes)[0]
                    p_value = self._compute_p_value(test_score, self.calibration_scores[source_id])
                    if p_value > self.alpha:
                        prediction_set[i, c] = 1

            return prediction_set
            
        elif self.task == 'regression':
            model = self.source_models[source_id]
            mu = model.predict_mu(X_test)
            sigma = model.predict_sigma(X_test)
            
            # For regression, return prediction intervals
            threshold_key = f'source_{source_id}'
            if threshold_key not in self.thresholds:
                raise ValueError(f"Threshold for source {source_id} not available. Did you calibrate?")
            threshold = self.thresholds[threshold_key]
            half_width = threshold * sigma
            
            prediction_sets = {
                'lower': mu - half_width,
                'upper': mu + half_width,
                'center': mu
            }
            return prediction_sets
    
    def predict_set_max_aggregated(self, X_test):
        """
        Generate prediction sets using max-p aggregation method.
        
        Max-p aggregation: Calculate p-value for each source distribution separately,
        and include the data point as long as there is one source's p-value above the desired alpha threshold.
        
        :param X_test: array
            Test features
            
        :returns prediction_sets: array or dict (for classification)
            Prediction sets
        :returns direct_union: dict (for regression)
            Prediction intervals aggregated by direct union
        :returns individual_sets_all: list (for both)
            True individual sets from each source 
            Preliminary structure for max-p aggregation (see `eval_utils.py` for later stage analysis)
        """
        valid_sources = list(self.calibration_scores.keys())
        if len(valid_sources) < 2:
            raise ValueError("Max aggregation requires at least 2 sources.")
        
        if self.task == 'classification':
            n_test = len(X_test)
            
            # Determine all possible classes from all sources
            all_classes = set()
            for j in valid_sources:
                model = self.source_models[j]
                if hasattr(model, 'classes'):
                    all_classes.update(model.classes)
                else:
                    # Include calibration classes for this source
                    all_classes.update(np.unique(self.calibration_labels[j]))
            all_classes = sorted(list(all_classes))
            n_classes_total = max(all_classes) + 1 if all_classes else 2
            
            prediction_sets = np.zeros((n_test, n_classes_total))
            
            individual_info = [[] for _ in range(len(self.source_models))]

            # For each test point, check if ANY source gives a high enough p-value
            for i in range(n_test):
                class_included = np.zeros(n_classes_total, dtype=bool)
                
                for j in valid_sources:
                    model = self.source_models[j]
                    
                    # Get classes for this model
                    if hasattr(model, 'classes'):
                        model_classes = sorted(model.classes)
                    else:
                        model_classes = all_classes
                    
                    joint_probs = model.joint_prob(X_test[i:i+1], model_classes)
                    marginal = joint_probs.sum(axis=1, keepdims=True)
                    cond_probs = joint_probs / np.maximum(marginal, 1e-12)
                    
                    selected_classes = []
                    # Compute conformal score for each class
                    for c_idx, c in enumerate(model_classes):
                        test_score = self._aps_score(cond_probs, np.array([c_idx]), model_classes)[0]
                        
                        # Calculate p-value: proportion of calibration scores >= test score
                        cal_scores_j = self.calibration_scores[j]
                        p_value = self._compute_p_value(test_score, cal_scores_j)
                        
                        # Include class if p-value > alpha for this source
                        if p_value > self.alpha:
                            class_included[c] = True
                            selected_classes.append(c)

                    individual_info[j].append(selected_classes)

                # Set prediction_sets based on which classes are included
                prediction_sets[i] = class_included.astype(int)

            return prediction_sets, individual_info

        elif self.task == 'regression':
            n_test = len(X_test)
            direct_union_sets = []
            
            individual_sets_all = [[] for _ in range(len(self.source_models))]

            for i in range(n_test):
                all_intervals = []
                
                for j in valid_sources:
                    model = self.source_models[j]
                    mu = model.predict_mu(X_test[i:i+1])[0]
                    sigma = model.predict_sigma(X_test[i:i+1])[0]
                    
                    # Use individual source threshold for this source
                    threshold_j = self.thresholds[f'source_{j}']
                    half_width = threshold_j * sigma
                    
                    # Calculate p-value by checking if test score would be <= threshold
                    # For regression, we create an interval and check coverage
                    interval = {
                        'lower': mu - half_width,
                        'upper': mu + half_width,
                        'center': mu
                    }
                    individual_sets_all[j].append([interval['lower'], interval['upper']])  # for max-p aggregation
                    all_intervals.append(interval)
                
                
                if all_intervals:
                    min_lower = min(interval['lower'] for interval in all_intervals)
                    max_upper = max(interval['upper'] for interval in all_intervals)
                    mean_center = np.mean([interval['center'] for interval in all_intervals])
                    direct_union_sets.append({
                        'lower': min_lower,
                        'upper': max_upper, 
                        'center': mean_center
                    })
                else:
                    direct_union_sets.append({
                        'lower': np.nan,
                        'upper': np.nan,
                        'center': 0.0
                    })
            
            # Convert to arrays for consistency
            direct_union = {
                'lower': np.array([p['lower'] for p in direct_union_sets]),
                'upper': np.array([p['upper'] for p in direct_union_sets]),
                'center': np.array([p['center'] for p in direct_union_sets])
            }
            return direct_union, individual_sets_all

    def has_cqr(self) -> bool:
        """Return True if CQR models are trained and calibrated."""
        if not (self._enable_cqr and self._cqr_ready):
            return False
        return any(
            model.q_lo_model is not None and model.q_hi_model is not None and model.tau is not None
            for model in self._cqr_models
        )

    def cqr_predict_intervals_per_source(self, X_test: np.ndarray) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Return calibrated CQR intervals per source for the provided points."""
        if not self.has_cqr():
            return []
        X_test = np.asarray(X_test)
        intervals: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for model in self._cqr_models:
            if model.q_lo_model is None or model.q_hi_model is None or model.tau is None:
                intervals.append(None)
                continue
            lo, hi = model.predict_interval(X_test)
            intervals.append((lo, hi))
        return intervals

