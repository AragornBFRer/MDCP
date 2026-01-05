import argparse
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

sys.path.append('..')
sys.path.append('../model')

from simulation import MultiSourceSimulator
from data_utils import (
    combine_sources_three_way,
    get_data_split_summary,
    reconstruct_source_data,
)
from baseline import BaselineConformalPredictor
from eval_utils import (
    add_joint_pdf_to_source,
    evaluate_baseline_classification_comprehensive,
    evaluate_baseline_regression_comprehensive,
    evaluate_mdcp_classification_performance,
    evaluate_mdcp_regression_performance,
    generate_y_grid_classification,
    generate_y_grid_regression,
    _format_gamma_name,
    _split_mimic_sets,
    _run_mdcp_for_gamma,
    _summarize_metrics_for_logging,
    _score_gamma_candidate,
)
from density_utils import save_density_snapshot

try:
    from model.MDCP import (
        SourceModelClassification,
        SourceModelRegressionGaussian,
        PooledConditionalClassifier,
        PooledConditionalRegressionGaussian,
        aggregated_conformal_set_multi,
        compute_source_weights_from_sizes,
        fit_lambda_from_sources,
        precompute_calibration_cache,
    )
    MDCP_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - handled at runtime
    print(f"Warning: Could not import MDCP: {exc}")
    MDCP_AVAILABLE = False

from model.const import *  # noqa: F403,F401


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
n_sources = 3
n_classes = 6
n_samples_per_source = [2000, 2000, 2000]
n_features = 10
alpha = 0.1

DEFAULT_TEMPERATURES: List[float] = [0.5, 1.5, 2.5, 3.5, 4.5]

TRAIN_SIZE = TRUE_TRAIN_RATIO
CAL_SIZE = TRUE_CAL_RATIO
TEST_SIZE = TRUE_TEST_RATIO
MIMIC_CAL_RATIO = MIMIC_CALIBRATION_RATIO

DEFAULT_NUM_TRIALS = 5
LAMBDA_SAMPLE_LIMIT = 50

GAMMA_GRID: List[float] = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_temperature_value(temperature: float) -> str:
    formatted = f"{temperature:.4f}".rstrip('0').rstrip('.')
    return formatted if formatted else '0'


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Iterative MDCP evaluation with multi-trial aggregation.',
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help='Number of independent trials to execute.',
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=RANDOM_SEED,  # type: ignore[name-defined]
        help='Base random seed used for deriving trial seeds.',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='*',
        default=None,
        help='Optional explicit seeds for each trial (overrides base-seed logic).',
    )
    parser.add_argument(
        '--lambda-sample-limit',
        type=int,
        default=LAMBDA_SAMPLE_LIMIT,
        help='Maximum number of test points to snapshot lambda values from.',
    )
    parser.add_argument(
        '--temperatures',
        type=float,
        nargs='+',
        default=DEFAULT_TEMPERATURES,
        help='Grid of temperature values to evaluate for each trial.',
    )
    return parser.parse_args()


TEMPERATURE_EVAL_FOLDER = eval_out_relative('temperature')  # noqa: F405
TEMPERATURE_EVAL_LAMBDA_FOLDER = TEMPERATURE_EVAL_FOLDER / 'lambda'


def ensure_output_dirs() -> Dict[str, Path]:
    directories = {
        'base': ensure_project_dir(TEMPERATURE_EVAL_FOLDER),  # noqa: F405
        'classification': ensure_project_dir(TEMPERATURE_EVAL_FOLDER / 'classification'),  # noqa: F405
        'regression': ensure_project_dir(TEMPERATURE_EVAL_FOLDER / 'regression'),  # noqa: F405
        'lambda': ensure_project_dir(TEMPERATURE_EVAL_LAMBDA_FOLDER),
        'summaries': ensure_project_dir(TEMPERATURE_EVAL_FOLDER / 'summaries'),  # noqa: F405
    }
    return directories


def sample_lambda_values(lambda_model: Any, X_test: np.ndarray, sample_limit: int) -> Dict[str, Any]:
    if len(X_test) == 0:
        return {
            'lambda_values': np.empty((0, 0)),
            'test_points': np.empty((0, 0)),
            'sample_indices': np.array([], dtype=int),
            'sample_size': 0,
        }

    sample_size = min(sample_limit, len(X_test))
    sample_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    sample_points = X_test[sample_indices]
    sample_lambda_vals = lambda_model.lambda_at_x(sample_points)

    return {
        'lambda_values': sample_lambda_vals,
        'test_points': sample_points,
        'sample_indices': sample_indices.astype(int),
        'sample_size': int(sample_size),
    }


def log_split_summary(summary: Dict[str, Any], label: str) -> None:
    print(
        f"{label} split - Train: {summary['train_samples']} ({summary['train_ratio']:.1%}), "
        f"Cal: {summary['cal_samples']} ({summary['cal_ratio']:.1%}), "
        f"Test: {summary['test_samples']} ({summary['test_ratio']:.1%})",
    )


def prepare_baseline(
    task_type: str,
    trial_seed: int,
    X_sources_train: List[np.ndarray],
    Y_sources_train: List[np.ndarray],
    X_sources_cal: List[np.ndarray],
    Y_sources_cal: List[np.ndarray],
) -> Optional[BaselineConformalPredictor]:
    baseline = BaselineConformalPredictor(random_seed=trial_seed)

    valid_X_train = [X for X in X_sources_train if len(X) > 0]
    valid_Y_train = [Y for Y in Y_sources_train if len(Y) > 0]
    valid_X_cal = [X for X in X_sources_cal if len(X) > 0]
    valid_Y_cal = [Y for Y in Y_sources_cal if len(Y) > 0]

    if len(valid_X_train) == 0 or len(valid_X_cal) == 0:
        print(f"  No valid sources for baseline {task_type}.")
        return None

    try:
        baseline.train_source_models(valid_X_train, valid_Y_train, task=task_type)
        baseline.calibrate(valid_X_cal, valid_Y_cal, alpha=alpha)
        print(f"  Baseline {task_type} trained with {len(valid_X_train)} sources.")
        return baseline
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"  Error setting up baseline {task_type}: {exc}")
        traceback.print_exc()
        return None


def save_lambda_snapshot(
    task_type: str,
    trial_seed: int,
    lambda_payload: Optional[Dict[str, Any]],
    temperature: float,
    directories: Dict[str, Path],
) -> Optional[Path]:
    if not lambda_payload:
        return None

    lambda_payload = dict(lambda_payload)  # shallow copy
    temperature_label = format_temperature_value(temperature)
    lambda_payload.update(
        {
            'random_seed': trial_seed,
            'alpha': alpha,
            'n_sources': n_sources,
            'n_features': n_features,
            'temperature': float(temperature),
            'task': task_type,
        }
    )
    if task_type == 'classification':
        lambda_payload['n_classes'] = n_classes
        identifier = LAMBDA_CLS_FILE.format(  # noqa: F405
            random_seed=trial_seed,
            alpha=alpha,
            n_sources=n_sources,
            n_classes=n_classes,
            temperature=temperature_label,
        )
    else:
        identifier = LAMBDA_REG_FILE.format(  # noqa: F405
            random_seed=trial_seed,
            alpha=alpha,
            n_sources=n_sources,
            temperature=temperature_label,
        )

    lambda_path = directories['lambda'] / f"{identifier}.npz"
    np.savez(lambda_path, **lambda_payload)
    print(f"  Lambda data saved: {prefer_relative_path(lambda_path)}")  # noqa: F405
    return lambda_path


def run_classification_pipeline(
    trial_seed: int,
    X_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_cal: np.ndarray,
    Y_test: np.ndarray,
    source_train: np.ndarray,
    source_cal: np.ndarray,
    source_test: np.ndarray,
    params_cls: Dict[str, Any],
    lambda_sample_limit: int,
) -> Dict[str, Any]:
    X_sources_train, Y_sources_train = reconstruct_source_data(
        X_train, Y_train, source_train, n_sources
    )
    X_sources_cal, Y_sources_cal = reconstruct_source_data(
        X_cal, Y_cal, source_cal, n_sources
    )

    train_sizes = [len(X) for X in X_sources_train]
    cal_sizes = [len(X) for X in X_sources_cal]
    source_weights = compute_source_weights_from_sizes(train_sizes)
    mimic_ratio = MIMIC_CAL_RATIO

    print("\n=== MDCP Classification Setup ===")
    sources_cls: List[Any] = []
    for idx, (X_src, Y_src) in enumerate(zip(X_sources_train, Y_sources_train)):
        try:
            source_model = SourceModelClassification(X_src, Y_src)
            sources_cls.append(source_model)
            print(f"  Source {idx}: {len(X_src)} samples, classes={len(np.unique(Y_src))}")
        except Exception as exc:
            print(f"  Error constructing source model for source {idx}: {exc}")
            traceback.print_exc()

    gamma_results: List[Dict[str, Any]] = []
    lambda_snapshots: List[Dict[str, Any]] = []

    # Fit pooled p_data(y|x) on the pooled *training* split only.
    pooled_p_data_model = PooledConditionalClassifier(X_train, Y_train, learner='gbm')
    p_data_conditional_train = pooled_p_data_model.conditional_at_pairs(X_train, Y_train)

    mdcp_ready = False
    mimic_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    mimic_error: Optional[str] = None
    if len(sources_cls) >= 2:
        try:
            mimic_components = _split_mimic_sets(
                X_train,
                Y_train,
                source_train,
                mimic_ratio,
                trial_seed + 101,
                stratify=True,
            )
        except Exception as exc:
            mimic_error = str(exc)
            print(f"  Unable to perform gamma mimic split: {exc}")
            traceback.print_exc()

        if mimic_components is not None:
            (
                X_mimic_cal,
                X_mimic_test,
                Y_mimic_cal,
                Y_mimic_test,
                source_mimic_cal,
                source_mimic_test,
            ) = mimic_components
            X_sources_mimic_cal, Y_sources_mimic_cal = reconstruct_source_data(
                X_mimic_cal,
                Y_mimic_cal,
                source_mimic_cal,
                n_sources,
            )
            y_grid_mimic = generate_y_grid_classification(
                Y_sources_train + Y_sources_mimic_cal
            )
        else:
            X_sources_mimic_cal = []
            Y_sources_mimic_cal = []
            y_grid_mimic = None

        y_grid_true = generate_y_grid_classification(Y_sources_train + Y_sources_cal)

        for gamma_value in GAMMA_GRID:
            gamma_name = _format_gamma_name(gamma_value)
            entry: Dict[str, Any] = {
                'gamma1': gamma_value,
                'gamma2': gamma_value,
                'gamma3': 0.0,
                'gamma_name': gamma_name,
            }

            if mimic_components is not None and y_grid_mimic is not None:
                try:
                    mimic_metrics, _ = _run_mdcp_for_gamma(
                        gamma_value,
                        sources_cls,
                        X_train,
                        Y_train,
                        X_sources_mimic_cal,  # type: ignore[arg-type]
                        Y_sources_mimic_cal,  # type: ignore[arg-type]
                        X_mimic_test,
                        Y_mimic_test,
                        y_grid_mimic,
                        alpha,
                        source_weights,
                        'classification',
                        source_mimic_test,
                        verbose=False,
                        p_data_conditional=p_data_conditional_train,
                    )
                    entry['mimic_metrics'] = mimic_metrics
                    entry['mimic_efficiency'] = _score_gamma_candidate(mimic_metrics, 'classification')
                    entry['mimic_summary'] = _summarize_metrics_for_logging(
                        mimic_metrics, 'classification'
                    )
                except Exception as exc:
                    entry['mimic_error'] = str(exc)
                    print(f"  Gamma {gamma_value} mimic evaluation failed: {exc}")
                    traceback.print_exc()
            else:
                entry['mimic_error'] = mimic_error

            try:
                true_metrics, lambda_model_true = _run_mdcp_for_gamma(
                    gamma_value,
                    sources_cls,
                    X_train,
                    Y_train,
                    X_sources_cal,
                    Y_sources_cal,
                    X_test,
                    Y_test,
                    y_grid_true,
                    alpha,
                    source_weights,
                    'classification',
                    source_test,
                    verbose=False,
                    p_data_conditional=p_data_conditional_train,
                )
                entry['true_metrics'] = true_metrics
                entry['true_efficiency'] = _score_gamma_candidate(true_metrics, 'classification')
                entry['true_summary'] = _summarize_metrics_for_logging(
                    true_metrics, 'classification'
                )

                if lambda_model_true is not None:
                    snapshot = sample_lambda_values(
                        lambda_model_true,
                        X_test,
                        lambda_sample_limit,
                    )
                    snapshot.update(
                        {
                            'gamma1': gamma_value,
                            'gamma2': gamma_value,
                            'gamma3': 0.0,
                            'gamma_name': gamma_name,
                            'task': 'classification',
                        }
                    )
                    lambda_snapshots.append(snapshot)
                mdcp_ready = True
            except Exception as exc:
                entry['true_error'] = str(exc)
                print(f"  Gamma {gamma_value} true evaluation failed: {exc}")
                traceback.print_exc()

            gamma_results.append(entry)
    else:
        print(f"  Not enough classification sources (found {len(sources_cls)}).")

    lambda_data = None
    if gamma_results:
        lambda_data = {
            'task': 'classification',
            'per_gamma_results': np.array(gamma_results, dtype=object),
            'lambda_snapshots': np.array(lambda_snapshots, dtype=object),
        }

    print("\n=== Baseline Classification ===")
    baseline = prepare_baseline(
        task_type='classification',
        trial_seed=trial_seed,
        X_sources_train=X_sources_train,
        Y_sources_train=Y_sources_train,
        X_sources_cal=X_sources_cal,
        Y_sources_cal=Y_sources_cal,
    )

    classification_results: Dict[str, Any] = {}
    baseline_comprehensive: Optional[Dict[str, Any]] = None

    if baseline:
        try:
            baseline_comprehensive = evaluate_baseline_classification_comprehensive(
                baseline,
                X_test,
                Y_test,
                source_test,
                alpha,
            )
            if 'Max_Aggregated' in baseline_comprehensive:
                overall = baseline_comprehensive['Max_Aggregated'].get('Overall')
                if overall:
                    classification_results['Max Aggregation'] = overall
            if 'Source_0' in baseline_comprehensive:
                overall = baseline_comprehensive['Source_0'].get('Overall')
                if overall:
                    classification_results['Single Source'] = overall

            print("\n--- Baseline Classification Comprehensive Metrics ---")
            for construct_method, eval_groups in baseline_comprehensive.items():
                print(f"{construct_method}:")
                for group, metrics in eval_groups.items():
                    coverage = metrics.get('coverage', float('nan'))
                    set_size = metrics.get('avg_set_size', float('nan'))
                    print(
                        f"  {group}: coverage={coverage:.3f}, avg_set_size={set_size:.3f}"
                    )
        except Exception as exc:
            print(f"  Error evaluating baseline classification: {exc}")
            traceback.print_exc()

    mdcp_metrics_map = {
        entry['gamma_name']: entry['true_metrics']
        for entry in gamma_results
        if 'true_metrics' in entry
    }

    if mdcp_metrics_map:
        classification_results['MDCP'] = mdcp_metrics_map
        print("\n--- MDCP Classification (per gamma) ---")
        for gamma_name, metrics in mdcp_metrics_map.items():
            coverage = metrics.get('coverage', float('nan'))
            set_size = metrics.get('avg_set_size', float('nan'))
            efficiency = metrics.get('avg_set_size', float('nan'))
            print(
                f"  {gamma_name}: coverage={coverage:.3f}, avg_set_size={set_size:.3f}, efficiency={efficiency:.3f}"
            )
    else:
        print("  No successful MDCP classification evaluations recorded.")

    return {
        'results': classification_results,
        'baseline_comprehensive': baseline_comprehensive,
        'lambda_data': lambda_data,
        'train_samples_per_source': train_sizes,
        'cal_samples_per_source': cal_sizes,
        'simulation_params': params_cls,
        'gamma_results': gamma_results,
        'lambda_snapshots': lambda_snapshots,
    }


def run_regression_pipeline(
    trial_seed: int,
    X_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_cal: np.ndarray,
    Y_test: np.ndarray,
    source_train: np.ndarray,
    source_cal: np.ndarray,
    source_test: np.ndarray,
    params_reg: Dict[str, Any],
    lambda_sample_limit: int,
) -> Dict[str, Any]:
    X_sources_train, Y_sources_train = reconstruct_source_data(
        X_train, Y_train, source_train, n_sources
    )
    X_sources_cal, Y_sources_cal = reconstruct_source_data(
        X_cal, Y_cal, source_cal, n_sources
    )

    train_sizes = [len(X) for X in X_sources_train]
    cal_sizes = [len(X) for X in X_sources_cal]
    source_weights = compute_source_weights_from_sizes(train_sizes)
    mimic_ratio = MIMIC_CAL_RATIO

    print("\n=== MDCP Regression Setup ===")
    sources_reg: List[Any] = []
    for idx, (X_src, Y_src) in enumerate(zip(X_sources_train, Y_sources_train)):
        try:
            source_model = SourceModelRegressionGaussian(X_src, Y_src)
            sources_reg.append(source_model)
            print(f"  Source {idx}: {len(X_src)} samples")
        except Exception as exc:
            print(f"  Error constructing regression source model {idx}: {exc}")
            traceback.print_exc()

    gamma_results: List[Dict[str, Any]] = []
    lambda_snapshots: List[Dict[str, Any]] = []
    mdcp_ready = False

    # Fit pooled p_data(y|x) on the pooled *training* split only.
    pooled_p_data_model = PooledConditionalRegressionGaussian(X_train, Y_train, learner='gbm')
    p_data_conditional_train = pooled_p_data_model.conditional_at_pairs(X_train, Y_train)

    mimic_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    mimic_error: Optional[str] = None
    if len(sources_reg) >= 2:
        try:
            mimic_components = _split_mimic_sets(
                X_train,
                Y_train,
                source_train,
                mimic_ratio,
                trial_seed + 303,
                stratify=False,
            )
        except Exception as exc:
            mimic_error = str(exc)
            print(f"  Unable to perform regression mimic split: {exc}")
            traceback.print_exc()

        if mimic_components is not None:
            (
                X_mimic_cal,
                X_mimic_test,
                Y_mimic_cal,
                Y_mimic_test,
                source_mimic_cal,
                source_mimic_test,
            ) = mimic_components
            X_sources_mimic_cal, Y_sources_mimic_cal = reconstruct_source_data(
                X_mimic_cal,
                Y_mimic_cal,
                source_mimic_cal,
                n_sources,
            )
            y_grid_mimic = generate_y_grid_regression(
                Y_sources_train + Y_sources_mimic_cal
            )
        else:
            X_sources_mimic_cal = []
            Y_sources_mimic_cal = []
            y_grid_mimic = None

        y_grid_true = generate_y_grid_regression(Y_sources_train + Y_sources_cal)

        for gamma_value in GAMMA_GRID:
            gamma_name = _format_gamma_name(gamma_value)
            entry: Dict[str, Any] = {
                'gamma1': gamma_value,
                'gamma2': gamma_value,
                'gamma3': 0.0,
                'gamma_name': gamma_name,
            }

            if mimic_components is not None and y_grid_mimic is not None:
                try:
                    mimic_metrics, _ = _run_mdcp_for_gamma(
                        gamma_value,
                        sources_reg,
                        X_train,
                        Y_train,
                        X_sources_mimic_cal,  # type: ignore[arg-type]
                        Y_sources_mimic_cal,  # type: ignore[arg-type]
                        X_mimic_test,
                        Y_mimic_test,
                        y_grid_mimic,
                        alpha,
                        source_weights,
                        'regression',
                        source_mimic_test,
                        verbose=False,
                        p_data_conditional=p_data_conditional_train,
                    )
                    entry['mimic_metrics'] = mimic_metrics
                    entry['mimic_efficiency'] = _score_gamma_candidate(mimic_metrics, 'regression')
                    entry['mimic_summary'] = _summarize_metrics_for_logging(
                        mimic_metrics, 'regression'
                    )
                except Exception as exc:
                    entry['mimic_error'] = str(exc)
                    print(f"  Gamma {gamma_value} regression mimic evaluation failed: {exc}")
                    traceback.print_exc()
            else:
                entry['mimic_error'] = mimic_error

            try:
                true_metrics, lambda_model_true = _run_mdcp_for_gamma(
                    gamma_value,
                    sources_reg,
                    X_train,
                    Y_train,
                    X_sources_cal,
                    Y_sources_cal,
                    X_test,
                    Y_test,
                    y_grid_true,
                    alpha,
                    source_weights,
                    'regression',
                    source_test,
                    verbose=False,
                    p_data_conditional=p_data_conditional_train,
                )
                entry['true_metrics'] = true_metrics
                entry['true_efficiency'] = _score_gamma_candidate(true_metrics, 'regression')
                entry['true_summary'] = _summarize_metrics_for_logging(
                    true_metrics, 'regression'
                )

                if lambda_model_true is not None:
                    snapshot = sample_lambda_values(
                        lambda_model_true,
                        X_test,
                        lambda_sample_limit,
                    )
                    snapshot.update(
                        {
                            'gamma1': gamma_value,
                            'gamma2': gamma_value,
                            'gamma3': 0.0,
                            'gamma_name': gamma_name,
                            'task': 'regression',
                        }
                    )
                    lambda_snapshots.append(snapshot)
                mdcp_ready = True
            except Exception as exc:
                entry['true_error'] = str(exc)
                print(f"  Gamma {gamma_value} regression true evaluation failed: {exc}")
                traceback.print_exc()

            gamma_results.append(entry)
    else:
        print(f"  Not enough regression sources (found {len(sources_reg)}).")

    lambda_data = None
    if gamma_results:
        lambda_data = {
            'task': 'regression',
            'per_gamma_results': np.array(gamma_results, dtype=object),
            'lambda_snapshots': np.array(lambda_snapshots, dtype=object),
        }

    print("\n=== Baseline Regression ===")
    baseline = prepare_baseline(
        task_type='regression',
        trial_seed=trial_seed,
        X_sources_train=X_sources_train,
        Y_sources_train=Y_sources_train,
        X_sources_cal=X_sources_cal,
        Y_sources_cal=Y_sources_cal,
    )

    regression_results: Dict[str, Any] = {}
    baseline_comprehensive: Optional[Dict[str, Any]] = None

    if baseline:
        try:
            baseline_comprehensive = evaluate_baseline_regression_comprehensive(
                baseline,
                X_test,
                Y_test,
                source_test,
                alpha,
            )
            if 'Max_Aggregated' in baseline_comprehensive:
                overall = baseline_comprehensive['Max_Aggregated'].get('Overall')
                if overall:
                    regression_results['Max Aggregation'] = overall
            if 'Source_0' in baseline_comprehensive:
                overall = baseline_comprehensive['Source_0'].get('Overall')
                if overall:
                    regression_results['Single Source'] = overall

            print("\n--- Baseline Regression Comprehensive Metrics ---")
            for construct_method, eval_groups in baseline_comprehensive.items():
                print(f"{construct_method}:")
                for group, metrics in eval_groups.items():
                    coverage = metrics.get('coverage', float('nan'))
                    width = metrics.get('avg_width', float('nan'))
                    print(f"  {group}: coverage={coverage:.3f}, avg_width={width:.3f}")
        except Exception as exc:
            print(f"  Error evaluating baseline regression: {exc}")
            traceback.print_exc()

    mdcp_metrics_map = {
        entry['gamma_name']: entry['true_metrics']
        for entry in gamma_results
        if 'true_metrics' in entry
    }

    if mdcp_metrics_map:
        regression_results['MDCP'] = mdcp_metrics_map
        print("\n--- MDCP Regression (per gamma) ---")
        for gamma_name, metrics in mdcp_metrics_map.items():
            coverage = metrics.get('coverage', float('nan'))
            width = metrics.get('avg_width', float('nan'))
            print(f"  {gamma_name}: coverage={coverage:.3f}, avg_width={width:.3f}")
    else:
        print("  No successful MDCP regression evaluations recorded.")

    return {
        'results': regression_results,
        'baseline_comprehensive': baseline_comprehensive,
        'lambda_data': lambda_data,
        'train_samples_per_source': train_sizes,
        'cal_samples_per_source': cal_sizes,
        'simulation_params': params_reg,
        'gamma_results': gamma_results,
        'lambda_snapshots': lambda_snapshots,
    }


def run_trial_for_temperature(
    trial_index: int,
    trial_seed: int,
    temperature: float,
    directories: Dict[str, Path],
    lambda_sample_limit: int,
) -> Dict[str, Any]:
    temperature_label = format_temperature_value(temperature)
    temperature_token = temperature_label.replace('-', 'm').replace('.', 'p')

    print("\n" + '-' * 70)
    print(f"Evaluating temperature {temperature_label}")
    print('-' * 70)

    np.random.seed(trial_seed)
    simulator = MultiSourceSimulator(random_seed=trial_seed, temperature=temperature)

    X_sources_cls, Y_sources_cls, params_cls = simulator.generate_multisource_classification(
        n_sources=n_sources,
        n_samples_per_source=n_samples_per_source,
        n_features=n_features,
        n_classes=n_classes,
    )
    X_sources_reg, Y_sources_reg, params_reg = simulator.generate_multisource_regression(
        n_sources=n_sources,
        n_samples_per_source=n_samples_per_source,
        n_features=n_features,
        apply_temperature_noise=True,
    )

    dataset_id = f"trial{trial_index:02d}_seed{trial_seed}_temp{temperature_token}"
    density_cls_path = save_density_snapshot(
        script_tag='evaluation_temperature_rand_avg',
        task='classification',
        dataset_id=dataset_id,
        X_sources=X_sources_cls,
        Y_sources=Y_sources_cls,
        simulation_params=params_cls,
        random_seed=trial_seed,
        temperature=temperature,
        extra_metadata={
            'alpha': alpha,
            'trial_index': trial_index,
            'n_classes': n_classes,
            'temperature_label': temperature_label,
        },
    )
    density_reg_path = save_density_snapshot(
        script_tag='evaluation_temperature_rand_avg',
        task='regression',
        dataset_id=dataset_id,
        X_sources=X_sources_reg,
        Y_sources=Y_sources_reg,
        simulation_params=params_reg,
        random_seed=trial_seed,
        temperature=temperature,
        extra_metadata={
            'alpha': alpha,
            'trial_index': trial_index,
            'temperature_label': temperature_label,
        },
    )

    print(f"\nClassification data summary (temp={temperature_label}):")
    cls_summary = simulator.get_data_summary(X_sources_cls, Y_sources_cls, 'classification')
    print(f"  Total samples: {cls_summary['total_samples']}")
    print(f"  Classes: {cls_summary['unique_classes']}")
    print(f"  Class distributions: {cls_summary['class_distributions']}")

    print(f"\nRegression data summary (temp={temperature_label}):")
    reg_summary = simulator.get_data_summary(X_sources_reg, Y_sources_reg, 'regression')
    print(f"  Total samples: {reg_summary['total_samples']}")
    for idx, stats in enumerate(reg_summary['target_stats']):
        print(f"  Source {idx + 1}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    print("\nSplitting data into train/calibration/test sets...")

    X_train_cls, X_cal_cls, X_test_cls, Y_train_cls, Y_cal_cls, Y_test_cls, source_train_cls, source_cal_cls, source_test_cls = combine_sources_three_way(
        X_sources_cls,
        Y_sources_cls,
        train_size=TRAIN_SIZE,
        cal_size=CAL_SIZE,
        test_size=TEST_SIZE,
        stratify=True,
    )

    X_train_reg, X_cal_reg, X_test_reg, Y_train_reg, Y_cal_reg, Y_test_reg, source_train_reg, source_cal_reg, source_test_reg = combine_sources_three_way(
        X_sources_reg,
        Y_sources_reg,
        train_size=TRAIN_SIZE,
        cal_size=CAL_SIZE,
        test_size=TEST_SIZE,
        stratify=False,
    )

    cls_split_summary = get_data_split_summary(
        X_train_cls,
        X_cal_cls,
        X_test_cls,
        Y_train_cls,
        Y_cal_cls,
        Y_test_cls,
        'classification',
    )
    reg_split_summary = get_data_split_summary(
        X_train_reg,
        X_cal_reg,
        X_test_reg,
        Y_train_reg,
        Y_cal_reg,
        Y_test_reg,
        'regression',
    )

    log_split_summary(cls_split_summary, 'Classification')
    log_split_summary(reg_split_summary, 'Regression')

    classification_payload = run_classification_pipeline(
        trial_seed=trial_seed,
        X_train=X_train_cls,
        X_cal=X_cal_cls,
        X_test=X_test_cls,
        Y_train=Y_train_cls,
        Y_cal=Y_cal_cls,
        Y_test=Y_test_cls,
        source_train=source_train_cls,
        source_cal=source_cal_cls,
        source_test=source_test_cls,
        params_cls=params_cls,
        lambda_sample_limit=lambda_sample_limit,
    )

    regression_payload = run_regression_pipeline(
        trial_seed=trial_seed,
        X_train=X_train_reg,
        X_cal=X_cal_reg,
        X_test=X_test_reg,
        Y_train=Y_train_reg,
        Y_cal=Y_cal_reg,
        Y_test=Y_test_reg,
        source_train=source_train_reg,
        source_cal=source_cal_reg,
        source_test=source_test_reg,
        params_reg=params_reg,
        lambda_sample_limit=lambda_sample_limit,
    )

    trial_metadata = {
        'trial_index': trial_index,
        'trial_seed': trial_seed,
        'alpha': alpha,
        'n_sources': n_sources,
        'n_classes': n_classes,
        'n_features': n_features,
        'n_samples_per_source': n_samples_per_source,
        'temperature': float(temperature),
        'temperature_label': temperature_label,
        'train_size': TRAIN_SIZE,
        'cal_size': CAL_SIZE,
        'test_size': TEST_SIZE,
    }

    classification_results_rel: Optional[str] = None
    if classification_payload['results']:
        cls_identifier = (
            f"trial_{trial_index:03d}_seed_{trial_seed}_alpha_{alpha}_"
            f"sources_{n_sources}_classes_{n_classes}_temperature_{temperature_label}"
        )
        cls_path = directories['classification'] / f"{cls_identifier}.npz"
        np.savez(
            cls_path,
            metadata=trial_metadata,
            results=classification_payload['results'],
            baseline_comprehensive=classification_payload['baseline_comprehensive'],
            test_samples=len(X_test_cls),
            train_samples_per_source=classification_payload['train_samples_per_source'],
            cal_samples_per_source=classification_payload['cal_samples_per_source'],
            simulation_params=classification_payload['simulation_params'],
            gamma_results=np.array(classification_payload.get('gamma_results', []), dtype=object),
            lambda_snapshots=np.array(classification_payload.get('lambda_snapshots', []), dtype=object),
            lambda_data=classification_payload.get('lambda_data'),
        )
        cls_rel = prefer_relative_path(cls_path)
        print(f"  Classification results saved: {cls_rel}")  # noqa: F405
        classification_results_rel = str(cls_rel)

    regression_results_rel: Optional[str] = None
    if regression_payload['results']:
        reg_identifier = (
            f"trial_{trial_index:03d}_seed_{trial_seed}_alpha_{alpha}_"
            f"sources_{n_sources}_temperature_{temperature_label}"
        )
        reg_path = directories['regression'] / f"{reg_identifier}.npz"
        np.savez(
            reg_path,
            metadata=trial_metadata,
            results=regression_payload['results'],
            baseline_comprehensive=regression_payload['baseline_comprehensive'],
            test_samples=len(X_test_reg),
            train_samples_per_source=regression_payload['train_samples_per_source'],
            cal_samples_per_source=regression_payload['cal_samples_per_source'],
            simulation_params=regression_payload['simulation_params'],
            gamma_results=np.array(regression_payload.get('gamma_results', []), dtype=object),
            lambda_snapshots=np.array(regression_payload.get('lambda_snapshots', []), dtype=object),
            lambda_data=regression_payload.get('lambda_data'),
        )
        reg_rel = prefer_relative_path(reg_path)
        print(f"  Regression results saved: {reg_rel}")  # noqa: F405
        regression_results_rel = str(reg_rel)

    lambda_paths = {
        'classification': save_lambda_snapshot(
            'classification', trial_seed, classification_payload['lambda_data'], temperature, directories
        ),
        'regression': save_lambda_snapshot(
            'regression', trial_seed, regression_payload['lambda_data'], temperature, directories
        ),
    }
    lambda_paths_rel = {
        key: str(prefer_relative_path(path)) if path else None  # noqa: F405
        for key, path in lambda_paths.items()
    }

    density_cls_rel = str(prefer_relative_path(density_cls_path))  # noqa: F405
    density_reg_rel = str(prefer_relative_path(density_reg_path))  # noqa: F405

    return {
        'temperature': float(temperature),
        'temperature_label': temperature_label,
        'metadata': trial_metadata,
        'classification': classification_payload,
        'regression': regression_payload,
        'classification_split_summary': cls_split_summary,
        'regression_split_summary': reg_split_summary,
        'paths': {
            'classification_results': classification_results_rel,
            'regression_results': regression_results_rel,
            'lambda_classification': lambda_paths_rel['classification'],
            'lambda_regression': lambda_paths_rel['regression'],
            'density_classification': density_cls_rel,
            'density_regression': density_reg_rel,
        },
    }


def run_trial(
    trial_index: int,
    trial_seed: int,
    temperatures: List[float],
    directories: Dict[str, Path],
    lambda_sample_limit: int,
) -> Dict[str, Any]:
    print("\n" + '=' * 70)
    print(f"Starting trial {trial_index} with seed {trial_seed}")
    print('=' * 70)

    temperature_grid = [float(t) for t in temperatures]
    temperature_summaries: List[Dict[str, Any]] = []

    for temperature in temperature_grid:
        summary = run_trial_for_temperature(
            trial_index=trial_index,
            trial_seed=trial_seed,
            temperature=temperature,
            directories=directories,
            lambda_sample_limit=lambda_sample_limit,
        )
        temperature_summaries.append(summary)

    trial_summary = {
        'trial_index': trial_index,
        'trial_seed': trial_seed,
        'alpha': alpha,
        'n_sources': n_sources,
        'n_classes': n_classes,
        'n_features': n_features,
        'n_samples_per_source': n_samples_per_source,
        'temperatures': temperature_grid,
        'temperature_labels': [format_temperature_value(t) for t in temperature_grid],
        'train_size': TRAIN_SIZE,
        'cal_size': CAL_SIZE,
        'test_size': TEST_SIZE,
        'per_temperature': temperature_summaries,
    }

    summary_path = directories['summaries'] / f"trial_{trial_index:03d}_seed_{trial_seed}_summary.npz"
    np.savez_compressed(summary_path, trial_summary=np.array([trial_summary], dtype=object))
    print(f"  Trial summary saved: {prefer_relative_path(summary_path)}")  # noqa: F405

    print('=' * 70)
    print(f"Trial {trial_index} complete.")
    print('=' * 70)

    return trial_summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"NumPy version: {np.__version__}")
    print(f"MDCP available: {MDCP_AVAILABLE}")

    if not MDCP_AVAILABLE:
        print('Error: MDCP modules could not be imported. Aborting.')
        sys.exit(1)

    if args.seeds is not None and len(args.seeds) != args.num_trials:
        raise ValueError('The number of explicit seeds must match --num-trials.')

    directories = ensure_output_dirs()

    temperature_grid = [float(t) for t in args.temperatures]
    temperature_display = ', '.join(format_temperature_value(t) for t in temperature_grid)
    print(f"Temperature grid: {temperature_display}")

    base_seed = args.base_seed
    rng = np.random.default_rng(int(time.time()) % 100000)

    trial_summaries: List[Dict[str, Any]] = []

    for trial_idx in range(1, args.num_trials + 1):
        if args.seeds is not None:
            trial_seed = args.seeds[trial_idx - 1]
        else:
            offset = int(rng.integers(1, 10000))
            base_seed += offset
            trial_seed = base_seed

        summary = run_trial(
            trial_index=trial_idx,
            trial_seed=trial_seed,
            temperatures=temperature_grid,
            directories=directories,
            lambda_sample_limit=args.lambda_sample_limit,
        )
        trial_summaries.append(summary)

    summary_metadata = {
        'num_trials': args.num_trials,
        'alpha': alpha,
        'temperature_grid': temperature_grid,
        'temperature_labels': [format_temperature_value(t) for t in temperature_grid],
        'n_sources': n_sources,
        'n_classes': n_classes,
        'n_features': n_features,
        'timestamp': time.time(),
    }

    temperature_token = '-'.join(
        format_temperature_value(t).replace('-', 'm').replace('.', 'p') for t in temperature_grid
    )
    if not temperature_token:
        temperature_token = 'default'

    batch_summary_path = directories['summaries'] / (
        f"multi_trial_summary_temps_{temperature_token}_alpha_{alpha}_trials_{args.num_trials}.npz"
    )
    np.savez_compressed(
        batch_summary_path,
        trials=np.array(trial_summaries, dtype=object),
        metadata=np.array([summary_metadata], dtype=object),
    )
    print(f"\nSaved multi-trial summary to {prefer_relative_path(batch_summary_path)}")  # noqa: F405
    print('All trials completed successfully.')


if __name__ == '__main__':
    main()