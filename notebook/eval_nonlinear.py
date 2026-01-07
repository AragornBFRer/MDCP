"""
Individual Nonlinear Term Evaluation Script

Classification terms tested individually:
- interaction, sinusoid, softplus

Regression terms tested individually:  
- interaction, sinusoid, softplus
"""

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

sys.path.append('..')
sys.path.append('../model')

from simulation_nonlinear import MultiSourceSimulator
from data_utils import combine_sources_three_way, get_data_split_summary, reconstruct_source_data
from baseline import BaselineConformalPredictor
from eval_utils import (
    evaluate_baseline_classification_comprehensive,
    evaluate_baseline_regression_comprehensive,
    evaluate_mdcp_classification_performance,
    evaluate_mdcp_regression_performance,
    add_joint_pdf_to_source,
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
        fit_lambda_from_sources,
        aggregated_conformal_set_multi,
        compute_source_weights_from_sizes,
        precompute_calibration_cache,
    )
    MDCP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MDCP: {e}")
    MDCP_AVAILABLE = False

from model.const import *
print(f"NumPy version: {np.__version__}")
print(f"MDCP available: {MDCP_AVAILABLE}")

# Global parameters
n_sources = 3
n_classes = 6
n_samples_per_source = [2000, 2000, 2000]
n_features = 10
alpha = 0.1
TEMPERATURE = 2.5

# Data split ratios sourced from model.const for reuse
train_size = TRUE_TRAIN_RATIO
cal_size = TRUE_CAL_RATIO
test_size = TRUE_TEST_RATIO

STANDARDIZE_FEATURES = True

GAMMA_GRID: List[float] = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def aggregate_mdcp_metrics(
    mdcp_metrics_map: Dict[str, Dict[str, Any]],
    task_type: str,
) -> Optional[Dict[str, float]]:
    if not mdcp_metrics_map:
        return None

    efficiency_key = 'avg_set_size' if task_type == 'classification' else 'avg_width'

    metric_entries = [
        metrics for metrics in mdcp_metrics_map.values() if isinstance(metrics, dict)
    ]
    if not metric_entries:
        return None

    coverage_vals = [float(metrics.get('coverage', np.nan)) for metrics in metric_entries]
    efficiency_vals = [
        float(metrics.get(efficiency_key, np.nan)) for metrics in metric_entries
    ]

    if np.all(np.isnan(coverage_vals)) or np.all(np.isnan(efficiency_vals)):
        return None

    aggregated: Dict[str, float] = {
        'coverage': float(np.nanmean(coverage_vals)),
        efficiency_key: float(np.nanmean(efficiency_vals)),
    }
    return aggregated


# Define individual nonlinear term configurations
CLASSIFICATION_TERMS: Dict[str, Dict[str, Any]] = {
    'linear': {
        'use_nonlinear': False,
        'config': None,
    },
    'interaction': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'interaction': 2.0,
                'sinusoid': 0.0,
                'softplus': 0.0,
            }
        },
    },
    'sinusoid': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'interaction': 0.0,
                'sinusoid': 2.0,
                'softplus': 0.0,
            }
        },
    },
    'softplus': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'interaction': 0.0,
                'sinusoid': 0.0,
                'softplus': 2.0,
            }
        },
    },
}

REGRESSION_TERMS: Dict[str, Dict[str, Any]] = {
    'linear': {
        'use_nonlinear': False,
        'config': None,
    },
    'interaction': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'quadratic': 0.0,
                'interaction': 2.0,
                'sinusoid': 0.0,
                'softplus': 0.0,
            }
        },
    },
    'sinusoid': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'quadratic': 0.0,
                'interaction': 0.0,
                'sinusoid': 2.0,
                'softplus': 0.0,
            }
        },
    },
    'softplus': {
        'use_nonlinear': True,
        'config': {
            'multipliers': {
                'quadratic': 0.0,
                'interaction': 0.0,
                'sinusoid': 0.0,
                'softplus': 2.0,
            }
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run multi-trial nonlinear evaluation for MDCP baselines and MDCP models'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=5,
        help='Number of independent trials to execute'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=RANDOM_SEED,
        help='Base random seed used to derive per-trial seeds'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='*',
        default=None,
        help='Explicit trial seeds (overrides base-seed logic)'
    )
    return parser.parse_args()


def run_single_evaluation(
    term_name: str,
    term_config: Dict[str, Any],
    task_type: str,
    trial_seed: int,
    trial_index: int,
) -> Dict[str, Any]:
    """
    Run evaluation for a single nonlinear term configuration.
    """
    print(f"\n{'=' * 70}")
    print(f"Trial {trial_index} | Testing {task_type.upper()} - {term_name.upper()} term")
    print(f"{'=' * 70}")

    np.random.seed(trial_seed)

    simulator = MultiSourceSimulator(random_seed=trial_seed, temperature=TEMPERATURE)

    if task_type == 'classification':
        X_sources, Y_sources, params = simulator.generate_multisource_classification(
            n_sources=n_sources,
            n_samples_per_source=n_samples_per_source,
            n_features=n_features,
            n_classes=n_classes,
            use_nonlinear=term_config['use_nonlinear'],
            standardize_features=STANDARDIZE_FEATURES,
            nonlinear_config=term_config['config'],
        )
    else:
        X_sources, Y_sources, params = simulator.generate_multisource_regression(
            n_sources=n_sources,
            n_samples_per_source=n_samples_per_source,
            n_features=n_features,
            use_nonlinear=term_config['use_nonlinear'],
            standardize_features=STANDARDIZE_FEATURES,
            nonlinear_config=term_config['config'],
        )

    dataset_id = f"trial{trial_index:02d}_seed{trial_seed}_{term_name}"
    save_density_snapshot(
        script_tag='evaluation_compact_nonlinear_rand_avg',
        task=task_type,
        dataset_id=dataset_id,
        X_sources=X_sources,
        Y_sources=Y_sources,
        simulation_params=params,
        random_seed=trial_seed,
        temperature=TEMPERATURE,
        extra_metadata={
            'alpha': alpha,
            'term_name': term_name,
            'trial_index': trial_index,
            'use_nonlinear': term_config['use_nonlinear'],
            'nonlinear_config': term_config['config'],
            'standardize_features': STANDARDIZE_FEATURES,
            'n_classes': n_classes if task_type == 'classification' else None,
        },
    )

    print(f"Generated {task_type} data with {term_name} nonlinear term")
    if term_config['use_nonlinear']:
        print(f"Nonlinear config: {term_config['config']['multipliers']}")

    if task_type == 'classification':
        X_train, X_cal, X_test, Y_train, Y_cal, Y_test, \
            source_train, source_cal, source_test = combine_sources_three_way(
                X_sources,
                Y_sources,
                train_size=train_size,
                cal_size=cal_size,
                test_size=test_size,
                stratify=True,
            )
    else:
        X_train, X_cal, X_test, Y_train, Y_cal, Y_test, \
            source_train, source_cal, source_test = combine_sources_three_way(
                X_sources,
                Y_sources,
                train_size=train_size,
                cal_size=cal_size,
                test_size=test_size,
                stratify=False,
            )

    print(f"Data split - Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

    X_sources_train, Y_sources_train = reconstruct_source_data(
        X_train, Y_train, source_train, n_sources
    )
    X_sources_cal, Y_sources_cal = reconstruct_source_data(
        X_cal, Y_cal, source_cal, n_sources
    )

    train_sizes = [len(X) for X in X_sources_train]
    source_weights = compute_source_weights_from_sizes(train_sizes)

    results: Dict[str, Any] = {
        'metadata': {
            'term_name': term_name,
            'task_type': task_type,
            'trial_seed': trial_seed,
            'trial_index': trial_index,
            'use_nonlinear': term_config['use_nonlinear'],
            'nonlinear_config': term_config['config'],
            'n_sources': n_sources,
            'n_classes': n_classes if task_type == 'classification' else None,
            'n_features': n_features,
            'n_samples_per_source': n_samples_per_source,
            'temperature': TEMPERATURE,
            'alpha': alpha,
            'train_size': train_size,
            'cal_size': cal_size,
            'test_size': test_size,
            'source_weights': source_weights.tolist(),
            'generation_params': params,
        },
        'data_info': {
            'train_samples': len(X_train),
            'cal_samples': len(X_cal),
            'test_samples': len(X_test),
            'train_samples_per_source': [len(X) for X in X_sources_train],
            'cal_samples_per_source': [len(X) for X in X_sources_cal],
        },
    }

    mdcp_ready = False
    mdcp_sources: List[Any] = []
    if MDCP_AVAILABLE:
        try:
            print(f"\n=== MDCP {task_type.title()} Setup ===")
            for j, (X_j, Y_j) in enumerate(zip(X_sources_train, Y_sources_train)):
                if task_type == 'classification':
                    source_model = SourceModelClassification(X_j, Y_j)
                else:
                    source_model = SourceModelRegressionGaussian(X_j, Y_j)
                mdcp_sources.append(source_model)
                print(f"  Source {j}: {len(X_j)} training samples")

            mdcp_ready = len(mdcp_sources) >= 2
            if not mdcp_ready:
                print(f"  Not enough sources for MDCP (found {len(mdcp_sources)})")
        except Exception as exc:
            print(f"  Error in MDCP setup: {exc}")
            traceback.print_exc()
            mdcp_sources = []
            mdcp_ready = False
    else:
        print("  Skipping MDCP setup because MDCP modules are unavailable")

    baseline_ready = False
    baseline_results: Dict[str, Any] = {}
    try:
        print(f"\n=== Baseline {task_type.title()} Setup ===")
        baseline = BaselineConformalPredictor(random_seed=trial_seed)

        valid_X_train = [X for X in X_sources_train if len(X) > 0]
        valid_Y_train = [Y for Y in Y_sources_train if len(Y) > 0]
        valid_X_cal = [X for X in X_sources_cal if len(X) > 0]
        valid_Y_cal = [Y for Y in Y_sources_cal if len(Y) > 0]

        if len(valid_X_train) > 0 and len(valid_X_cal) > 0:
            baseline.train_source_models(valid_X_train, valid_Y_train, task=task_type)
            baseline.calibrate(valid_X_cal, valid_Y_cal, alpha=alpha)
            baseline_ready = True
            print(f"  Baseline trained and calibrated with {len(valid_X_train)} sources")
        else:
            print("  No valid sources for baseline")
    except Exception as exc:
        print(f"  Error in baseline setup: {exc}")
        traceback.print_exc()

    evaluation_results: Dict[str, Any] = {}

    if baseline_ready:
        try:
            if task_type == 'classification':
                baseline_results = evaluate_baseline_classification_comprehensive(
                    baseline, X_test, Y_test, source_test, alpha
                )
            else:
                baseline_results = evaluate_baseline_regression_comprehensive(
                    baseline, X_test, Y_test, source_test, alpha
                )
            evaluation_results['baseline_comprehensive'] = baseline_results

            if 'Max_Aggregated' in baseline_results and 'Overall' in baseline_results['Max_Aggregated']:
                evaluation_results['Max_Aggregation'] = baseline_results['Max_Aggregated']['Overall']
            if 'Source_0' in baseline_results and 'Overall' in baseline_results['Source_0']:
                evaluation_results['Single_Source'] = baseline_results['Source_0']['Overall']

            print("  Baseline evaluation completed")
        except Exception as exc:
            print(f"  Error in baseline evaluation: {exc}")
            traceback.print_exc()

    gamma_results: List[Dict[str, Any]] = []
    lambda_snapshots: List[Dict[str, Any]] = []

    if mdcp_ready:
        # Fit pooled p_data(y|x) on the pooled *training* split only.
        if task_type == 'classification':
            pooled_p_data_model = PooledConditionalClassifier(X_train, Y_train, learner='gbm')
        else:
            pooled_p_data_model = PooledConditionalRegressionGaussian(X_train, Y_train, learner='gbm')
        p_data_conditional_train = pooled_p_data_model.conditional_at_pairs(X_train, Y_train)

        mimic_ratio = MIMIC_CALIBRATION_RATIO
        mimic_components: Optional[
            Tuple[
                np.ndarray,   # X_mimic_cal
                np.ndarray,   # X_mimic_test
                np.ndarray,   # Y_mimic_cal
                np.ndarray,   # Y_mimic_test
                np.ndarray,   # source_mimic_cal
                np.ndarray,   # source_mimic_test
            ]
        ] = None
        mimic_error: Optional[str] = None

        try:
            mimic_components = _split_mimic_sets(
                X_train,
                Y_train,
                source_train,
                mimic_ratio,
                trial_seed + 101,
                stratify=(task_type == 'classification'),
            )
        except Exception as exc:
            mimic_error = str(exc)
            print(f"  Unable to perform gamma tuning with mimic split: {exc}")
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

            if task_type == 'classification':
                y_grid_mimic = generate_y_grid_classification(
                    Y_sources_train + Y_sources_mimic_cal
                )
            else:
                y_grid_mimic = generate_y_grid_regression(
                    Y_sources_train + Y_sources_mimic_cal
                )
        else:
            X_sources_mimic_cal = []
            Y_sources_mimic_cal = []
            y_grid_mimic = None

        if task_type == 'classification':
            y_grid_true = generate_y_grid_classification(Y_sources_train + Y_sources_cal)
        else:
            y_grid_true = generate_y_grid_regression(Y_sources_train + Y_sources_cal)

        successful_gamma = False

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
                        mdcp_sources,
                        X_train,
                        Y_train,
                        X_sources_mimic_cal,  # type: ignore[arg-type]
                        Y_sources_mimic_cal,  # type: ignore[arg-type]
                        X_mimic_test,
                        Y_mimic_test,
                        y_grid_mimic,
                        alpha,
                        source_weights,
                        task_type,
                        source_mimic_test,
                        verbose=False,
                        p_data_conditional=p_data_conditional_train,
                    )
                    entry['mimic_metrics'] = mimic_metrics
                    entry['mimic_efficiency'] = _score_gamma_candidate(
                        mimic_metrics,
                        task_type,
                    )
                    entry['mimic_summary'] = _summarize_metrics_for_logging(
                        mimic_metrics,
                        task_type,
                    )
                except Exception as exc:
                    entry['mimic_error'] = str(exc)
                    print(f"  Gamma {gamma_value} mimic evaluation failed: {exc}")
                    traceback.print_exc()
            else:
                entry['mimic_error'] = mimic_error

            try:
                true_metrics, lambda_model = _run_mdcp_for_gamma(
                    gamma_value,
                    mdcp_sources,
                    X_train,
                    Y_train,
                    X_sources_cal,
                    Y_sources_cal,
                    X_test,
                    Y_test,
                    y_grid_true,
                    alpha,
                    source_weights,
                    task_type,
                    source_test,
                    verbose=False,
                    p_data_conditional=p_data_conditional_train,
                )
                entry['true_metrics'] = true_metrics
                entry['true_efficiency'] = _score_gamma_candidate(
                    true_metrics,
                    task_type,
                )
                entry['true_summary'] = _summarize_metrics_for_logging(
                    true_metrics,
                    task_type,
                )

                if lambda_model is not None:
                    sample_size = min(50, len(X_test))
                    if sample_size > 0:
                        sample_indices = np.random.choice(
                            len(X_test), size=sample_size, replace=False
                        )
                        sample_points = X_test[sample_indices]
                        sample_lambda_vals = lambda_model.lambda_at_x(sample_points)
                    else:
                        sample_indices = np.array([], dtype=int)
                        sample_points = np.empty((0, X_test.shape[1]))
                        sample_lambda_vals = np.empty((0, len(mdcp_sources)))

                    snapshot = {
                        'lambda_values': sample_lambda_vals,
                        'test_points': sample_points,
                        'sample_indices': sample_indices.astype(int),
                        'sample_size': int(sample_size),
                        'gamma1': gamma_value,
                        'gamma2': gamma_value,
                        'gamma3': 0.0,
                        'gamma_name': gamma_name,
                        'task': task_type,
                    }
                    lambda_snapshots.append(snapshot)

                successful_gamma = True
            except Exception as exc:
                entry['true_error'] = str(exc)
                print(f"  Gamma {gamma_value} true evaluation failed: {exc}")
                traceback.print_exc()

            gamma_results.append(entry)

        if not successful_gamma:
            print("  No successful MDCP evaluations recorded across gamma grid")

    mdcp_metrics_map = {
        entry['gamma_name']: entry['true_metrics']
        for entry in gamma_results
        if 'true_metrics' in entry
    }

    if mdcp_metrics_map:
        aggregated_metrics = aggregate_mdcp_metrics(mdcp_metrics_map, task_type)
        if aggregated_metrics:
            evaluation_results['MDCP'] = aggregated_metrics
        evaluation_results['MDCP_per_gamma'] = mdcp_metrics_map

    if gamma_results:
        results['gamma_results'] = gamma_results
        results['lambda_data'] = {
            'task': task_type,
            'per_gamma_results': np.array(gamma_results, dtype=object),
            'lambda_snapshots': np.array(lambda_snapshots, dtype=object),
        }

    results['evaluation_results'] = evaluation_results

    print(f"\n=== {task_type.title()} - {term_name.title()} Results Summary ===")
    for method_name, metrics in evaluation_results.items():
        if isinstance(metrics, dict) and 'coverage' in metrics:
            if task_type == 'classification':
                print(
                    f"  {method_name}: Coverage={metrics['coverage']:.3f}, "
                    f"Avg Set Size={metrics.get('avg_set_size', 0):.3f}"
                )
            else:
                print(
                    f"  {method_name}: Coverage={metrics['coverage']:.3f}, "
                    f"Avg Width={metrics.get('avg_width', 0):.3f}"
                )

    return results


def save_results(results: Dict[str, Any], term_name: str, task_type: str, trial_index: int) -> Path:
    trial_seed = results['metadata']['trial_seed']

    if task_type == 'classification':
        results_dir = ensure_project_dir(NONLINEAR_CLASSIFICATION_FOLDER)  # noqa: F405
    else:
        results_dir = ensure_project_dir(NONLINEAR_REGRESSION_FOLDER)  # noqa: F405
    lambda_dir = ensure_project_dir(NONLINEAR_LAMBDA_FOLDER)  # noqa: F405

    if task_type == 'classification':
        results_filename = (
            f"trial_{trial_index:02d}_{term_name}_seed_{trial_seed}_temperature_{TEMPERATURE}_"
            f"alpha_{alpha}_sources_{n_sources}_classes_{n_classes}.npz"
        )
        results_path = results_dir / results_filename
    else:
        results_filename = (
            f"trial_{trial_index:02d}_{term_name}_seed_{trial_seed}_temperature_{TEMPERATURE}_"
            f"alpha_{alpha}_sources_{n_sources}.npz"
        )
        results_path = results_dir / results_filename

    np.savez(results_path, **results)
    print(f"  Main results saved: {prefer_relative_path(results_path)}")  # noqa: F405

    if 'lambda_data' in results:
        lambda_data = results['lambda_data'].copy()
        lambda_data.update(results['metadata'])
        lambda_data['trial_index'] = trial_index

        if task_type == 'classification':
            lambda_filename = (
                f"lambda_trial_{trial_index:02d}_{term_name}_seed_{trial_seed}_temperature_{TEMPERATURE}_"
                f"alpha_{alpha}_sources_{n_sources}_classes_{n_classes}.npz"
            )
        else:
            lambda_filename = (
                f"lambda_trial_{trial_index:02d}_{term_name}_seed_{trial_seed}_temperature_{TEMPERATURE}_"
                f"alpha_{alpha}_sources_{n_sources}.npz"
            )

        lambda_path = lambda_dir / lambda_filename
        np.savez(lambda_path, **lambda_data)
        print(f"  Lambda data saved: {prefer_relative_path(lambda_path)}")  # noqa: F405
        return results_path

    return results_path


def main():
    args = parse_args()

    if not MDCP_AVAILABLE:
        print("Error: MDCP not available. Please check imports.")
        sys.exit(1)

    if args.seeds is not None and len(args.seeds) != args.num_trials:
        raise ValueError("The number of explicit seeds must match --num-trials.")

    print("=" * 80)
    print("INDIVIDUAL NONLINEAR TERM EVALUATION (MULTI-TRIAL)")
    print("=" * 80)
    print(f"Testing {len(CLASSIFICATION_TERMS)} classification configurations")
    print(f"Testing {len(REGRESSION_TERMS)} regression configurations")
    print(f"Trials: {args.num_trials}")

    base_seed = args.base_seed
    rng = np.random.default_rng()

    trial_summaries: List[Dict[str, Any]] = []

    for trial_idx in range(args.num_trials):
        if args.seeds is not None:
            trial_seed = args.seeds[trial_idx]
        else:
            offset = int(rng.integers(1, 10000))
            base_seed += offset
            trial_seed = base_seed

        print(f"\n{'#' * 80}")
        print(f"TRIAL {trial_idx + 1}/{args.num_trials} | Seed = {trial_seed}")
        print(f"{'#' * 80}")

        trial_record: Dict[str, Any] = {
            'trial_index': trial_idx + 1,
            'trial_seed': trial_seed,
            'classification': {},
            'regression': {},
        }

        print(f"\n{'=' * 80}")
        print("CLASSIFICATION NONLINEAR TERMS")
        print(f"{'=' * 80}")

        for term_name, term_config in CLASSIFICATION_TERMS.items():
            try:
                results = run_single_evaluation(
                    term_name,
                    term_config,
                    'classification',
                    trial_seed,
                    trial_idx + 1,
                )
                save_results(results, term_name, 'classification', trial_idx + 1)
                trial_record['classification'][term_name] = results
                print(f"-->  {term_name.upper()} classification completed successfully")
            except Exception as exc:
                print(f"-->  {term_name.upper()} classification failed: {exc}")
                traceback.print_exc()

        print(f"\n{'=' * 80}")
        print("REGRESSION NONLINEAR TERMS")
        print(f"{'=' * 80}")

        for term_name, term_config in REGRESSION_TERMS.items():
            try:
                results = run_single_evaluation(
                    term_name,
                    term_config,
                    'regression',
                    trial_seed,
                    trial_idx + 1,
                )
                save_results(results, term_name, 'regression', trial_idx + 1)
                trial_record['regression'][term_name] = results
                print(f"--> {term_name.upper()} regression completed successfully")
            except Exception as exc:
                print(f"(X) {term_name.upper()} regression failed: {exc}")
                traceback.print_exc()

        trial_summaries.append(trial_record)

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY ACROSS TRIALS")
    print(f"{'=' * 80}")

    for record in trial_summaries:
        print(f"\nTRIAL {record['trial_index']} (seed={record['trial_seed']}):")
        print("  Classification:")
        for term_name, results in record['classification'].items():
            eval_results = results.get('evaluation_results', {})
            mdcp_metrics = eval_results.get('MDCP')
            if isinstance(mdcp_metrics, dict) and 'coverage' in mdcp_metrics:
                print(
                    f"    {term_name.upper()} - MDCP: Coverage={mdcp_metrics['coverage']:.3f}, "
                    f"Set Size={mdcp_metrics.get('avg_set_size', 0):.3f}"
                )
        print("  Regression:")
        for term_name, results in record['regression'].items():
            eval_results = results.get('evaluation_results', {})
            mdcp_metrics = eval_results.get('MDCP')
            if isinstance(mdcp_metrics, dict) and 'coverage' in mdcp_metrics:
                print(
                    f"    {term_name.upper()} - MDCP: Coverage={mdcp_metrics['coverage']:.3f}, "
                    f"Width={mdcp_metrics.get('avg_width', 0):.3f}"
                )

    print(f"\n{'=' * 80}")
    print("INDIVIDUAL NONLINEAR TERM EVALUATION COMPLETE!")
    print(f"{'=' * 80}")

    return trial_summaries


if __name__ == "__main__":
    main()