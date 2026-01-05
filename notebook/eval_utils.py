import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from model.MDCP import (
        fit_lambda_from_sources,
        aggregated_conformal_set_multi,
        precompute_calibration_cache,
    )

def _slice_individual_info(individual_info, indices):
    """Return a copy of individual_info containing only entries at the specified indices."""
    if individual_info is None or 'individual_sets_all' not in individual_info:
        return individual_info

    indices = list(indices)
    sliced = {k: v for k, v in individual_info.items() if k != 'individual_sets_all'}
    sliced_sets = []

    for source_sets in individual_info['individual_sets_all']:
        sliced_sets.append([source_sets[idx] for idx in indices])

    sliced['individual_sets_all'] = sliced_sets
    return sliced


def _subset_metrics_from_overall(overall_metrics, indices, value_key, avg_key):
    """Construct subset metrics by slicing precomputed per-point arrays."""
    subset_indices = np.asarray(indices, dtype=int)
    if subset_indices.size == 0:
        empty_bool = np.zeros(0, dtype=bool)
        empty_float = np.zeros(0, dtype=float)
        return {
            'coverage': np.nan,
            avg_key: np.nan,         # 'avg_set_size' or 'avg_width'
            'covered': empty_bool,
            value_key: empty_float,  # 'set_sizes' or 'widths'
        }

    covered_subset = overall_metrics['covered'][subset_indices]
    values_subset = overall_metrics[value_key][subset_indices]

    return {
        'coverage': float(np.mean(covered_subset)),
        avg_key: float(np.mean(values_subset)),
        'covered': covered_subset,
        value_key: values_subset,
    }


def _interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, Any]:
    """Compute coverage and width statistics for continuous intervals."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper interval bounds must share the same shape.")
    if y_true.shape[0] != lower.shape[0]:
        raise ValueError("Interval predictions must align with y_true length.")

    inside = (y_true >= lower) & (y_true <= upper)
    widths = np.maximum(0.0, upper - lower)
    coverage = float(np.mean(inside)) if inside.size else np.nan
    avg_width = float(np.mean(widths)) if widths.size else np.nan
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'covered': inside,
        'widths': widths,
    }


def _union_hull(lower_list: List[np.ndarray], upper_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the convex hull covering all provided intervals."""
    if len(lower_list) != len(upper_list):
        raise ValueError("Union hull requires equal numbers of lower and upper arrays.")
    if not lower_list:
        raise ValueError("Union hull requires at least one interval.")
    lower_stack = np.vstack([np.asarray(lo, dtype=float) for lo in lower_list])
    upper_stack = np.vstack([np.asarray(hi, dtype=float) for hi in upper_list])
    return np.min(lower_stack, axis=0), np.max(upper_stack, axis=0)


# --------------------------
#   Baseline evaluation functions
# --------------------------
def evaluate_classification_performance(prediction_sets, y_true, individual_info=None):
    """Evaluate coverage and set size for classification.
    Uses correct union coverage by checking individual sets.
    """
    n_test = len(y_true)
    union_covered = np.zeros(n_test, dtype=bool)
    set_sizes = np.zeros(n_test)

    for i in range(n_test):
        # Count size of prediction set
        set_sizes[i] = np.sum(prediction_sets[i])
        
        # Calculate true union coverage if individual_info is provided
        if individual_info is not None and 'individual_sets_all' in individual_info:
            union_covered[i] = False
            # Check if y_true[i] is in ANY of the individual sets
            for j in range(len(individual_info['individual_sets_all'])):
                individual_set_j = individual_info['individual_sets_all'][j][i]
                if y_true[i] in individual_set_j:
                    union_covered[i] = True
                    break
        else:
            # for Single source baseline
            # Fallback: check if true class is in prediction set
            if y_true[i] < prediction_sets.shape[1]:
                union_covered[i] = prediction_sets[i, y_true[i]] == 1
            else:
                # If true class is outside prediction set bounds, it's not covered
                union_covered[i] = False
    
    coverage = np.mean(union_covered)
    avg_set_size = np.mean(set_sizes)
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'covered': union_covered,
        'set_sizes': set_sizes
    }


def evaluate_classification_performance_with_individual_sets(prediction_sets, y_true, individual_info=None, source_test=None):
    """Enhanced evaluation for classification with source-specific individual set analysis.
    
    For baseline methods:
    - Evaluates (K+1)^2 matrix: K+1 construction methods x K+1 evaluation groups
    - Construction methods: K individual sources + 1 max-p aggregated
    - Evaluation groups: K individual source test sets + 1 overall test set
    
    For MDCP methods:
    - Evaluates K+1 performance: max-p set on K source test sets + overall
    
    Args:
        prediction_sets: Prediction sets (max-p aggregated for MDCP, varies for baseline)
        y_true: True labels
        individual_info: Dict with 'individual_sets_all' for individual source sets
        source_test: Array indicating source of each test point (required for new metrics)
    """
    basic_metrics = evaluate_classification_performance(prediction_sets, y_true, individual_info)
    
    if source_test is None:
        # Return basic metrics if no source information
        return basic_metrics
    
    n_test = len(y_true)
    source_test = np.asarray(source_test)
    unique_sources = np.unique(source_test)
    n_sources = len(unique_sources)
    
    # Individual_coverage: coverage rate for test points FROM each source j
    individual_coverage = np.zeros(n_sources)
    
    # Individual_widths: avg set size for test points FROM each source j  
    individual_widths = np.zeros(n_sources)
    
    for j, source_id in enumerate(unique_sources):
        # Get test points from this source
        source_mask = (source_test == source_id)
        source_indices = np.where(source_mask)[0]
        
        if len(source_indices) > 0:
            # Coverage: how many test points from source j are covered by the (max-p) set
            covered_from_source = basic_metrics['covered'][source_indices]
            individual_coverage[j] = np.mean(covered_from_source)
            
            # Width: average set size for test points from source j
            set_sizes_from_source = basic_metrics['set_sizes'][source_indices]
            individual_widths[j] = np.mean(set_sizes_from_source)
        else:
            individual_coverage[j] = 0
            individual_widths[j] = 0

    enhanced_metrics = basic_metrics.copy()
    enhanced_metrics.update({
        'individual_coverage': individual_coverage,
        'individual_widths': individual_widths,
        'n_sources': n_sources,
        'unique_sources': unique_sources
    })
    
    return enhanced_metrics


def evaluate_regression_performance(direct_union, y_true, individual_info=None):
    """Evaluate coverage and interval width for regression."""
    n_test = len(y_true)
    union_covered = np.zeros(n_test, dtype=bool)
    widths = np.zeros(n_test)
    
    for i in range(n_test):
        if np.isnan(direct_union['lower'][i]) or np.isnan(direct_union['upper'][i]):
            # handle NaN intervals
            widths[i] = np.nan
        else:
            # Calculate max-p aggregated set width using segment validity checking
            if individual_info is not None and 'individual_sets_all' in individual_info:
                # Collect all endpoints from individual sets
                all_endpoints = set()
                
                for j in range(len(individual_info['individual_sets_all'])):
                    individual_set_j = individual_info['individual_sets_all'][j][i]
                    if len(individual_set_j) > 0:
                        all_endpoints.add(np.min(individual_set_j))
                        all_endpoints.add(np.max(individual_set_j))
                
                if len(all_endpoints) > 0:
                    # Sort endpoints and create segments
                    sorted_endpoints = np.array(sorted(all_endpoints))
                    
                    if len(sorted_endpoints) == 1:
                        widths[i] = 0.0
                    else:
                        total_width = 0.0
                        
                        for k in range(len(sorted_endpoints) - 1):
                            segment_start = sorted_endpoints[k]
                            segment_end = sorted_endpoints[k + 1]
                            
                            sample_point = (segment_start + segment_end) / 2.0
                            
                            # Check if sample point is valid (contained in max-p aggregated set)
                            # A point is valid if it's contained in at least one individual set
                            is_valid = False
                            for j in range(len(individual_info['individual_sets_all'])):
                                individual_set_j = individual_info['individual_sets_all'][j][i]
                                if len(individual_set_j) > 0:
                                    min_val = np.min(individual_set_j)
                                    max_val = np.max(individual_set_j)
                                    if min_val <= sample_point <= max_val:
                                        is_valid = True
                                        break
                            
                            # If segment is valid, add its width
                            if is_valid:
                                total_width += (segment_end - segment_start)
                        
                        widths[i] = total_width
                else:
                    # No individual sets available, use interval width as fallback
                    widths[i] = direct_union['upper'][i] - direct_union['lower'][i]
            else:
                # Single source case
                # Fallback when no individual_info available
                widths[i] = direct_union['upper'][i] - direct_union['lower'][i]
        
        # Calculate coverage using min/max interval check for each individual set
        if individual_info is not None and 'individual_sets_all' in individual_info:
            union_covered[i] = False
            # Check if y_true[i] is in ANY of the individual sets using min/max interval
            for j in range(len(individual_info['individual_sets_all'])):
                individual_set_j = individual_info['individual_sets_all'][j][i]
                if len(individual_set_j) > 0:
                    min_val = np.min(individual_set_j)
                    max_val = np.max(individual_set_j)
                    # Simple min/max interval check
                    if min_val <= y_true[i] <= max_val:
                        union_covered[i] = True
                        break
        else:
            # for Single source baseline
            # Fallback: check if true value is in prediction interval
            # NaN intervals are considered as not covered
            if np.isnan(direct_union['lower'][i]) or np.isnan(direct_union['upper'][i]):
                union_covered[i] = False
            else:
                union_covered[i] = (direct_union['lower'][i] <= y_true[i] <= 
                                   direct_union['upper'][i])
    
    coverage = np.mean(union_covered)
    avg_width = np.mean(np.where(np.isnan(widths), 0, widths))
    
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'covered': union_covered,
        'widths': widths
    }


def evaluate_regression_performance_with_individual_sets(direct_union, y_true, individual_info=None, source_test=None):
    """Enhanced evaluation for regression with source-specific individual set analysis.
    
    For baseline methods:
    - Evaluates (K+1)^2 matrix: K+1 construction methods x K+1 evaluation groups
    - Construction methods: K individual sources + 1 max-p aggregated
    - Evaluation groups: K individual source test sets + 1 overall test set
    
    For MDCP methods:
    - Evaluates K+1 performance: max-p set on K source test sets + overall
    
    Args:
        direct_union: Regression intervals (max-p aggregated for MDCP, varies for baseline)
        y_true: True targets
        individual_info: Dict with 'individual_sets_all' for individual source sets
        source_test: Array indicating source of each test point (required for new metrics)
    """
    basic_metrics = evaluate_regression_performance(direct_union, y_true, individual_info)
    
    if source_test is None:
        # Return basic metrics if no source information
        return basic_metrics
    
    n_test = len(y_true)
    source_test = np.asarray(source_test)
    unique_sources = np.unique(source_test)
    n_sources = len(unique_sources)
    
    # Individual_coverage: coverage rate for test points FROM each source j
    individual_coverage = np.zeros(n_sources)
    
    # Individual_widths: avg interval width for test points FROM each source j
    individual_widths = np.zeros(n_sources)
    
    for j, source_id in enumerate(unique_sources):
        # Get test points from this source
        source_mask = (source_test == source_id)
        source_indices = np.where(source_mask)[0]
        
        if len(source_indices) > 0:
            # Coverage: how many test points from source j are covered by the (max-p) set
            covered_from_source = basic_metrics['covered'][source_indices]
            individual_coverage[j] = np.mean(covered_from_source)
            
            # Width: average interval width for test points from source j
            widths_from_source = basic_metrics['widths'][source_indices]
            individual_widths[j] = np.mean(np.where(np.isnan(widths_from_source), 0, widths_from_source))
        else:
            individual_coverage[j] = 0
            individual_widths[j] = 0
    
    enhanced_metrics = basic_metrics.copy()
    enhanced_metrics.update({
        'individual_coverage': individual_coverage,
        'individual_widths': individual_widths,
        'n_sources': n_sources,
        'unique_sources': unique_sources
    })
    
    return enhanced_metrics


# --------------------------
#   Comprehensive baseline evaluation functions
# --------------------------
def evaluate_baseline_classification_comprehensive(baseline_predictor, X_test, y_true, source_test, alpha):
    """
    Evaluate baseline classification with (K+1)^2 comprehensive analysis.
    
    Constructs prediction sets using:
    - K individual sources (source-specific sets)
    - 1 max-p aggregated set
    
    Evaluates each construction method on:
    - K individual source test subsets
    - 1 overall test set
    
    Returns (K+1) x (K+1) results matrix.
    """
    source_test = np.asarray(source_test)
    unique_sources = np.unique(source_test)
    K = len(unique_sources)
    n_test = len(y_true)
    
    results = {}
    
    # Construction methods
    construction_methods = {}
    
    # Individual source predictions
    for k, source_id in enumerate(unique_sources):
        try:
            pred_sets = baseline_predictor.predict_set_single_source(X_test, source_id=int(source_id))
            construction_methods[f'Source_{source_id}'] = {'prediction_sets': pred_sets, 'individual_info': None}
        except Exception as e:
            raise ValueError(f"Warning: Could not get predictions from source {source_id}: {e}")
    
    # Max-p aggregated prediction
    try:
        pred_sets_max, individual_info = baseline_predictor.predict_set_max_aggregated(X_test)
        construction_methods['Max_Aggregated'] = {'prediction_sets': pred_sets_max, 'individual_info': {'individual_sets_all': individual_info}}
    except Exception as e:
        raise ValueError(f"Warning: Could not get max-aggregated predictions: {e}")
    
    source_indices_map = {
        source_id: np.where(source_test == source_id)[0] for source_id in unique_sources
    }

    # Evaluate each construction method on each test subset (K+1 as outer loop)
    for construct_name, construct_data in construction_methods.items():
        results[construct_name] = {}

        # Overall evaluation (1 per loop)
        overall_metrics = evaluate_classification_performance_with_individual_sets(
            construct_data['prediction_sets'], y_true, construct_data['individual_info'], source_test
        )
        results[construct_name]['Overall'] = overall_metrics

        # Source-specific evaluations (K per loop) reusing overall metrics
        for source_id, source_indices in source_indices_map.items():
            if source_indices.size == 0:
                continue

            subset_metrics = _subset_metrics_from_overall(
                overall_metrics,
                source_indices,
                value_key='set_sizes',
                avg_key='avg_set_size'
            )
            subset_info = _slice_individual_info(construct_data['individual_info'], source_indices)
            if subset_info is not None and 'individual_sets_all' in subset_info:
                subset_metrics['individual_sets_all'] = subset_info['individual_sets_all']

            results[construct_name][f'Source_{source_id}'] = subset_metrics

    return results


def evaluate_baseline_regression_comprehensive(baseline_predictor, X_test, y_true, source_test, alpha):
    """
    Evaluate baseline regression with (K+1)^2 comprehensive analysis.
    
    Constructs prediction intervals using:
    - K individual sources (source-specific intervals)
    - 1 max-p aggregated interval
    
    Evaluates each construction method on:
    - K individual source test subsets
    - 1 overall test set
    
    Returns (K+1) x (K+1) results matrix.
    """
    source_test = np.asarray(source_test)
    unique_sources = np.unique(source_test)
    K = len(unique_sources)
    n_test = len(y_true)
    
    results = {}
    
    # Construction methods
    construction_methods = {}
    
    # Individual source predictions
    for k, source_id in enumerate(unique_sources):
        try:
            pred_intervals = baseline_predictor.predict_set_single_source(X_test, source_id=int(source_id))
            construction_methods[f'Source_{source_id}'] = {'prediction_intervals': pred_intervals, 'individual_info': None}
        except Exception as e:
            raise ValueError(f"Warning: Could not get predictions from source {source_id}: {e}")
    
    # Max-p aggregated prediction
    try:
        pred_intervals_max, individual_info = baseline_predictor.predict_set_max_aggregated(X_test)
        construction_methods['Max_Aggregated'] = {'prediction_intervals': pred_intervals_max, 'individual_info': {'individual_sets_all': individual_info}}
    except Exception as e:
        raise ValueError(f"Warning: Could not get max-aggregated predictions: {e}")
    
    source_indices_map = {
        source_id: np.where(source_test == source_id)[0] for source_id in unique_sources
    }

    # Evaluate each construction method on each test subset (K+1 as outer loop)
    for construct_name, construct_data in construction_methods.items():
        results[construct_name] = {}

        # Overall evaluation (1 per loop)
        overall_metrics = evaluate_regression_performance_with_individual_sets(
            construct_data['prediction_intervals'], y_true, construct_data['individual_info'], source_test
        )
        results[construct_name]['Overall'] = overall_metrics

        # Source-specific evaluations (K per loop) reusing overall metrics
        for source_id, source_indices in source_indices_map.items():
            if source_indices.size == 0:
                continue

            subset_metrics = _subset_metrics_from_overall(
                overall_metrics,
                source_indices,
                value_key='widths',
                avg_key='avg_width'
            )
            subset_info = _slice_individual_info(construct_data['individual_info'], source_indices)
            if subset_info is not None and 'individual_sets_all' in subset_info:
                subset_metrics['individual_sets_all'] = subset_info['individual_sets_all']

            results[construct_name][f'Source_{source_id}'] = subset_metrics

    # Optional CQR baseline evaluation (per-source + union aggregation)
    if hasattr(baseline_predictor, 'has_cqr') and callable(getattr(baseline_predictor, 'has_cqr')):
        if baseline_predictor.has_cqr():
            per_source_intervals = baseline_predictor.cqr_predict_intervals_per_source(X_test)
            if per_source_intervals:
                valid_entries: List[Tuple[int, np.ndarray, np.ndarray]] = []
                lower_list: List[np.ndarray] = []
                upper_list: List[np.ndarray] = []
                for idx, intervals in enumerate(per_source_intervals):
                    if intervals is None:
                        continue
                    lo, hi = intervals
                    lo = np.asarray(lo, dtype=float)
                    hi = np.asarray(hi, dtype=float)
                    if lo.shape[0] != n_test or hi.shape[0] != n_test:
                        continue
                    valid_entries.append((idx, lo, hi))
                    lower_list.append(lo)
                    upper_list.append(hi)

                ordered_sources = sorted(source_indices_map.keys())

                for source_idx, lo, hi in valid_entries:
                    metrics_overall = _interval_metrics(y_true, lo, hi)
                    per_source_cov: List[float] = []
                    per_source_width: List[float] = []
                    entry = {'Overall': metrics_overall}
                    for target_source in ordered_sources:
                        target_indices = source_indices_map[target_source]
                        subset_metrics = _subset_metrics_from_overall(
                            metrics_overall,
                            target_indices,
                            value_key='widths',
                            avg_key='avg_width',
                        )
                        entry[f'Source_{target_source}'] = subset_metrics
                        per_source_cov.append(float(subset_metrics['coverage']))
                        per_source_width.append(float(subset_metrics['avg_width']))

                    metrics_overall['unique_sources'] = np.asarray(ordered_sources, dtype=int)
                    metrics_overall['individual_coverage'] = np.asarray(per_source_cov, dtype=float)
                    metrics_overall['individual_widths'] = np.asarray(per_source_width, dtype=float)
                    results[f'CQR_Source_{source_idx}'] = entry

                if lower_list and upper_list:
                    agg_lower, agg_upper = _union_hull(lower_list, upper_list)
                    agg_metrics = _interval_metrics(y_true, agg_lower, agg_upper)
                    agg_entry = {'Overall': agg_metrics}
                    per_source_cov: List[float] = []
                    per_source_width: List[float] = []
                    for target_source in ordered_sources:
                        target_indices = source_indices_map[target_source]
                        subset_metrics = _subset_metrics_from_overall(
                            agg_metrics,
                            target_indices,
                            value_key='widths',
                            avg_key='avg_width',
                        )
                        agg_entry[f'Source_{target_source}'] = subset_metrics
                        per_source_cov.append(float(subset_metrics['coverage']))
                        per_source_width.append(float(subset_metrics['avg_width']))

                    agg_metrics['unique_sources'] = np.asarray(ordered_sources, dtype=int)
                    agg_metrics['individual_coverage'] = np.asarray(per_source_cov, dtype=float)
                    agg_metrics['individual_widths'] = np.asarray(per_source_width, dtype=float)
                    results['CQR_Max_Aggregated'] = agg_entry
    
    return results


# --------------------------
#   MDCP-specific evaluation functions
# --------------------------
def calculate_discontinuous_width(y_grid, union_mask, extend_points=1):
    """Calculate width for discontinuous sets by connecting nearby grid points.
    
    For MDCP, the prediction set can be discontinuous. We calculate width by:
    1. Finding connected components of included grid points
    2. For each component, extending by 'extend_points' on both sides
    3. Summing the widths of all extended components
    
    Args:
        y_grid: array of candidate y values
        union_mask: boolean mask indicating which grid points are included
        extend_points: number of grid points to extend on each side of components
    
    Returns:
        total_width: sum of widths of all connected components
    """
    if not np.any(union_mask):
        return 0.0
    
    included_indices = np.where(union_mask)[0]
    if len(included_indices) == 0:
        return 0.0
    
    # Find connected components (consecutive indices)
    components = []
    current_component = [included_indices[0]]
    
    for i in range(1, len(included_indices)):
        if included_indices[i] == included_indices[i-1] + 1:
            # Consecutive, add to current component
            current_component.append(included_indices[i])
        else:
            # Gap found, start new component
            components.append(current_component)
            current_component = [included_indices[i]]
    
    # Don't forget the last component
    components.append(current_component)
    
    total_width = 0.0
    n_grid = len(y_grid)
    
    for component in components:
        # Extend component by extend_points on both sides
        start_idx = max(0, min(component) - extend_points)
        end_idx = min(n_grid - 1, max(component) + extend_points)
        
        # Calculate width of this extended component
        component_width = y_grid[end_idx] - y_grid[start_idx]
        total_width += component_width
    
    return total_width


def evaluate_mdcp_classification_performance(mdcp_results, y_true, y_grid, alpha, source_test=None,
                                             collect_individual_sets=False):
    """Evaluate MDCP classification performance using max-p rule.
    
    Args:
        mdcp_results: list of dicts, one per test point, each containing:
            - 'p_values_y_grid': (K, m) p-values per source per y
            - 'union_mask_y_grid': (m,) boolean mask
            - 'union_mask_true_y': scalar boolean, whether true y is covered
        y_true: (n_test,) true labels
        y_grid: (m,) grid of candidate labels
        alpha: significance level
        source_test: Array indicating source of each test point (optional)
        collect_individual_sets: Whether to build per-source individual sets (default True).
            Disabling this avoids allocating large intermediate arrays when only summary
            metrics are required.
    
    Returns:
        dict with coverage and efficiency metrics, including source-specific analysis if source_test provided
    """
    n_test = len(mdcp_results)
    if n_test == 0:
        raise ValueError("mdcp_results must contain at least one entry")

    y_true = np.asarray(y_true)
    y_grid = np.asarray(y_grid)

    union_masks = np.stack([result['union_mask_y_grid'] for result in mdcp_results], axis=0)
    union_covered = np.fromiter(
        (result['union_mask_true_y'] for result in mdcp_results), dtype=bool, count=n_test
    )
    set_sizes = union_masks.sum(axis=1).astype(float)

    n_sources = mdcp_results[0]['p_values_y_grid'].shape[0]

    individual_sets_all = None
    if collect_individual_sets:
        p_values_stack = np.stack([result['p_values_y_grid'] for result in mdcp_results], axis=0)
        mask_stack = p_values_stack > alpha
        individual_sets_all = []
        for j in range(n_sources):
            masks_for_source = mask_stack[:, j, :]
            source_sets = [
                y_grid[row_mask] if np.any(row_mask) else np.array([], dtype=y_grid.dtype)
                for row_mask in masks_for_source
            ]
            individual_sets_all.append(source_sets)
    
    coverage = np.mean(union_covered)
    avg_set_size = np.mean(set_sizes)
    
    basic_metrics = {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'covered': union_covered,
        'set_sizes': set_sizes,
        'individual_sets_all': individual_sets_all,
        'n_sources': n_sources
    }
    
    # Add source-specific analysis if source_test is provided
    if source_test is not None:
        source_test = np.asarray(source_test)
        unique_sources, inverse = np.unique(source_test, return_inverse=True)
        counts = np.bincount(inverse)
        coverage_sum = np.bincount(inverse, weights=union_covered.astype(float))
        size_sum = np.bincount(inverse, weights=set_sizes)

        individual_coverage = np.divide(
            coverage_sum,
            counts,
            out=np.zeros_like(coverage_sum),
            where=counts > 0
        )
        individual_widths = np.divide(
            size_sum,
            counts,
            out=np.zeros_like(size_sum),
            where=counts > 0
        )

        basic_metrics.update({
            'individual_coverage': individual_coverage,
            'individual_widths': individual_widths,
            'unique_sources': unique_sources
        })
    
    return basic_metrics


def evaluate_mdcp_regression_performance(mdcp_results, y_true, y_grid, alpha, extend_points=1,
                                         source_test=None, collect_individual_sets=False):
    """Evaluate MDCP regression performance using max-p rule and discontinuous width.
    
    Args:
        mdcp_results: list of dicts, one per test point, each containing:
            - 'p_values_y_grid': (K, m) p-values per source per y
            - 'union_mask_y_grid': (m,) boolean mask
            - 'union_mask_true_y': scalar boolean, whether true y is covered
        y_true: (n_test,) true targets
        y_grid: (m,) grid of candidate y values
        alpha: significance level
        extend_points: number of grid points to extend connected components
        source_test: Array indicating source of each test point (optional)
        collect_individual_sets: Whether to build per-source individual sets (default True).
            Disabling this avoids allocating large intermediate arrays when only summary
            metrics are required.
    
    Returns:
        dict with coverage and efficiency metrics, including source-specific analysis if source_test provided
    """
    n_test = len(mdcp_results)
    if n_test == 0:
        raise ValueError("mdcp_results must contain at least one entry")

    y_true = np.asarray(y_true)
    y_grid = np.asarray(y_grid)

    union_masks = np.stack([result['union_mask_y_grid'] for result in mdcp_results], axis=0)
    union_covered = np.fromiter(
        (result['union_mask_true_y'] for result in mdcp_results), dtype=bool, count=n_test
    )
    widths = np.array([
        calculate_discontinuous_width(y_grid, union_mask, extend_points)
        for union_mask in union_masks
    ], dtype=float)

    n_sources = mdcp_results[0]['p_values_y_grid'].shape[0]

    individual_sets_all = None
    if collect_individual_sets:
        p_values_stack = np.stack([result['p_values_y_grid'] for result in mdcp_results], axis=0)
        mask_stack = p_values_stack > alpha
        individual_sets_all = []
        for j in range(n_sources):
            masks_for_source = mask_stack[:, j, :]
            source_sets = [
                y_grid[row_mask] if np.any(row_mask) else np.array([], dtype=y_grid.dtype)
                for row_mask in masks_for_source
            ]
            individual_sets_all.append(source_sets)
    
    coverage = np.mean(union_covered)
    avg_width = np.mean(widths)
    
    basic_metrics = {
        'coverage': coverage,
        'avg_width': avg_width,
        'covered': union_covered,
        'widths': widths,
        'individual_sets_all': individual_sets_all,
        'n_sources': n_sources
    }
    
    # Add source-specific analysis if source_test is provided
    if source_test is not None:
        source_test = np.asarray(source_test)
        unique_sources, inverse = np.unique(source_test, return_inverse=True)
        counts = np.bincount(inverse)
        coverage_sum = np.bincount(inverse, weights=union_covered.astype(float))
        width_sum = np.bincount(inverse, weights=widths)

        individual_coverage = np.divide(
            coverage_sum,
            counts,
            out=np.zeros_like(coverage_sum),
            where=counts > 0
        )
        individual_widths = np.divide(
            width_sum,
            counts,
            out=np.zeros_like(width_sum),
            where=counts > 0
        )

        basic_metrics.update({
            'individual_coverage': individual_coverage,
            'individual_widths': individual_widths,
            'unique_sources': unique_sources
        })
    
    return basic_metrics


# --------------------------
#   Helper functions for generating y_grid for MDCP
# --------------------------
def generate_y_grid_classification(Y_sources_list, n_classes=None):
    """Generate y_grid for classification tasks."""
    if n_classes is None:
        # Determine number of classes from all sources
        all_classes = set()
        for Y_source in Y_sources_list:
            if len(Y_source) > 0:
                all_classes.update(np.unique(Y_source))
        n_classes = len(all_classes)
        y_grid = np.array(sorted(all_classes))
    else:
        y_grid = np.arange(n_classes)
    
    return y_grid

def generate_y_grid_regression(Y_sources_list, n_grid_points=100, margin_factor=0.0):
    """Generate y_grid for regression tasks."""
    # Find global min and max across all sources
    all_values = []
    for Y_source in Y_sources_list:
        if len(Y_source) > 0:
            all_values.extend(Y_source)
    
    if len(all_values) == 0:
        return np.linspace(-1, 1, n_grid_points)
    
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    
    # Add margin to cover potential prediction ranges
    y_min_extended = y_min - margin_factor * y_range
    y_max_extended = y_max + margin_factor * y_range
    
    return np.linspace(y_min_extended, y_max_extended, n_grid_points)


# --------------------------
#   Helper functions for extracting individual sets
# --------------------------
def extract_individual_sets_from_pvalues(p_values_per_y, y_grid, alpha):
    """Extract individual conformal sets from per-source p-values."""
    K, m = p_values_per_y.shape
    individual_sets = []
    
    for k in range(K):
        # Individual set for source k: all y where p_k(y) > alpha
        source_mask = p_values_per_y[k, :] > alpha
        if np.any(source_mask):
            individual_sets.append(y_grid[source_mask])
        else:
            individual_sets.append(np.array([]))
    
    return individual_sets


# --------------------------
#   Helper functions for model compatibility
# --------------------------
def add_joint_pdf_to_source(source_model):
    """Add joint_pdf method to source model if it doesn't exist."""
    if not hasattr(source_model, 'joint_pdf'):
        if hasattr(source_model, 'joint_prob'):
            source_model.joint_pdf = source_model.joint_prob
        else:
            raise AttributeError(f"Source model {type(source_model)} doesn't have joint_prob or joint_pdf method")
        if hasattr(source_model, 'joint_prob_at_pairs'):
            source_model.joint_pdf_at_pairs = source_model.joint_prob_at_pairs
        else:
            raise AttributeError(f"Source model {type(source_model)} doesn't have joint_prob_at_pairs or joint_pdf_at_pairs method")
    return source_model




# ---------------------------
#  Parameter tuning and utility functions
# ---------------------------
def _format_gamma_name(value: float) -> str:
    value_str = format(value, 'g')
    return f'g1_{value_str}_g2_{value_str}_g3_0.0'


def _split_mimic_sets(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    source_train: np.ndarray,
    mimic_cal_ratio: float,
    seed: int,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mimic_cal_ratio = float(np.clip(mimic_cal_ratio, 1e-3, 1 - 1e-3))
    test_size = float(np.clip(1.0 - mimic_cal_ratio, 1e-3, 1 - 1e-3))
    stratify_labels = None
    if stratify and len(np.unique(Y_train)) > 1:
        stratify_labels = Y_train

    try:
        splitting = train_test_split(
            X_train,
            Y_train,
            source_train,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        splitting = train_test_split(
            X_train,
            Y_train,
            source_train,
            test_size=test_size,
            random_state=seed,
        )

    return tuple(splitting)  # type: ignore[return-value]


def _score_gamma_candidate(metrics: Dict[str, Any], task_type: str) -> float:
    """Return efficiency metric (set size or width) used for gamma comparison."""
    size_key = 'avg_set_size' if task_type == 'classification' else 'avg_width'
    return float(metrics.get(size_key, np.inf))


def _summarize_metrics_for_logging(metrics: Dict[str, Any], task_type: str) -> Dict[str, float]:
    summary: Dict[str, float] = {'coverage': float(metrics.get('coverage', np.nan))}
    if task_type == 'classification':
        summary['avg_set_size'] = float(metrics.get('avg_set_size', np.nan))
    else:
        summary['avg_width'] = float(metrics.get('avg_width', np.nan))
    return summary


def _run_mdcp_for_gamma(
    gamma_value: float,
    sources: List[Any],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_cal_list: List[np.ndarray],
    Y_cal_list: List[np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_grid: np.ndarray,
    alpha_val: float,
    source_weights: np.ndarray,
    task_type: str,
    source_test: Optional[np.ndarray],
    verbose: bool = False,
    lambda_mode: str = 'spline',
    p_data_model: Any = None,
    p_data_conditional: Any = None,
    nn_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    lambda_mode = lambda_mode or 'spline'
    spline_kwargs = {
        'gamma1': gamma_value,
        'gamma2': gamma_value,
        'gamma3': 0.0,
        'n_splines': 4,
        'degree': 2,
    }

    solver_kwargs = None
    if lambda_mode == 'nn':
        # Use gamma grid values as the LambdaNN L2 penalty multiplier (gamma1).
        solver_kwargs = {'gamma1': float(gamma_value)}

    lambda_model = fit_lambda_from_sources(
        sources,
        lambda_mode,
        X_train,
        Y_train,
        alpha=alpha_val,
        spline_kwargs=spline_kwargs if lambda_mode == 'spline' else None,
        nn_kwargs=nn_kwargs if lambda_mode == 'nn' else None,
        solver_kwargs=solver_kwargs,
        verbose=verbose,
        source_weights=source_weights,
        p_data_model=p_data_model,
        p_data_conditional=p_data_conditional,
    )

    sources_with_pdf = [add_joint_pdf_to_source(src) for src in sources]

    calibration_cache = precompute_calibration_cache(
        lambda_model.lambda_at_x,
        sources_with_pdf,
        X_cal_list,
        Y_cal_list,
    )

    if len(X_test) == 0:
        empty_metrics = {'coverage': np.nan}
        if task_type == 'classification':
            empty_metrics['avg_set_size'] = np.nan
        else:
            empty_metrics['avg_width'] = np.nan
        return empty_metrics, lambda_model

    lam_test = np.asarray(lambda_model.lambda_at_x(X_test))
    if lam_test.ndim == 1:
        lam_test = lam_test.reshape(1, -1)

    mdcp_results = []
    for idx, x_test in enumerate(X_test):
        conf_set = aggregated_conformal_set_multi(
            lam_model=lambda_model.lambda_at_x,
            sources=sources_with_pdf,
            X_cal_list=X_cal_list,
            Y_cal_list=Y_cal_list,
            X_test=x_test,
            Y_test=Y_test[idx],
            y_grid=y_grid,
            alpha=alpha_val,
            randomize_ties=True,
            calibration_cache=calibration_cache,
            lam_x=lam_test[idx],
        )
        mdcp_results.append(conf_set)

    if task_type == 'classification':
        metrics = evaluate_mdcp_classification_performance(
            mdcp_results,
            Y_test,
            y_grid,
            alpha_val,
            source_test=source_test,
        )
    else:
        metrics = evaluate_mdcp_regression_performance(
            mdcp_results,
            Y_test,
            y_grid,
            alpha_val,
            extend_points=1,
            source_test=source_test,
        )

    return metrics, lambda_model

