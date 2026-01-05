"""
Data Utilities

This module contains data processing and splitting utilities for multi-source datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def combine_sources_three_way(X_sources, Y_sources, train_size=0.6, cal_size=0.2, test_size=0.2, stratify=True):
    """
    Combine multi-source data and create train/calibration/test splits using 60-20-20 principle.
    
    :param X_sources: list
        List of feature arrays
    :param Y_sources: list
        List of target arrays
    :param train_size: float
        Fraction of data to use for training (default: 0.6)
    :param cal_size: float
        Fraction of data to use for calibration (default: 0.2)
    :param test_size: float
        Fraction of data to use for testing (default: 0.2)
    :param stratify: bool
        Whether to stratify the split (for classification)
        
    :returns: tuple
        (X_train, X_cal, X_test, Y_train, Y_cal, Y_test, 
         source_train, source_cal, source_test)
    """
    total_size = train_size + cal_size + test_size
    train_size = train_size / total_size
    cal_size = cal_size / total_size
    test_size = test_size / total_size
    
    X_all = np.vstack(X_sources)
    Y_all = np.hstack(Y_sources)
    
    # Create source indicators
    source_labels = []
    for j, X_j in enumerate(X_sources):
        source_labels.extend([j] * len(X_j))  # (j, j, ..., j) for len(X_j) times
    source_labels = np.array(source_labels)
    
    # First split: train vs (cal + test)
    temp_test_size = cal_size + test_size
    if stratify and len(np.unique(Y_all)) > 1:
        try:
            X_train, X_temp, Y_train, Y_temp, source_train, source_temp = train_test_split(
                X_all, Y_all, source_labels, test_size=temp_test_size, 
                stratify=Y_all, random_state=42
            )
        except ValueError:
            # Fallback to non-stratified if stratification fails
            X_train, X_temp, Y_train, Y_temp, source_train, source_temp = train_test_split(
                X_all, Y_all, source_labels, test_size=temp_test_size, random_state=42
            )
    else:
        X_train, X_temp, Y_train, Y_temp, source_train, source_temp = train_test_split(
            X_all, Y_all, source_labels, test_size=temp_test_size, random_state=42
        )
    
    # Second split: calibration vs test
    cal_ratio = cal_size / temp_test_size  # Ratio of cal in the remaining data
    if stratify and len(np.unique(Y_temp)) > 1:
        try:
            X_cal, X_test, Y_cal, Y_test, source_cal, source_test = train_test_split(
                X_temp, Y_temp, source_temp, test_size=(1-cal_ratio), 
                stratify=Y_temp, random_state=42
            )
        except ValueError:
            X_cal, X_test, Y_cal, Y_test, source_cal, source_test = train_test_split(
                X_temp, Y_temp, source_temp, test_size=(1-cal_ratio), random_state=42
            )
    else:
        X_cal, X_test, Y_cal, Y_test, source_cal, source_test = train_test_split(
            X_temp, Y_temp, source_temp, test_size=(1-cal_ratio), random_state=42
        )
    
    return X_train, X_cal, X_test, Y_train, Y_cal, Y_test, source_train, source_cal, source_test


def reconstruct_source_data(X_data, Y_data, source_data, n_sources):
    """
    Reconstruct source-specific data from combined data arrays.
    
    :param X_data: array-like
        Combined feature data
    :param Y_data: array-like
        Combined target data
    :param source_data: array-like
        Source indicator array
    :param n_sources: int
        Number of sources
        
    :returns: tuple
        (X_sources, Y_sources) - lists of source-specific data
    """
    X_sources = []
    Y_sources = []
    
    for j in range(n_sources):
        mask = source_data == j
        if np.sum(mask) > 0:
            X_sources.append(X_data[mask])
            Y_sources.append(Y_data[mask])
        else:
            # If no data for this source, create empty arrays
            X_sources.append(np.empty((0, X_data.shape[1])))
            Y_sources.append(np.empty(0))
    
    return X_sources, Y_sources


def get_data_split_summary(X_train, X_cal, X_test, Y_train, Y_cal, Y_test, task='classification'):
    """
    Get summary statistics for the three-way data split.
    
    :param task: str
        Either 'classification' or 'regression'
        
    :returns: dict
        Summary statistics
    """
    summary = {
        'total_samples': len(X_train) + len(X_cal) + len(X_test),
        'train_samples': len(X_train),
        'cal_samples': len(X_cal),
        'test_samples': len(X_test),
        'train_ratio': len(X_train) / (len(X_train) + len(X_cal) + len(X_test)),
        'cal_ratio': len(X_cal) / (len(X_train) + len(X_cal) + len(X_test)),
        'test_ratio': len(X_test) / (len(X_train) + len(X_cal) + len(X_test))
    }
    
    if task == 'classification':
        summary['train_class_dist'] = np.bincount(Y_train)
        summary['cal_class_dist'] = np.bincount(Y_cal)
        summary['test_class_dist'] = np.bincount(Y_test)
    elif task == 'regression':
        summary['train_target_stats'] = {
            'mean': np.mean(Y_train), 'std': np.std(Y_train),
            'min': np.min(Y_train), 'max': np.max(Y_train)
        }
        summary['cal_target_stats'] = {
            'mean': np.mean(Y_cal), 'std': np.std(Y_cal),
            'min': np.min(Y_cal), 'max': np.max(Y_cal)
        }
        summary['test_target_stats'] = {
            'mean': np.mean(Y_test), 'std': np.std(Y_test),
            'min': np.min(Y_test), 'max': np.max(Y_test)
        }
    
    return summary
