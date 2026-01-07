"""
metrics.py

Evaluation metrics calculation (CER, WER, accuracy)
"""

import Levenshtein
from collections import defaultdict


def calculate_cer(predictions, ground_truths):
    """
    Calculate Character Error Rate (CER) for a batch of predictions

    predictions: List of predicted text strings
    ground_truths: List of ground truth text strings

    cer: Character Error Rate as a percentage
    """
    total_distance = 0
    total_length = 0

    for pred, truth in zip(predictions, ground_truths):
        distance = Levenshtein.distance(pred, truth)
        total_distance += distance
        total_length += len(truth)

    if total_length == 0:
        return 0.0

    return (total_distance / total_length) * 100


def calculate_metrics(predictions, ground_truths):
    """
    Calculate comprehensive metrics for predictions vs ground truths

    predictions: List of predicted text strings
    ground_truths: List of ground truth text strings

    metrics: Dictionary containing all calculated metrics
    """
    metrics = {
        'total_samples': len(predictions),
        'exact_matches': 0,
        'total_edit_distance': 0,
        'total_chars_pred': 0,
        'total_chars_gt': 0,
        'char_errors': 0,
        'word_errors': 0,
        'total_words': 0,
        'length_differences': [],
        'common_errors': defaultdict(int),
        'correct_predictions': [],
        'incorrect_predictions': []
    }

    for pred, gt in zip(predictions, ground_truths):
        # Exact match
        if pred == gt:
            metrics['exact_matches'] += 1
            metrics['correct_predictions'].append((pred, gt))
        else:
            metrics['incorrect_predictions'].append((pred, gt))

        # Character-level metrics
        edit_dist = Levenshtein.distance(pred, gt)
        metrics['total_edit_distance'] += edit_dist
        metrics['total_chars_pred'] += len(pred)
        metrics['total_chars_gt'] += len(gt)
        metrics['char_errors'] += edit_dist

        # Length difference
        metrics['length_differences'].append(len(pred) - len(gt))

        # Word-level metrics
        pred_words = pred.split()
        gt_words = gt.split()
        metrics['total_words'] += len(gt_words)
        metrics['word_errors'] += Levenshtein.distance(pred_words, gt_words)

        # Track common error patterns
        if pred != gt:
            metrics['common_errors'][f"{gt[:20]}... -> {pred[:20]}..."] += 1

    # Calculate rates
    metrics['exact_match_rate'] = metrics['exact_matches'] / metrics['total_samples']
    metrics['cer'] = metrics['char_errors'] / max(metrics['total_chars_gt'], 1)
    metrics['wer'] = metrics['word_errors'] / max(metrics['total_words'], 1)
    metrics['avg_length_diff'] = sum(metrics['length_differences']) / len(metrics['length_differences'])

    return metrics
