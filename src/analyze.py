"""
analyze.py
"""

import os
import torch
import random
import argparse
from collections import defaultdict
from datetime import datetime
import Levenshtein
from src.core.config import TEST_DIR, TEST_LABELS_FILE, CHARACTER_SET
from src.core.utils import decode_predictions, find_latest_checkpoint, load_model_checkpoint, decode_ground_truth
from src.models.handwriting_recognition_model import HandwritingRecognitionModel
from src.data.handwriting_dataset import HandwritingDataset
from src.data.handwriting_transforms import HandwritingTransform


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


def analyze_predictions(test_dir, test_labels, checkpoint_path, num_samples=100, use_all=False, device='cuda'):
    """
    Analyze model predictions on random test samples or entire dataset

    test_dir: Directory containing test images
    test_labels: Path to test labels CSV file
    checkpoint_path: Path to model checkpoint file to load
    num_samples: Number of random samples to analyze (ignored if use_all=True)
    use_all: If True, analyze entire dataset instead of random samples
    device: Device to test on ('cuda' or 'cpu')

    None: Results are printed and saved to file
    """
    # Configuration
    print(f"Using device: {device}")
    if use_all:
        print(f"Analyzing entire test set\n")
    else:
        print(f"Analyzing {num_samples} random samples from test set\n")

    # Find and load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")

    model = HandwritingRecognitionModel().to(device)
    model, epoch_info = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    print(f"Model loaded successfully{epoch_info}\n")

    # Load dataset
    transform = HandwritingTransform(random_blur=False, rotate=False)
    dataset = HandwritingDataset(test_dir, test_labels, transform=transform)

    print(f"Total test samples: {len(dataset)}")

    # Select samples (all or random)
    if use_all:
        sample_indices = list(range(len(dataset)))
        print(f"Analyzing all {len(sample_indices)} samples...\n")
    else:
        sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        print(f"Analyzing {len(sample_indices)} random samples...\n")

    # Run predictions
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for idx in sample_indices:
            img, label_indices = dataset[idx]
            img = img.unsqueeze(0).to(device)

            output = model(img)
            pred_text = decode_predictions(output, CHARACTER_SET)[0]
            gt_text = decode_ground_truth(label_indices)

            predictions.append(pred_text)
            ground_truths.append(gt_text)
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)

    # Print results
    print("=" * 70)
    print("PREDICTION ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nTotal Samples Analyzed: {metrics['total_samples']}")
    print(f"\n{'ACCURACY METRICS':-^70}")
    print(f"Exact Match Rate:        {metrics['exact_match_rate']:.2%} ({metrics['exact_matches']}/{metrics['total_samples']})")
    print(f"Character Error Rate:    {metrics['cer']:.2%}")
    print(f"Word Error Rate:         {metrics['wer']:.2%}")

    print(f"\n{'CHARACTER-LEVEL STATISTICS':-^70}")
    print(f"Total Characters (GT):   {metrics['total_chars_gt']}")
    print(f"Total Characters (Pred): {metrics['total_chars_pred']}")
    print(f"Character Errors:        {metrics['char_errors']}")
    print(f"Avg Length Difference:   {metrics['avg_length_diff']:+.2f} chars")

    print(f"\n{'WORD-LEVEL STATISTICS':-^70}")
    print(f"Total Words (GT):        {metrics['total_words']}")
    print(f"Word Errors:             {metrics['word_errors']}")

    # Show examples
    print(f"\n{'CORRECT PREDICTIONS (First 5)':-^70}")
    for i, (pred, gt) in enumerate(metrics['correct_predictions'][:5], 1):
        print(f"{i}. '{gt}'")

    print(f"\n{'INCORRECT PREDICTIONS (First 10)':-^70}")
    for i, (pred, gt) in enumerate(metrics['incorrect_predictions'][:10], 1):
        edit_dist = Levenshtein.distance(pred, gt)
        print(f"\n{i}. Edit Distance: {edit_dist}")
        print(f"   GT:   '{gt}'")
        print(f"   Pred: '{pred}'")

    # Common error patterns
    if metrics['common_errors']:
        print(f"\n{'MOST COMMON ERROR PATTERNS':-^70}")
        sorted_errors = sorted(metrics['common_errors'].items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_errors[:5]:
            print(f"{count}x: {pattern}")

    # Save detailed results in checkpoint directory with timestamp
    checkpoint_dir = os.path.dirname(checkpoint_path)
    analyze_dir = os.path.join(checkpoint_dir, "analyze")
    os.makedirs(analyze_dir, exist_ok=True)

    # Generate timestamp in YYMMDD_HHMMSS format
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    output_file = os.path.join(analyze_dir, f"prediction_{timestamp}_analyze_results.txt")

    with open(output_file, 'w') as f:
        # Write header with timestamp
        f.write(f"=== prediction_{timestamp} ===\n\n\n")

        # Write all metrics
        f.write("=" * 70 + "\n")
        f.write("PREDICTION ANALYSIS RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Samples Analyzed: {metrics['total_samples']}\n")
        f.write(f"\n{'ACCURACY METRICS':-^70}\n")
        f.write(f"Exact Match Rate:        {metrics['exact_match_rate']:.2%} ({metrics['exact_matches']}/{metrics['total_samples']})\n")
        f.write(f"Character Error Rate:    {metrics['cer']:.2%}\n")
        f.write(f"Word Error Rate:         {metrics['wer']:.2%}\n")

        f.write(f"\n{'CHARACTER-LEVEL STATISTICS':-^70}\n")
        f.write(f"Total Characters (GT):   {metrics['total_chars_gt']}\n")
        f.write(f"Total Characters (Pred): {metrics['total_chars_pred']}\n")
        f.write(f"Character Errors:        {metrics['char_errors']}\n")
        f.write(f"Avg Length Difference:   {metrics['avg_length_diff']:+.2f} chars\n")

        f.write(f"\n{'WORD-LEVEL STATISTICS':-^70}\n")
        f.write(f"Total Words (GT):        {metrics['total_words']}\n")
        f.write(f"Word Errors:             {metrics['word_errors']}\n")

        # Write correct predictions
        f.write(f"\n{'CORRECT PREDICTIONS (First 5)':-^70}\n")
        for i, (pred, gt) in enumerate(metrics['correct_predictions'][:5], 1):
            f.write(f"{i}. '{gt}'\n")

        # Write incorrect predictions
        f.write(f"\n{'INCORRECT PREDICTIONS (First 10)':-^70}\n")
        for i, (pred, gt) in enumerate(metrics['incorrect_predictions'][:10], 1):
            edit_dist = Levenshtein.distance(pred, gt)
            f.write(f"\n{i}. Edit Distance: {edit_dist}\n")
            f.write(f"   GT:   '{gt}'\n")
            f.write(f"   Pred: '{pred}'\n")

        # Write common error patterns
        if metrics['common_errors']:
            f.write(f"\n{'MOST COMMON ERROR PATTERNS':-^70}\n")
            sorted_errors = sorted(metrics['common_errors'].items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_errors[:5]:
                f.write(f"{count}x: {pattern}\n")

        # Write detailed predictions
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("DETAILED PREDICTION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths), 1):
            match = "OK" if pred == gt else "WRONG"
            edit_dist = Levenshtein.distance(pred, gt)
            f.write(f"Sample {i} {match} (Edit Distance: {edit_dist})\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write(f"Prediction:   {pred}\n\n")

    print(f"\n{'='*70}")
    print(f"Detailed results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model predictions on test samples")
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    parser.add_argument('--num-samples', '-n', type=int, default=100,
                       help='Number of random samples to analyze (default: 100, ignored if --all is used)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze entire test dataset instead of random samples')
    parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                       help='Directory containing test images')
    parser.add_argument('--test-labels', type=str, default=TEST_LABELS_FILE,
                       help='Path to test labels CSV file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")

    # Analyze predictions
    analyze_predictions(
        test_dir=args.test_dir,
        test_labels=args.test_labels,
        checkpoint_path=checkpoint_path,
        num_samples=args.num_samples,
        use_all=args.all,
        device=device
    )
