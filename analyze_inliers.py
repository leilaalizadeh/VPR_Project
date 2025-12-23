import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze correlation between R@1 correctness and inliers')
    parser.add_argument('--preds-dir', type=str, required=True, 
                        help='Directory with VPR predictions (e.g., cosplace_sf)')
    parser.add_argument('--matcher', type=str, required=True,
                        help='Matcher name (e.g., superpoint-lg, loftr, superglue)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory to save analysis plots')
    parser.add_argument('--positive-dist-threshold', type=float, default=25,
                        help='Distance threshold in meters for correct prediction (default: 25)')
    return parser.parse_args()


def load_ground_truth(preds_dir):
    """Load ground truth from prediction text files"""
    gt_dict = {}
    txt_files = sorted(glob(os.path.join(preds_dir, "*.txt")), 
                       key=lambda x: int(Path(x).stem))
    
    for txt_file in txt_files:
        query_id = Path(txt_file).stem
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            query_path = lines[0].strip()
            # Ground truth is the first prediction (before any re-ranking)
            # Format: "prediction_path distance_or_similarity"
            first_pred = lines[1].strip().split()[0]
            gt_dict[query_id] = {
                'query_path': query_path,
                'gt_path': first_pred  # This is actually the first prediction
            }
    return gt_dict


def check_correct_prediction(preds_dir, positive_dist_threshold=25):
    """Check which queries have correct R@1 (first prediction is correct)
    
    Args:
        preds_dir: Directory containing prediction .txt files
        positive_dist_threshold: Distance threshold in meters for correct prediction (default: 25m)
    
    Returns:
        dict: query_id -> bool (True if R@1 is correct)
    """
    correct_queries = {}
    txt_files = sorted(glob(os.path.join(preds_dir, "*.txt")),
                       key=lambda x: int(Path(x).stem))
    
    for txt_file in txt_files:
        query_id = Path(txt_file).stem
        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            
            # File format:
            # Line 0: "Query path:"
            # Line 1: actual query path
            # Line 2: blank
            # Line 3: "Predictions paths:"
            # Line 4+: prediction paths
            query_path = lines[1]
            
            # Extract UTM coordinates from query
            # Format: path/to/@utm_easting@utm_northing@...@.jpg
            try:
                query_parts = query_path.split('@')
                query_utm_e = float(query_parts[1])
                query_utm_n = float(query_parts[2])
            except (IndexError, ValueError):
                print(f"Warning: Could not extract UTM from query: {query_path}")
                correct_queries[query_id] = False
                continue
            
            # Extract UTM coordinates from first prediction (line 4)
            first_pred = lines[4]
            try:
                pred_parts = first_pred.split('@')
                pred_utm_e = float(pred_parts[1])
                pred_utm_n = float(pred_parts[2])
            except (IndexError, ValueError):
                print(f"Warning: Could not extract UTM from prediction: {first_pred}")
                correct_queries[query_id] = False
                continue
            
            # Calculate Euclidean distance
            distance = np.sqrt((query_utm_e - pred_utm_e)**2 + (query_utm_n - pred_utm_n)**2)
            
            # Check if prediction is correct (within threshold)
            is_correct = (distance <= positive_dist_threshold)
            correct_queries[query_id] = is_correct
    
    return correct_queries


def load_inliers(preds_dir, matcher_name):
    """Load inlier counts from matching results"""
    inliers_dict = {}
    
    # Load torch files for each query from the original preds directory
    torch_files = sorted(glob(os.path.join(preds_dir, "*.torch")),
                        key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else 0)
    
    # If not found, try matcher output directory
    if not torch_files:
        matcher_output_dir = preds_dir + f"_{matcher_name}"
        torch_files = sorted(glob(os.path.join(matcher_output_dir, "*.torch")),
                           key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else 0)
    
    if not torch_files:
        raise FileNotFoundError(f"No .torch files found in {preds_dir} or {preds_dir}_{matcher_name}")
    
    for torch_file in torch_files:
        query_id = Path(torch_file).stem
        if not query_id.isdigit():
            continue
            
        results = torch.load(torch_file, weights_only=False)
        # First prediction's inliers
        if len(results) > 0:
            first_result = results[0]
            # Count inliers - different matchers store this differently
            if 'num_inliers' in first_result:
                num_inliers = first_result['num_inliers']
            elif 'inliers' in first_result:
                num_inliers = len(first_result['inliers'])
            elif 'matches0' in first_result:
                matches = first_result['matches0']
                num_inliers = (matches >= 0).sum().item()
            else:
                num_inliers = 0
            
            inliers_dict[query_id] = num_inliers
    
    return inliers_dict


def analyze_and_plot(correct_queries, inliers_dict, output_dir, matcher_name, preds_dir):
    """Analyze correlation and create plots"""
    
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Collect data
    correct_inliers = []
    wrong_inliers = []
    
    for query_id in correct_queries.keys():
        if query_id in inliers_dict:
            num_inliers = inliers_dict[query_id]
            if correct_queries[query_id]:
                correct_inliers.append(num_inliers)
            else:
                wrong_inliers.append(num_inliers)
    
    correct_inliers = np.array(correct_inliers)
    wrong_inliers = np.array(wrong_inliers)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Analysis Results: {Path(preds_dir).name} + {matcher_name}")
    print(f"{'='*60}")
    print(f"Total queries analyzed: {len(correct_queries)}")
    print(f"Correct R@1 predictions: {len(correct_inliers)} ({len(correct_inliers)/len(correct_queries)*100:.1f}%)")
    print(f"Wrong R@1 predictions: {len(wrong_inliers)} ({len(wrong_inliers)/len(correct_queries)*100:.1f}%)")
    print(f"\nInlier Statistics:")
    print(f"  Correct queries - Mean: {correct_inliers.mean():.2f}, Median: {np.median(correct_inliers):.2f}, Std: {correct_inliers.std():.2f}")
    print(f"  Wrong queries   - Mean: {wrong_inliers.mean():.2f}, Median: {np.median(wrong_inliers):.2f}, Std: {wrong_inliers.std():.2f}")
    print(f"  Mean difference: {correct_inliers.mean() - wrong_inliers.mean():.2f}")
    print(f"{'='*60}\n")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Inlier Analysis: {Path(preds_dir).name} + {matcher_name}', fontsize=16, fontweight='bold')
    
    # 1. Overlapping histograms
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(max(correct_inliers, default=0), max(wrong_inliers, default=0)), 50)
    ax1.hist(correct_inliers, bins=bins, alpha=0.6, label=f'Correct R@1 (n={len(correct_inliers)})', color='green', edgecolor='black')
    ax1.hist(wrong_inliers, bins=bins, alpha=0.6, label=f'Wrong R@1 (n={len(wrong_inliers)})', color='red', edgecolor='black')
    ax1.set_xlabel('Number of Inliers', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Inliers (Overlapping)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Side-by-side histograms
    ax2 = axes[0, 1]
    x_pos = np.arange(2)
    means = [correct_inliers.mean(), wrong_inliers.mean()]
    stds = [correct_inliers.std(), wrong_inliers.std()]
    colors_bar = ['green', 'red']
    ax2.bar(x_pos, means, yerr=stds, color=colors_bar, alpha=0.7, capsize=10, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Correct R@1', 'Wrong R@1'], fontsize=11)
    ax2.set_ylabel('Mean Number of Inliers', fontsize=12)
    ax2.set_title('Mean Inliers Comparison (with Std Dev)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 5, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Box plot
    ax3 = axes[1, 0]
    box_data = [correct_inliers, wrong_inliers]
    bp = ax3.boxplot(box_data, labels=['Correct R@1', 'Wrong R@1'], patch_artist=True,
                      boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Number of Inliers', fontsize=12)
    ax3.set_title('Inlier Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    if len(correct_inliers) > 0:
        sorted_correct = np.sort(correct_inliers)
        cumulative_correct = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct) * 100
        ax4.plot(sorted_correct, cumulative_correct, label='Correct R@1', color='green', linewidth=2)
    
    if len(wrong_inliers) > 0:
        sorted_wrong = np.sort(wrong_inliers)
        cumulative_wrong = np.arange(1, len(sorted_wrong) + 1) / len(sorted_wrong) * 100
        ax4.plot(sorted_wrong, cumulative_wrong, label='Wrong R@1', color='red', linewidth=2)
    
    ax4.set_xlabel('Number of Inliers', fontsize=12)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax4.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f"{Path(preds_dir).name}_{matcher_name}_inlier_analysis.png"
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Additional analysis: threshold-based classification
    print("\nThreshold-based Classification Analysis:")
    print("-" * 60)
    
    # Try different thresholds
    thresholds = [10, 20, 50, 100, 200]
    for threshold in thresholds:
        correct_above = np.sum(correct_inliers >= threshold)
        correct_below = np.sum(correct_inliers < threshold)
        wrong_above = np.sum(wrong_inliers >= threshold)
        wrong_below = np.sum(wrong_inliers < threshold)
        
        total = len(correct_inliers) + len(wrong_inliers)
        if total > 0:
            accuracy = (correct_above + wrong_below) / total * 100
            print(f"Threshold = {threshold:3d} inliers: "
                  f"Accuracy = {accuracy:5.2f}% "
                  f"(Correct≥threshold: {correct_above}/{len(correct_inliers)}, "
                  f"Wrong<threshold: {wrong_below}/{len(wrong_inliers)})")


def main():
    args = parse_arguments()
    
    print(f"Loading data from: {args.preds_dir}")
    print(f"Matcher: {args.matcher}")
    
    # Check R@1 correctness
    print("Checking R@1 correctness...")
    correct_queries = check_correct_prediction(args.preds_dir, args.positive_dist_threshold)
    
    # Load inliers
    print("Loading inlier counts...")
    inliers_dict = load_inliers(args.preds_dir, args.matcher)
    
    # Analyze and plot
    print("Generating analysis and plots...")
    analyze_and_plot(correct_queries, inliers_dict, args.output_dir, args.matcher, args.preds_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
