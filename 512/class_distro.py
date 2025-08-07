#!/usr/bin/env python3
"""
Hi-C TAD Boundary Distribution Comparator

This script compares TAD boundary distributions between ground truth and predictions
for both training and test sets. It analyzes:
1. Training set ground truth TAD boundary distribution
2. Training set prediction TAD boundary distribution  
3. Test set ground truth TAD boundary distribution
4. Test set prediction TAD boundary distribution

The script reads train.txt and test.txt files to determine train/test splits,
then analyzes TAD boundary masks from two directories (ground truth and predictions).

Requirements:
- numpy
- opencv-python (cv2)
- matplotlib
- Pillow (PIL)
- pandas
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import pandas as pd

def read_file_pairs(file_path):
    """
    Read a text file containing Hi-C matrix and TAD boundary pairs and extract mask filenames.
    
    Args:
        file_path (str): Path to the text file containing matrix-mask pairs
        
    Returns:
        list: List of TAD boundary mask filenames (just the filename, not full path)
    """
    mask_filenames = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by space and take the second part (TAD boundary mask path)
                    parts = line.split()
                    if len(parts) >= 2:
                        mask_path = parts[1]  # Second column is the TAD boundary mask path
                        # Extract just the filename from the path
                        mask_filename = os.path.basename(mask_path)
                        mask_filenames.append(mask_filename)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []
    
    return mask_filenames

def analyze_single_tad_mask(image_path):
    """
    Analyze a single TAD boundary mask image and return boundary distribution.
    Handles TIF files with exact binary values: 0 (background) and 1 (TAD boundaries).
    
    Args:
        image_path (str): Full path to the TAD boundary mask image
        
    Returns:
        dict: Dictionary containing TAD boundary distribution info
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            return None
        
        # Load image using PIL to handle TIF files properly
        img = Image.open(image_path)
        
        # Convert to numpy array
        mask = np.array(img)
        
        # Handle different image formats (RGB, RGBA, grayscale)
        if len(mask.shape) == 3:
            # Convert to grayscale if multi-channel
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Debug: Check unique values in the mask
        unique_values = np.unique(mask)
        print(f"Debug - {os.path.basename(image_path)}: unique values = {unique_values}")
        
        # Handle different possible encodings:
        # Case 1: Values are exactly 0 and 1 (ideal case)
        if set(unique_values).issubset({0, 1}):
            binary_mask = mask.astype(np.uint8)
        # Case 2: Values are 0 and 255 (standard binary image)
        elif set(unique_values).issubset({0, 255}):
            binary_mask = (mask == 255).astype(np.uint8)
        # Case 3: Values are in range 0-255, use threshold
        else:
            # Use threshold for any other case
            binary_mask = (mask > 0.5 * mask.max()).astype(np.uint8)
        
        print(f"Debug - After conversion: unique values = {np.unique(binary_mask)}")
        
        # Calculate TAD boundary distribution
        total_pixels = binary_mask.size
        boundary_pixels = np.sum(binary_mask == 1)  # Count class 1 (TAD boundaries)
        background_pixels = np.sum(binary_mask == 0)  # Count class 0 (background)
        
        # Verify counts add up
        assert boundary_pixels + background_pixels == total_pixels, \
            f"Pixel counts don't add up: {boundary_pixels} + {background_pixels} != {total_pixels}"
        
        boundary_ratio = boundary_pixels / total_pixels
        background_ratio = background_pixels / total_pixels
        
        # Additional TAD-specific metrics
        # Count connected components (individual TAD boundaries)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        num_boundaries = num_labels - 1  # Subtract 1 for background
        
        # Calculate average boundary length (perimeter)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_boundary_length = sum(cv2.arcLength(contour, False) for contour in contours)
        avg_boundary_length = total_boundary_length / max(num_boundaries, 1)
        
        result = {
            'filename': os.path.basename(image_path),
            'full_path': image_path,
            'total_pixels': total_pixels,
            'background_pixels': background_pixels,  # Class 0
            'boundary_pixels': boundary_pixels,      # Class 1
            'background_ratio': background_ratio,    # Class 0 ratio
            'boundary_ratio': boundary_ratio,        # Class 1 ratio
            'num_boundaries': num_boundaries,
            'total_boundary_length': total_boundary_length,
            'avg_boundary_length': avg_boundary_length,
            'boundary_density': boundary_pixels / total_pixels if total_pixels > 0 else 0,
            'image_shape': binary_mask.shape,
            'original_unique_values': unique_values.tolist()  # For debugging
        }
        
        # Print verification for first few files
        print(f"  {os.path.basename(image_path)}: {boundary_pixels}/{total_pixels} = {boundary_ratio:.6f} boundary ratio")
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def analyze_tad_mask_set(mask_filenames, mask_directory, set_name="", is_predictions=False):
    """
    Analyze a set of TAD boundary mask images from a directory and return statistics.
    
    Args:
        mask_filenames (list): List of TAD boundary mask filenames to analyze
        mask_directory (str): Directory containing the mask files
        set_name (str): Name of the dataset for logging
        is_predictions (bool): True if analyzing predictions (adds _mask suffix)
        
    Returns:
        pd.DataFrame: DataFrame containing analysis results
    """
    print(f"Analyzing {len(mask_filenames)} TAD boundary masks for {set_name}...")
    
    results = []
    successful = 0
    missing = 0
    
    for filename in mask_filenames:
        # Handle prediction files that have _mask suffix
        if is_predictions:
            # Remove extension and add _mask.tif
            base_name = os.path.splitext(filename)[0]
            prediction_filename = f"{base_name}.tif"
            full_path = os.path.join(mask_directory, prediction_filename)
        else:
            # Ground truth files use original filename
            full_path = os.path.join(mask_directory, filename)
        
        result = analyze_single_tad_mask(full_path)
        if result:
            results.append(result)
            successful += 1
        else:
            missing += 1
            if is_predictions:
                print(f"Warning: Prediction file not found: {prediction_filename}")
    
    print(f"Successfully processed {successful}/{len(mask_filenames)} TAD masks for {set_name}")
    if missing > 0:
        print(f"Warning: {missing} TAD masks were missing or could not be processed for {set_name}")
    
    if not results:
        print(f"Warning: No valid TAD masks found for {set_name}")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Print summary of unique value patterns found
    if not df.empty and 'original_unique_values' in df.columns:
        unique_patterns = df['original_unique_values'].apply(tuple).value_counts()
        print(f"\nUnique value patterns found in {set_name}:")
        for pattern, count in unique_patterns.items():
            print(f"  {list(pattern)}: {count} files")
    
    return df

def print_tad_summary_stats(df, set_name):
    """Print summary statistics for a TAD boundary dataset."""
    if df.empty:
        print(f"\nNo data available for {set_name}")
        return
    
    print(f"\n{set_name} STATISTICS:")
    print("-" * 60)
    print(f"Total Hi-C matrices processed: {len(df)}")
    print(f"Average TAD boundary ratio: {df['boundary_ratio'].mean():.6f}")
    print(f"Std dev TAD boundary ratio: {df['boundary_ratio'].std():.6f}")
    print(f"Min TAD boundary ratio: {df['boundary_ratio'].min():.6f}")
    print(f"Max TAD boundary ratio: {df['boundary_ratio'].max():.6f}")
    print(f"Median TAD boundary ratio: {df['boundary_ratio'].median():.6f}")
    print(f"25th percentile: {df['boundary_ratio'].quantile(0.25):.6f}")
    print(f"75th percentile: {df['boundary_ratio'].quantile(0.75):.6f}")
    print(f"Average number of boundaries per matrix: {df['num_boundaries'].mean():.2f}")
    print(f"Average boundary length: {df['avg_boundary_length'].mean():.2f} pixels")
    print(f"Average boundary density: {df['boundary_density'].mean():.6f}")

def create_tad_comparison_plots(train_gt_df, train_pred_df, test_gt_df, test_pred_df, output_dir=None):
    """
    Create comprehensive comparison plots for all four TAD boundary distributions.
    
    Args:
        train_gt_df, train_pred_df, test_gt_df, test_pred_df: DataFrames with analysis results
        output_dir: Directory to save plots (optional)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle('Hi-C TAD Boundary Distribution Comparison: Ground Truth vs Predictions', 
                 fontsize=16, fontweight='bold')
    
    
    # Define colors for consistency
    colors = {
        'train_gt': 'blue',
        'train_pred': 'purple', 
        'test_gt': 'red',
        'test_pred': 'orange'
    }
    
    # 1. Training set TAD boundary ratio comparison
    if not train_gt_df.empty and not train_pred_df.empty:
        axes[0, 0].hist([train_gt_df['boundary_ratio'], train_pred_df['boundary_ratio']], 
                       bins=50, alpha=0.7, 
                       label=['Ground Truth', 'Predictions'],
                       color=[colors['train_gt'], colors['train_pred']])
        axes[0, 0].set_title('Training Set: TAD Boundary Ratio Comparison')
        axes[0, 0].set_xlabel('TAD Boundary Ratio')
        axes[0, 0].set_ylabel('Number of Hi-C Matrices')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test set TAD boundary ratio comparison  
    if not test_gt_df.empty and not test_pred_df.empty:
        axes[0, 1].hist([test_gt_df['boundary_ratio'], test_pred_df['boundary_ratio']], 
                       bins=50, alpha=0.7,
                       label=['Ground Truth', 'Predictions'],
                       color=[colors['test_gt'], colors['test_pred']])
        axes[0, 1].set_title('Test Set: TAD Boundary Ratio Comparison')
        axes[0, 1].set_xlabel('TAD Boundary Ratio')
        axes[0, 1].set_ylabel('Number of Hi-C Matrices')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of boundaries comparison (train)
    if not train_gt_df.empty and not train_pred_df.empty:
        axes[1, 0].hist([train_gt_df['num_boundaries'], train_pred_df['num_boundaries']], 
                       bins=50, alpha=0.7,
                       label=['Ground Truth', 'Predictions'],
                       color=[colors['train_gt'], colors['train_pred']])
        axes[1, 0].set_title('Training Set: Number of TAD Boundaries')
        axes[1, 0].set_xlabel('Number of Boundaries per Matrix')
        axes[1, 0].set_ylabel('Number of Hi-C Matrices')
        axes[1, 0].legend()
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_visible(False) 
    
    
    # 4. Number of boundaries comparison (test)
    if not test_gt_df.empty and not test_pred_df.empty:
        axes[1, 1].hist([test_gt_df['num_boundaries'], test_pred_df['num_boundaries']], 
                       bins=50, alpha=0.7,
                       label=['Ground Truth', 'Predictions'],
                       color=[colors['test_gt'], colors['test_pred']])
        axes[1, 1].set_title('Test Set: Number of TAD Boundaries')
        axes[1, 1].set_xlabel('Number of Boundaries per Matrix')
        axes[1, 1].set_ylabel('Number of Hi-C Matrices')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_visible(False) 
    
    # 5. Ground truth comparison (train vs test)
    if not train_gt_df.empty and not test_gt_df.empty:
        axes[2, 0].hist([train_gt_df['boundary_ratio'], test_gt_df['boundary_ratio']], 
                       bins=50, alpha=0.7,
                       label=['Training', 'Test'],
                       color=[colors['train_gt'], colors['test_gt']])
        axes[2, 0].set_title('Ground Truth: Training vs Test TAD Boundaries')
        axes[2, 0].set_xlabel('TAD Boundary Ratio')
        axes[2, 0].set_ylabel('Number of Hi-C Matrices')
        axes[2, 0].legend()
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Predictions comparison (train vs test)
    if not train_pred_df.empty and not test_pred_df.empty:
        axes[2, 1].hist([train_pred_df['boundary_ratio'], test_pred_df['boundary_ratio']], 
                       bins=50, alpha=0.7,
                       label=['Training', 'Test'],
                       color=[colors['train_pred'], colors['test_pred']])
        axes[2, 1].set_title('Predictions: Training vs Test TAD Boundaries')
        axes[2, 1].set_xlabel('TAD Boundary Ratio')
        axes[2, 1].set_ylabel('Number of Hi-C Matrices')
        axes[2, 1].legend()
        axes[2,1].set_xlim(0, 1)
        axes[2, 1].grid(True, alpha=0.3)
    
    # 7. Box plots comparison - Boundary Ratio
    box_data = []
    box_labels = []
    
    for df, label in [(train_gt_df, 'Train GT'), (train_pred_df, 'Train Pred'), 
                      (test_gt_df, 'Test GT'), (test_pred_df, 'Test Pred')]:
        if not df.empty:
            box_data.append(df['boundary_ratio'])
            box_labels.append(label)
    
    if box_data:
        axes[3, 0].boxplot(box_data, labels=box_labels)
        axes[3, 0].set_title('Box Plot: TAD Boundary Ratio Comparison')
        axes[3, 0].set_ylabel('TAD Boundary Ratio')
        axes[3, 0].tick_params(axis='x', rotation=45)
        axes[3, 0].grid(True, alpha=0.3)
    
    # 8. Summary statistics table
    axes[3, 1].axis('off')
    table_data = []
    
    for df, name in [(train_gt_df, 'Train GT'), (train_pred_df, 'Train Pred'),
                     (test_gt_df, 'Test GT'), (test_pred_df, 'Test Pred')]:
        if not df.empty:
            table_data.append([
                name,
                len(df),
                f"{df['boundary_ratio'].mean():.6f}",
                f"{df['boundary_ratio'].std():.6f}"
            ])
    
    if table_data:
        table = axes[3, 1].table(cellText=table_data,
                               colLabels=['Dataset', 'Count', 'Mean Ratio', 'Std Ratio', 'Avg Boundaries'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[3, 1].set_title('Summary Statistics')
    
    plt.tight_layout(rect=[0, 0, 1, 0.85]) 
    
    # Save plot if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / 'tad_boundary_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"TAD boundary comparison plot saved to: {plot_path}")
    
    plt.show()

def save_detailed_tad_results(train_gt_df, train_pred_df, test_gt_df, test_pred_df, output_dir):
    """Save detailed CSV files for each TAD dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    datasets = [
        (train_gt_df, 'train_tad_ground_truth.csv'),
        (train_pred_df, 'train_tad_predictions.csv'), 
        (test_gt_df, 'test_tad_ground_truth.csv'),
        (test_pred_df, 'test_tad_predictions.csv')
    ]
    
    for df, filename in datasets:
        if not df.empty:
            csv_path = output_dir / filename
            df.to_csv(csv_path, index=False)
            print(f"Detailed TAD results saved to: {csv_path}")

def main():
    """Main function to handle command line arguments and run TAD boundary analysis."""
    parser = argparse.ArgumentParser(
        description="Compare Hi-C TAD boundary distributions between ground truth and predictions for train/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hic_tad_comparator.py train.txt test.txt ground_truth_masks predictions_masks
  python hic_tad_comparator.py train.txt test.txt /path/to/gt_tad_masks /path/to/pred_tad_masks --output results/
        """
    )
    
    parser.add_argument('train_file', help='Path to train.txt file (defines training set)')
    parser.add_argument('test_file', help='Path to test.txt file (defines test set)')
    parser.add_argument('ground_truth_dir', help='Directory containing all ground truth TAD boundary masks')
    parser.add_argument('predictions_dir', help='Directory containing all predicted TAD boundary masks')
    parser.add_argument('-o', '--output', help='Output directory for saving results (optional)')
    
    args = parser.parse_args()
    
    try:
        # Read file lists to determine train/test splits
        print("Reading Hi-C train/test split files...")
        train_mask_filenames = read_file_pairs(args.train_file)
        test_mask_filenames = read_file_pairs(args.test_file)
        
        if not train_mask_filenames:
            raise ValueError(f"No TAD mask filenames found in {args.train_file}")
        if not test_mask_filenames:
            raise ValueError(f"No TAD mask filenames found in {args.test_file}")
        
        print(f"Found {len(train_mask_filenames)} training Hi-C matrices and {len(test_mask_filenames)} test Hi-C matrices")
        
        # Check if directories exist
        if not os.path.exists(args.ground_truth_dir):
            raise ValueError(f"Ground truth TAD directory does not exist: {args.ground_truth_dir}")
        if not os.path.exists(args.predictions_dir):
            raise ValueError(f"Predictions TAD directory does not exist: {args.predictions_dir}")
        
        # Analyze all four TAD boundary distributions
        print("\nAnalyzing TAD boundary distributions...")
        train_gt_df = analyze_tad_mask_set(train_mask_filenames, args.ground_truth_dir, "Training Ground Truth TADs", is_predictions=False)
        train_pred_df = analyze_tad_mask_set(train_mask_filenames, args.predictions_dir, "Training Predicted TADs", is_predictions=True)
        test_gt_df = analyze_tad_mask_set(test_mask_filenames, args.ground_truth_dir, "Test Ground Truth TADs", is_predictions=False)
        test_pred_df = analyze_tad_mask_set(test_mask_filenames, args.predictions_dir, "Test Predicted TADs", is_predictions=True)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("HI-C TAD BOUNDARY DISTRIBUTION COMPARISON RESULTS")
        print("="*80)
        
        print_tad_summary_stats(train_gt_df, "TRAINING GROUND TRUTH TAD BOUNDARIES")
        print_tad_summary_stats(train_pred_df, "TRAINING PREDICTED TAD BOUNDARIES")
        print_tad_summary_stats(test_gt_df, "TEST GROUND TRUTH TAD BOUNDARIES")
        print_tad_summary_stats(test_pred_df, "TEST PREDICTED TAD BOUNDARIES")
        
        # Create comparison plots
        create_tad_comparison_plots(train_gt_df, train_pred_df, test_gt_df, test_pred_df, args.output)
        
        # Save detailed results if output directory specified
        if args.output:
            save_detailed_tad_results(train_gt_df, train_pred_df, test_gt_df, test_pred_df, args.output)
            
            # Save summary statistics
            summary_path = Path(args.output) / 'tad_boundary_comparison_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("Hi-C TAD Boundary Distribution Comparison Summary\n")
                f.write("=" * 60 + "\n\n")
                
                for df, name in [(train_gt_df, "Training Ground Truth TADs"),
                               (train_pred_df, "Training Predicted TADs"),
                               (test_gt_df, "Test Ground Truth TADs"), 
                               (test_pred_df, "Test Predicted TADs")]:
                    if not df.empty:
                        f.write(f"{name}:\n")
                        f.write(f"  Hi-C Matrices: {len(df)}\n")
                        f.write(f"  Mean boundary ratio: {df['boundary_ratio'].mean():.6f}\n")
                        f.write(f"  Std boundary ratio: {df['boundary_ratio'].std():.6f}\n")
                        f.write(f"  Median boundary ratio: {df['boundary_ratio'].median():.6f}\n")
                        f.write(f"  Avg boundaries per matrix: {df['num_boundaries'].mean():.2f}\n")
                        f.write(f"  Avg boundary length: {df['avg_boundary_length'].mean():.2f}\n\n")
            
            print(f"TAD boundary summary saved to: {summary_path}")
        
        print(f"\nHi-C TAD boundary analysis complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())