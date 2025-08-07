#!/usr/bin/env python3
"""
Mask Processing and Probability Visualization Tool

This script processes mask images to:
1. Calculate probability distributions/decay constants
2. Generate gradient images showing the distribution
3. Apply weighted predictions based on crosshair probabilities
4. Visualize weight matrices
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Tuple, List, Optional
import seaborn as sns
from scipy import ndimage
from skimage import measure


def parse_training_file(filepath: str) -> List[Tuple[str, str]]:
    """Parse the training file to extract image and mask pairs."""
    pairs = []
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
        
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            image_path = parts[0]
            mask_path = parts[1]
            pairs.append((image_path, mask_path))
    
    return pairs


def load_image(filepath: str) -> np.ndarray:
    """Load image file (supports various formats including TIFF)."""
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def calculate_decay_constant(mask: np.ndarray) -> float:
    """Calculate decay constant from mask probability distribution."""
    # Normalize mask to [0, 1]
    mask_norm = mask.astype(np.float32) / 255.0
    
    # Calculate center of mass
    center_y, center_x = ndimage.center_of_mass(mask_norm)
    
    # Calculate distances from center
    y_indices, x_indices = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Calculate decay constant using exponential fit
    # Avoid division by zero
    mask_flat = mask_norm.flatten()
    dist_flat = distances.flatten()
    
    # Remove zero values for log calculation
    non_zero_idx = mask_flat > 0.001
    if np.sum(non_zero_idx) > 0:
        log_probs = np.log(mask_flat[non_zero_idx])
        dists = dist_flat[non_zero_idx]
        
        # Linear regression to find decay constant
        if len(dists) > 1:
            decay_const = -np.polyfit(dists, log_probs, 1)[0]
            return max(decay_const, 0.01)  # Ensure positive
    
    return 0.1  # Default fallback


def generate_gradient_image(decay_const: float, size: Tuple[int, int]) -> np.ndarray:
    """Generate gradient image showing probability distribution."""
    height, width = size
    center_x, center_y = width // 2, height // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Apply exponential decay
    gradient = np.exp(-decay_const * distances)
    
    # Normalize to [0, 255]
    gradient = (gradient * 255).astype(np.uint8)
    
    return gradient


def create_crosshair_probabilities(gradient: np.ndarray) -> np.ndarray:
    """Create crosshair probability map from gradient."""
    height, width = gradient.shape
    center_x, center_y = width // 2, height // 2
    
    # Create crosshair pattern
    crosshair = np.zeros_like(gradient, dtype=np.float32)
    
    # Horizontal line
    crosshair[center_y, :] = 1.0
    
    # Vertical line
    crosshair[:, center_x] = 1.0
    
    # Apply gradient weighting
    gradient_norm = gradient.astype(np.float32) / 255.0
    crosshair_prob = crosshair * gradient_norm
    
    return crosshair_prob


def apply_weighted_predictions(predictions: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """Apply weight matrix to predictions using (1 - weight_matrix)."""
    weight_applied = 1.0 - weight_matrix
    return predictions * weight_applied


def visualize_weight_matrix(weight_matrix: np.ndarray, title: str = "Weight Matrix") -> plt.Figure:
    """Visualize the weight matrix using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(weight_matrix, cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight Value')
    
    plt.tight_layout()
    return fig


def process_single_mask(mask_path: str, predictions_dir: str, output_size: Tuple[int, int]) -> dict:
    """Process a single mask file and return results."""
    # Load mask from predictions directory
    mask_filename = Path(mask_path).name
    prediction_mask_path = os.path.join(predictions_dir, mask_filename)
    
    if not os.path.exists(prediction_mask_path):
        print(f"Warning: Prediction mask not found: {prediction_mask_path}")
        return None
    mask = load_image(prediction_mask_path)
    if mask is None:
        print(f"Error: Could not load mask: {prediction_mask_path}")
        return None
    
    # Calculate decay constant
    decay_const = calculate_decay_constant(mask)
    
    # Generate gradient image
    gradient = generate_gradient_image(decay_const, output_size)
    
    # Create crosshair probabilities
    crosshair_probs = create_crosshair_probabilities(gradient)
    
    # Resize original mask to match output size for weight matrix
    mask_resized = cv2.resize(mask, (output_size[1], output_size[0]))
    weight_matrix = mask_resized.astype(np.float32) / 255.0
    
    # Apply weighted predictions
    weighted_predictions = apply_weighted_predictions(crosshair_probs, weight_matrix)
    
    return {
        'mask_path': prediction_mask_path,
        'decay_constant': decay_const,
        'gradient': gradient,
        'crosshair_probs': crosshair_probs,
        'weight_matrix': weight_matrix,
        'weighted_predictions': weighted_predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Process masks and generate probability visualizations')
    parser.add_argument('training_file', help='Path to training file with image/mask pairs')
    parser.add_argument('--predictions_dir', default='predictions', 
                       help='Directory containing prediction masks')
    parser.add_argument('--output_dir', default='output_gradient', 
                       help='Directory to save output visualizations')
    parser.add_argument('--image_size', type=int, default=512, 
                       help='Size for generated images (creates square images)')
    parser.add_argument('--save_individual', action='store_true', 
                       help='Save individual visualizations for each mask')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse training file
    pairs = parse_training_file(args.training_file)
    print(f"Found {len(pairs)} image/mask pairs")
    
    # Process each mask
    results = []
    output_size = (args.image_size, args.image_size)
    
    for i, (image_path, mask_path) in enumerate(pairs):
        print(f"Processing {i+1}/{len(pairs)}: {mask_path}")
        
        result = process_single_mask(mask_path, args.predictions_dir, output_size)
        if result:
            results.append(result)
            
            # Save individual visualizations if requested
            if args.save_individual:
                base_name = Path(mask_path).stem
                
                # Save gradient image
                cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_gradient.png"), 
                           result['gradient'])
                
                # Save weight matrix visualization
                fig = visualize_weight_matrix(result['weight_matrix'], 
                                            f"Weight Matrix - {base_name}")
                fig.savefig(os.path.join(args.output_dir, f"{base_name}_weight_matrix.png"))
                plt.close(fig)
                
                # Save weighted predictions
                weighted_vis = (result['weighted_predictions'] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_weighted_predictions.png"), 
                           weighted_vis)
    
    # Generate summary statistics
    if results:
        decay_constants = [r['decay_constant'] for r in results]
        
        print(f"\nSummary Statistics:")
        print(f"Average decay constant: {np.mean(decay_constants):.4f}")
        print(f"Std decay constant: {np.std(decay_constants):.4f}")
        print(f"Min decay constant: {np.min(decay_constants):.4f}")
        print(f"Max decay constant: {np.max(decay_constants):.4f}")
        
        # Create summary visualizations
        fig, axes = plt.subplots(1, 1, figsize=(12, 10))
        
        # Decay constant distribution
        # axes[0, 0].hist(decay_constants, bins=20, alpha=0.7)
        # axes[0, 0].set_title('Decay Constant Distribution')
        # axes[0, 0].set_xlabel('Decay Constant')
        # axes[0, 0].set_ylabel('Frequency')
        
        # Average weight matrix
        avg_weight = np.mean([r['weight_matrix'] for r in results], axis=0)
        im1 = axes.imshow(avg_weight, cmap='viridis')
        axes.set_title('Average Weight Matrix')
        plt.colorbar(im1, ax=axes)
        
        # Average weighted predictions
        # avg_weighted = np.mean([r['weighted_predictions'] for r in results], axis=0)
        # im2 = axes[1, 0].imshow(avg_weighted, cmap='hot')
        # axes[1, 0].set_title('Average Weighted Predictions')
        # plt.colorbar(im2, ax=axes[1, 0])
        
        # Sample gradient
        # if results:
        #     im3 = axes[1, 1].imshow(results[0]['gradient'], cmap='gray')
        #     axes[1, 1].set_title('Sample Gradient Image')
        #     plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'summary_analysis.png'), dpi=300)
        plt.close()
        
        print(f"\nResults saved to {args.output_dir}")
        print(f"Summary analysis saved as summary_analysis.png")


if __name__ == "__main__":
    main()