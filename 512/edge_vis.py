import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy import ndimage
import argparse
from pathlib import Path

class EdgeWeightMapVisualizer:
    """
    Visualizer for edge weight maps created by EdgeWeightedLoss
    """
    def __init__(self, edge_weight=3.0, center_weight=0.5, 
                 image_edge_weight=0.3, edge_thickness=3):
        self.edge_weight = edge_weight
        self.center_weight = center_weight
        self.image_edge_weight = image_edge_weight
        self.edge_thickness = edge_thickness
        
    def create_edge_weight_map(self, masks):
        """
        Create weight map with higher weights at object edges
        Args:
            masks: Ground truth masks [B, H, W] or [H, W]
        Returns:
            weight_map: [B, H, W] or [H, W] with edge-based weights
        """
        # Handle both 2D and 3D input
        if len(masks.shape) == 2:
            # Convert 2D mask to 3D batch format
            masks = masks[np.newaxis, :, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Convert to torch tensor if numpy array
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
            
        batch_size, height, width = masks.shape
        weight_maps = torch.ones_like(masks, dtype=torch.float32)
        
        diagonal_weight_map = self._create_diagonal_distance_map(height, width)
        
        for b in range(batch_size):
            mask = masks[b].cpu().numpy().astype(np.uint8)
            weight_map = np.ones_like(mask, dtype=np.float32)
            
            # Find object edges using morphological operations
            for class_id in [1]:  # class 1 is your "tads"
                class_mask = (mask == class_id).astype(np.uint8)
                if class_mask.sum() == 0:
                    continue
                
               
                
                # Create edge map
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(class_mask, kernel, iterations=1)
                edges = class_mask - eroded
                
                # Dilate edges to create thick edge regions
                thick_edges = cv2.dilate(edges, kernel, iterations=self.edge_thickness-1)
                
                
                
                # Apply weights
                weight_map[thick_edges > 0] = self.edge_weight
                
                weight_map = weight_map * diagonal_weight_map
            
            # Reduce weights near image boundaries
            boundary_mask = np.zeros_like(mask, dtype=bool)
            boundary_width = 10
            boundary_mask[:boundary_width, :] = True
            boundary_mask[-boundary_width:, :] = True
            boundary_mask[:, :boundary_width] = True
            boundary_mask[:, -boundary_width:] = True
            
            # weight_map[boundary_mask] *= self.image_edge_weight
            
            weight_maps[b] = torch.from_numpy(weight_map).to(masks.device)
        
        # Return original shape if input was 2D
        if squeeze_output:
            return weight_maps[0]
        else:
            return weight_maps
    
    def _create_diagonal_distance_map(self, height, width):
        """
        Create a weight map based on distance from the middle diagonal
        Higher weights at the center, decreasing towards edges
        
        Args:
            height: Image height
            width: Image width
        Returns:
            diagonal_weight_map: [H, W] numpy array with diagonal distance weights
        """
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate distance from main diagonal (y = x * height/width)
        # Normalize coordinates to [0, 1]
        x_norm = x_coords / (width - 1)
        y_norm = y_coords / (height - 1)
        
        # Distance from main diagonal line
        diagonal_distance = np.abs(y_norm - x_norm)
        
        
        
        # Convert distance to weight (closer to diagonal = higher weight)
        # Max distance is sqrt(2)/2 ≈ 0.707 (corners), min is 0 (center)
        max_distance = np.sqrt(2) / 2
        
        # Create weight map: higher weight at center, lower at edges
        # You can adjust these parameters:
        diagonal_weight_strength = getattr(self, 'diagonal_weight_strength', 3.0)  # How much to emphasize diagonal
        diagonal_weight_map = 1.0 + diagonal_weight_strength * (1.0 - diagonal_distance / max_distance)
        
        return diagonal_weight_map.astype(np.float32)
    
    def create_distance_based_weights(self, mask):
        """
        Alternative: Distance-based weighting (smoother transitions)
        """
        height, width = mask.shape
        weight_map = np.ones_like(mask, dtype=np.float32)
        
        for class_id in [1]:  # Your tads class
            class_mask = (mask == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
            
            # Distance from edges (inside object)
            dist_inside = ndimage.distance_transform_edt(class_mask)
            
            # Distance from edges (outside object)
            dist_outside = ndimage.distance_transform_edt(1 - class_mask)
            
            # Create smooth weight transition
            # High weights near edges, low weights at centers
            edge_region = (dist_outside <= 5) | (dist_inside <= 5)
            center_region = dist_inside > 10
            
            # Gaussian-like weighting based on distance
            inside_weights = np.exp(-dist_inside / 5.0) * (self.edge_weight - 1) + 1
            weight_map[class_mask > 0] = inside_weights[class_mask > 0]
            
            # Boost immediate edge pixels
            immediate_edges = (dist_outside == 1) & (class_mask == 0)
            weight_map[immediate_edges] = self.edge_weight
        
        # Image boundary penalty
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        dist_from_boundary = np.minimum(
            np.minimum(x_coords, width - 1 - x_coords),
            np.minimum(y_coords, height - 1 - y_coords)
        )
        boundary_penalty = np.clip(dist_from_boundary / 20.0, self.image_edge_weight, 1.0)
        weight_map *= boundary_penalty
        
        return weight_map
    
    def load_mask(self, mask_path):
        """Load and preprocess mask"""
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure binary mask (0 and 1)
        unique_vals = np.unique(mask)
        if len(unique_vals) == 2 and 255 in unique_vals:
            mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def visualize_single_image(self, image_path, mask_path, output_path=None, 
                              show_both_methods=False, target_size=(512, 512)):
        """
        Visualize edge weights with guaranteed image sizes
        """
        # Load image and mask
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            mask = self.load_mask(mask_path)
            image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        mask = self.load_mask(mask_path)
        
        # Ensure images are exactly the target size
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        if mask.shape != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Create weight maps - now handles 2D input correctly
        morphological_weights = self.create_edge_weight_map(mask)
        distance_weights = self.create_distance_based_weights(mask)
        
        # Convert to numpy if tensor
        if isinstance(morphological_weights, torch.Tensor):
            morphological_weights = morphological_weights.cpu().numpy()
        
        # METHOD 1: Calculate exact figure size for 512x512 images
        def calculate_figure_size_for_exact_pixels(image_size, dpi=100, num_cols=3, num_rows=1):
            """Calculate figure size to get exact pixel dimensions"""
            width_pixels = image_size[1] * num_cols  # 512 * 3 = 1536
            height_pixels = image_size[0] * num_rows  # 512 * 1 = 512
            
            width_inches = width_pixels / dpi
            height_inches = height_pixels / dpi
            
            return width_inches, height_inches
        
        # Create visualization with exact sizing
        if show_both_methods:
            rows, cols = 2, 3
            fig_width, fig_height = calculate_figure_size_for_exact_pixels(target_size, dpi=100, num_cols=cols, num_rows=rows)
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=100)
        else:
            rows, cols = 1, 3
            fig_width, fig_height = calculate_figure_size_for_exact_pixels(target_size, dpi=100, num_cols=cols, num_rows=rows)
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=100)
            axes = [axes]
        
        # Remove all padding and margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Define colormap
        cmap = plt.cm.plasma
        
        # First row: Morphological method
        row = 0
        
        # Remove axes ticks and labels for exact pixel mapping
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
        
        axes[row][0].imshow(image)
        axes[row][0].axis('off')
        
        axes[row][1].imshow(mask, cmap='gray')
        axes[row][1].axis('off')
        
        im1 = axes[row][2].imshow(morphological_weights, cmap=cmap, vmin=0, vmax=self.edge_weight)
        axes[row][2].axis('off')
        
        if show_both_methods:
            # Second row: Distance-based method
            row = 1
            axes[row][0].imshow(image)
            axes[row][0].axis('off')
            
            axes[row][1].imshow(mask, cmap='gray')
            axes[row][1].axis('off')
            
            im2 = axes[row][2].imshow(distance_weights, cmap=cmap, vmin=0, vmax=self.edge_weight)
            axes[row][2].axis('off')
        
        # Save with exact pixel dimensions
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return morphological_weights, distance_weights
        
    def process_folder(self, image_folder, mask_folder, output_folder, 
                      image_ext='*.tif', mask_ext='*.tif', max_images=10):
        """
        Process a folder of images and masks
        """
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Find image and mask files
        image_files = glob.glob(os.path.join(image_folder, image_ext))
        mask_files = glob.glob(os.path.join(mask_folder, mask_ext))
        
        print(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        # Match images with masks (assumes same base filename)
        processed = 0
        for img_path in image_files[:max_images]:
            img_name = Path(img_path).stem
            
            # Look for corresponding mask
            mask_path = None
            for mask_file in mask_files:
                if Path(mask_file).stem == img_name:
                    mask_path = mask_file
                    break
            
            if mask_path is None:
                print(f"No mask found for {img_name}, skipping...")
                continue
            
            # Create output path
            output_path = os.path.join(output_folder, f"{img_name}_edge_weights.png")
            
            print(f"Processing {img_name}...")
            try:
                self.visualize_single_image(img_path, mask_path, output_path)
                processed += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        
        print(f"Successfully processed {processed} images")
    
    def create_weight_statistics(self, mask_folder, mask_ext='*.png'):
        """
        Create statistics about weight distributions
        """
        mask_files = glob.glob(os.path.join(mask_folder, mask_ext))
        
        all_weights_morph = []
        all_weights_dist = []
        
        for mask_path in mask_files:
            mask = self.load_mask(mask_path)
            
            morph_weights = self.create_edge_weight_map(mask)
            dist_weights = self.create_distance_based_weights(mask)
            
            # Convert to numpy if needed
            if isinstance(morph_weights, torch.Tensor):
                morph_weights = morph_weights.cpu().numpy()
            
            all_weights_morph.extend(morph_weights.flatten())
            all_weights_dist.extend(dist_weights.flatten())
        
        # Create histogram comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(all_weights_morph, bins=50, alpha=0.7, label='Morphological')
        ax1.set_title('Weight Distribution - Morphological Method')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(all_weights_dist, bins=50, alpha=0.7, label='Distance-based', color='orange')
        ax2.set_title('Weight Distribution - Distance-based Method')
        ax2.set_xlabel('Weight Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Morphological Method:")
        print(f"  Mean: {np.mean(all_weights_morph):.3f}")
        print(f"  Std: {np.std(all_weights_morph):.3f}")
        print(f"  Min: {np.min(all_weights_morph):.3f}")
        print(f"  Max: {np.max(all_weights_morph):.3f}")
        
        print(f"\nDistance-based Method:")
        print(f"  Mean: {np.mean(all_weights_dist):.3f}")
        print(f"  Std: {np.std(all_weights_dist):.3f}")
        print(f"  Min: {np.min(all_weights_dist):.3f}")
        print(f"  Max: {np.max(all_weights_dist):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize edge weight maps')
    parser.add_argument('--mode', choices=['single', 'folder', 'stats'], required=True,
                       help='Processing mode')
    parser.add_argument('--image_path', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--mask_path', type=str, help='Path to single mask (for single mode)')
    parser.add_argument('--image_folder', type=str, help='Folder containing images')
    parser.add_argument('--mask_folder', type=str, help='Folder containing masks')
    parser.add_argument('--output_folder', type=str, default='edge_weight_visualizations',
                       help='Output folder for visualizations')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Maximum number of images to process')
    parser.add_argument('--edge_weight', type=float, default=8.0,
                       help='Weight for edge pixels')
    parser.add_argument('--center_weight', type=float, default=0.5,
                       help='Weight for center pixels')
    parser.add_argument('--image_edge_weight', type=float, default=0.3,
                       help='Weight for image boundary pixels')
    parser.add_argument('--edge_thickness', type=int, default=3,
                       help='Thickness of edge regions')
    
    args = parser.parse_args()
    
    # Create visualizer with custom parameters
    visualizer = EdgeWeightMapVisualizer(
        edge_weight=args.edge_weight,
        center_weight=args.center_weight,
        image_edge_weight=args.image_edge_weight,
        edge_thickness=args.edge_thickness
    )
    
    if args.mode == 'single':
        if not args.image_path or not args.mask_path:
            print("Error: --image_path and --mask_path required for single mode")
            return
        
        visualizer.visualize_single_image(args.image_path, args.mask_path)
    
    elif args.mode == 'folder':
        if not args.image_folder or not args.mask_folder:
            print("Error: --image_folder and --mask_folder required for folder mode")
            return
        
        visualizer.process_folder(
            args.image_folder, 
            args.mask_folder, 
            args.output_folder,
            max_images=args.max_images
        )
    
    elif args.mode == 'stats':
        if not args.mask_folder:
            print("Error: --mask_folder required for stats mode")
            return
        
        visualizer.create_weight_statistics(args.mask_folder)


if __name__ == "__main__":
    # Example usage without command line args
    # Uncomment and modify these lines to run directly
    
    # visualizer = EdgeWeightMapVisualizer(edge_weight=8.0, center_weight=0.5, 
    #                                     image_edge_weight=0.3, edge_thickness=3)
    
    # Single image example:
    # visualizer.visualize_single_image('path/to/image.jpg', 'path/to/mask.png')
    
    # Folder processing example:
    # visualizer.process_folder('images/', 'masks/', 'output_visualizations/')
    
    # Statistics example:
    # visualizer.create_weight_statistics('masks/')

    visualizer = EdgeWeightMapVisualizer(
        edge_weight=8.0,        # Same as your training config
        center_weight=0.5,
        image_edge_weight=0.3,
        edge_thickness=3
    )

    # Process your dataset folder
    visualizer.process_folder(
        image_folder='4DNFISWHXA16_analysis/original/',
        mask_folder='4DNFISWHXA16_analysis/masks/',
        output_folder='edge_weight_visualizations/',
        max_images=15
    )
    
    # main()