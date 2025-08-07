import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from pathlib import Path
import glob
from tqdm import tqdm

def load_image(image_path):
    """Load image and convert to numpy array"""
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            # Use cv2 for TIFF files to handle different formats
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Try with PIL if cv2 fails
                img = np.array(Image.open(image_path))
        else:
            # Use PIL for other formats
            img = np.array(Image.open(image_path))
        
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def normalize_image(img, method='minmax'):
    """Normalize image for display"""
    if img is None:
        return None
    
    img = img.astype(np.float32)
    
    if method == 'minmax':
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
    elif method == 'percentile':
        # Use 1st and 99th percentiles for better contrast
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
    
    return img

def create_overlay(hic_matrix, mask, alpha=0.5, mask_color=[1, 0, 0]):
    """Create overlay of mask on HiC matrix"""
    if hic_matrix is None or mask is None:
        return None
    
    # Normalize HiC matrix
    hic_norm = normalize_image(hic_matrix)
    
    # Convert to RGB if grayscale
    if len(hic_norm.shape) == 2:
        hic_rgb = np.stack([hic_norm] * 3, axis=-1)
    else:
        hic_rgb = hic_norm
    
    # Normalize mask
    mask_norm = normalize_image(mask)
    
    # Create colored mask
    mask_colored = np.zeros_like(hic_rgb)
    if len(mask_norm.shape) == 2:
        # Binary mask
        mask_binary = mask_norm > 0.5
        for i in range(3):
            mask_colored[:, :, i] = mask_binary * mask_color[i]
    else:
        # Multi-class mask - use different colors for each class
        for class_id in np.unique(mask_norm):
            if class_id == 0:  # Background
                continue
            class_mask = mask_norm == class_id
            color_idx = int(class_id) % len(mask_color)
            mask_colored[class_mask] = mask_color[color_idx]
    
    # Create overlay
    overlay = hic_rgb * (1 - alpha) + mask_colored * alpha
    
    return np.clip(overlay, 0, 1)

def find_matching_files(hic_dir, pred_dir, gt_dir, extensions=['.tif', '.tiff', '.png', '.jpg', '.jpeg']):
    """Find matching files across all three directories"""
    matching_files = []
    
    # Get all files from HiC directory
    hic_files = []
    for ext in extensions:
        hic_files.extend(glob.glob(os.path.join(hic_dir, f"*{ext}")))
        hic_files.extend(glob.glob(os.path.join(hic_dir, f"*{ext.upper()}")))
    
    for hic_path in hic_files:
        base_name = os.path.splitext(os.path.basename(hic_path))[0]
        
        # Find corresponding prediction file
        pred_path = None
        for ext in extensions:
            pred_candidate = os.path.join(pred_dir, f"{base_name}{ext}")
            if os.path.exists(pred_candidate):
                pred_path = pred_candidate
                break
        
        # Find corresponding ground truth file
        gt_path = None
        for ext in extensions:
            gt_candidate = os.path.join(gt_dir, f"{base_name}{ext}")
            if os.path.exists(gt_candidate):
                gt_path = gt_candidate
                break
        
        if pred_path and gt_path:
            matching_files.append({
                'base_name': base_name,
                'hic': hic_path,
                'pred': pred_path,
                'gt': gt_path
            })
        else:
            print(f"⚠️ Missing files for {base_name}:")
            if not pred_path:
                print(f"   - Prediction file not found")
            if not gt_path:
                print(f"   - Ground truth file not found")
    
    return matching_files
def create_five_panel_visualization(hic_path, pred_path, gt_path, output_path, 
                                  figsize=None, dpi=300, alpha=0.5):
    """Create 5-panel visualization"""
    
    # Load images
    hic_matrix = load_image(hic_path)
    pred_mask = load_image(pred_path)
    gt_mask = load_image(gt_path)

    top_margin = 0.60 
    wspace_margin = 0.4
    
    if hic_matrix is None or pred_mask is None or gt_mask is None:
        print(f"❌ Failed to load images for {os.path.basename(hic_path)}")
        return False
    
    if figsize is None:
        img_height = 512
        img_width = 512
        # 5 panels side by side, maintain aspect ratio
        panel_width = img_width / dpi  # Convert pixels to inches
        panel_height = img_height / dpi
        figsize = (panel_width * 5 + (panel_width * wspace_margin * 4), 
                  panel_height + (panel_height * top_margin))
    
    # Create overlays
    pred_overlay = create_overlay(hic_matrix, pred_mask, alpha=alpha, mask_color=[1, 0, 0])
    gt_overlay = create_overlay(hic_matrix, gt_mask, alpha=alpha, mask_color=[0, 1, 0])
    
    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=figsize)
    
    # Panel 1: HiC Matrix
    hic_norm = normalize_image(hic_matrix, method='percentile')
    axes[0].imshow(hic_norm, cmap='hot', aspect='equal')
    axes[0].set_title('HiC Matrix', fontsize=7)
    axes[0].axis('off')
    
    # Panel 2: Predicted Mask
    pred_norm = normalize_image(pred_mask)
    axes[1].imshow(pred_norm, cmap='hot', aspect='equal')
    axes[1].set_title('Predicted Mask', fontsize=7)
    axes[1].axis('off')
    
    # Panel 3: Ground Truth Mask
    gt_norm = normalize_image(gt_mask)
    axes[2].imshow(gt_norm, cmap='hot', aspect='equal')
    axes[2].set_title('Ground Truth Mask', fontsize=7)
    axes[2].axis('off')
    
    # Panel 4: Predicted Overlay
    if pred_overlay is not None:
        axes[3].imshow(pred_overlay, aspect='equal')
        axes[3].set_title('Predicted Overlay', fontsize=7)
    else:
        axes[3].text(0.5, 0.5, 'Failed to create\noverlay', ha='center', va='center')
        axes[3].set_title('Predicted Overlay', fontsize=7)
    axes[3].axis('off')
    
    # Panel 5: Ground Truth Overlay
    if gt_overlay is not None:
        axes[4].imshow(gt_overlay, aspect='equal')
        axes[4].set_title('Ground Truth Overlay', fontsize=7)
    else:
        axes[4].text(0.5, 0.5, 'Failed to create\noverlay', ha='center', va='center')
        axes[4].set_title('Ground Truth Overlay', fontsize=7)
    axes[4].axis('off')
    
    # Add main title
    base_name = os.path.splitext(os.path.basename(hic_path))[0]
    fig.suptitle(f'{base_name}', fontsize=8, y=0.95)
    
    # REMOVE plt.tight_layout() - it conflicts with subplots_adjust
    # plt.tight_layout()  # Comment this out!
    
    # Adjust layout - convert your margins to subplot coordinates
    plt.subplots_adjust(
        top=1 - (top_margin / (1 + top_margin)),  # Convert to subplot coordinate
        wspace=wspace_margin,
        left=0.05,
        right=0.95,
        bottom=0.05
    )
    
    # Save figure - remove bbox_inches='tight' to preserve your calculated size
    plt.savefig(output_path, dpi=dpi, facecolor='white')  # Removed bbox_inches='tight'
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create 5-panel visualizations for HiC matrices and masks')
    parser.add_argument('--hic_dir', type=str, required=True,
                       help='Directory containing HiC matrix images')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted mask images')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth mask images')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualization images')
    parser.add_argument('--figsize', type=int, nargs=2, default=None,
                       help='Figure size (width height) in inches')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output images')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha value for overlay transparency (0-1)')
    parser.add_argument('--extensions', type=str, nargs='+', 
                       default=['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
                       help='File extensions to search for')
    
    args = parser.parse_args()
    
    # Validate input directories
    for dir_path, dir_name in [(args.hic_dir, 'HiC'), (args.pred_dir, 'Predicted'), (args.gt_dir, 'Ground Truth')]:
        if not os.path.exists(dir_path):
            print(f"❌ {dir_name} directory not found: {dir_path}")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find matching files
    print("🔍 Finding matching files...")
    matching_files = find_matching_files(args.hic_dir, args.pred_dir, args.gt_dir, args.extensions)
    
    if not matching_files:
        print("❌ No matching files found across all three directories")
        print("\nPlease check that:")
        print("1. All directories exist and contain files")
        print("2. Files have the same base names across directories")
        print("3. File extensions match the supported formats")
        return
    
    print(f"✅ Found {len(matching_files)} matching file sets")
    
    # Process each file set
    successful = 0
    failed = 0
    
    print("\n🎨 Creating visualizations...")
    for file_info in tqdm(matching_files, desc="Processing"):
        base_name = file_info['base_name']
        output_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
        
        success = create_five_panel_visualization(
            file_info['hic'], file_info['pred'], file_info['gt'],
            output_path, figsize=None, dpi=args.dpi, alpha=args.alpha
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n✅ Processing complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output directory: {args.output_dir}")
    
    if successful > 0:
        print("\n📊 Visualization panels (left to right):")
        print("   1. HiC Matrix")
        print("   2. Predicted Mask") 
        print("   3. Ground Truth Mask")
        print("   4. Predicted Overlay (red on HiC)")
        print("   5. Ground Truth Overlay (green on HiC)")

if __name__ == "__main__":
    main()