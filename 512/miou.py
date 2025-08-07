#!/usr/bin/env python3
"""
Script to calculate mean Intersection over Union (mIoU) between 
prediction images and ground truth images.

Usage:
python miou_calculator.py --pred_folder /path/to/predictions --gt_folder /path/to/ground_truth

Requirements:
pip install numpy opencv-python pillow argparse
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def calculate_iou(pred_mask, gt_mask, num_classes=None):
    """
    Calculate IoU for each class between prediction and ground truth masks.
    
    Args:
        pred_mask: Predicted segmentation mask (numpy array)
        gt_mask: Ground truth segmentation mask (numpy array)
        num_classes: Number of classes (if None, will be inferred)
    
    Returns:
        iou_per_class: IoU score for each class
        valid_classes: Classes that exist in ground truth
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
    
    # Flatten arrays
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Get unique classes from ground truth
    if num_classes is None:
        classes = np.unique(gt_flat)
    else:
        classes = np.arange(num_classes)
    
    iou_per_class = []
    valid_classes = []
    
    for cls in classes:
        # Create binary masks for current class
        pred_binary = (pred_flat == cls)
        gt_binary = (gt_flat == cls)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Skip classes that don't exist in ground truth
        if union == 0:
            continue
            
        iou = intersection / union
        iou_per_class.append(iou)
        valid_classes.append(cls)
    
    return np.array(iou_per_class), valid_classes

def load_image_as_mask(image_path):
    """
    Load image and convert to mask format.
    Handles both grayscale and RGB images.
    """
    try:
        # Try loading with PIL first
        img = Image.open(image_path)
        
        # Convert to grayscale if RGB
        if img.mode == 'RGB':
            img = img.convert('L')
        elif img.mode == 'RGBA':
            img = img.convert('L')
        
        # Convert to numpy array
        mask = np.array(img)
        return mask
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_matching_files(pred_folder, gt_folder):
    """
    Get pairs of prediction and ground truth files with matching names.
    """
    pred_path = Path(pred_folder)
    gt_path = Path(gt_folder)
    
    # Get all image files from prediction folder
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    pred_files = [f for f in pred_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    matching_pairs = []
    
    for pred_file in pred_files:
        # Look for corresponding ground truth file
        gt_file = gt_path / pred_file.name
        
        if gt_file.exists():
            matching_pairs.append((pred_file, gt_file))
        else:
            print(f"Warning: No ground truth found for {pred_file.name}")
    
    return matching_pairs

def parse_file_list(file_path):
    """
    Parse a file containing image pairs and extract just the filenames.
    
    Expected format:
    original\chrX_43.663_centered.tif masks\chrX_43.663_centered.tif
    original\chr9_95.945_centered.tif masks\chr9_95.945_centered.tif
    
    Returns:
        List of filenames (without folder paths)
    """
    filenames = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by whitespace to get the two paths
                parts = line.split()
                if len(parts) >= 2:
                    # Extract filename from first path (original image)
                    # Handle both forward and backward slashes
                    original_path = parts[0]
                    filename = Path(original_path).name
                    filenames.append(filename)
                else:
                    print(f"Warning: Invalid line format: {line}")
    
    except FileNotFoundError:
        raise ValueError(f"File list not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file list {file_path}: {e}")
    
    return filenames

def get_files_from_list(pred_folder, gt_folder, filename_list):
    """
    Get pairs of prediction and ground truth files based on a list of filenames.
    """
    pred_path = Path(pred_folder)
    gt_path = Path(gt_folder)
    
    matching_pairs = []
    missing_files = []
    
    for filename in filename_list:
        pred_file = pred_path / filename
        gt_file = gt_path / filename
        
        if pred_file.exists() and gt_file.exists():
            matching_pairs.append((pred_file, gt_file))
        else:
            missing_info = []
            if not pred_file.exists():
                missing_info.append(f"pred: {pred_file}")
            if not gt_file.exists():
                missing_info.append(f"gt: {gt_file}")
            missing_files.append(f"{filename} - Missing: {', '.join(missing_info)}")
    
    if missing_files:
        print("Warning: Some files from the list were not found:")
        for missing in missing_files:
            print(f"  {missing}")
    
    return matching_pairs

def calculate_miou(pred_folder, gt_folder, num_classes=None, ignore_background=False, filename_list=None):
    """
    Calculate mean IoU between all matching image pairs in two folders.
    
    Args:
        pred_folder: Path to folder containing prediction images
        gt_folder: Path to folder containing ground truth images  
        num_classes: Number of classes (if None, will be inferred)
        ignore_background: Whether to ignore class 0 (background) in mIoU calculation
        filename_list: Optional list of specific filenames to process
    
    Returns:
        miou: Mean IoU across all images and classes
        detailed_results: Dictionary with per-image and per-class results
    """
    if filename_list is not None:
        matching_pairs = get_files_from_list(pred_folder, gt_folder, filename_list)
    else:
        matching_pairs = get_matching_files(pred_folder, gt_folder)
    
    if not matching_pairs:
        raise ValueError("No matching image pairs found!")
    
    print(f"Found {len(matching_pairs)} matching image pairs")
    
    all_ious = []
    all_classes = set()
    detailed_results = {
        'per_image': [],
        'per_class': {},
        'total_images': len(matching_pairs)
    }
    
    for pred_path, gt_path in matching_pairs:
        #print(f"Processing: {pred_path.name}")
        
        # Load images
        pred_mask = load_image_as_mask(pred_path)
        gt_mask = load_image_as_mask(gt_path)
        
        if pred_mask is None or gt_mask is None:
            print(f"Skipping {pred_path.name} due to loading error")
            continue
        
        # Calculate IoU for this image pair
        try:
            iou_per_class, valid_classes = calculate_iou(pred_mask, gt_mask, num_classes)
            
            # Store per-image results
            image_miou = np.mean(iou_per_class)
            detailed_results['per_image'].append({
                'filename': pred_path.name,
                'miou': image_miou,
                'iou_per_class': dict(zip(valid_classes, iou_per_class))
            })
            
            # Accumulate for overall statistics
            all_ious.extend(iou_per_class)
            all_classes.update(valid_classes)
            
            # Accumulate per-class IoUs
            for cls, iou in zip(valid_classes, iou_per_class):
                if cls not in detailed_results['per_class']:
                    detailed_results['per_class'][cls] = []
                detailed_results['per_class'][cls].append(iou)
            
            #print(f"  mIoU: {image_miou:.4f}")
            
        except Exception as e:
            print(f"Error processing {pred_path.name}: {e}")
            continue
    
    if not all_ious:
        raise ValueError("No valid IoU calculations were performed!")
    
    # Calculate final mIoU
    if ignore_background and 0 in all_classes:
        # Calculate mIoU excluding background class
        class_ious = []
        for cls in sorted(all_classes):
            if cls != 0:  # Skip background
                class_mean_iou = np.mean(detailed_results['per_class'][cls])
                class_ious.append(class_mean_iou)
        miou = np.mean(class_ious)
    else:
        # Calculate mean IoU across all classes
        class_ious = []
        for cls in sorted(all_classes):
            class_mean_iou = np.mean(detailed_results['per_class'][cls])
            class_ious.append(class_mean_iou)
        miou = np.mean(class_ious)
    
    # Add summary statistics to detailed results
    detailed_results['summary'] = {
        'miou': miou,
        'classes_found': sorted(list(all_classes)),
        'num_classes': len(all_classes)
    }
    
    # Add per-class mean IoUs
    for cls in sorted(all_classes):
        detailed_results['per_class'][cls] = {
            'ious': detailed_results['per_class'][cls],
            'mean_iou': np.mean(detailed_results['per_class'][cls])
        }
    
    return miou, detailed_results

def main():
    parser = argparse.ArgumentParser(description='Calculate mIoU between prediction and ground truth images')
    parser.add_argument('--pred_folder', required=True, help='Path to folder containing prediction images')
    parser.add_argument('--gt_folder', required=True, help='Path to folder containing ground truth images')
    parser.add_argument('--train_file', help='Path to file containing train image pairs (optional)')
    parser.add_argument('--val_file', help='Path to file containing validation image pairs (optional)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (auto-detect if not specified)')
    parser.add_argument('--ignore_background', action='store_true', help='Ignore background class (class 0) in mIoU calculation')
    parser.add_argument('--verbose', action='store_true', help='Print detailed per-image results')
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.pred_folder):
        raise ValueError(f"Prediction folder does not exist: {args.pred_folder}")
    if not os.path.exists(args.gt_folder):
        raise ValueError(f"Ground truth folder does not exist: {args.gt_folder}")
    
    print(f"Calculating mIoU...")
    print(f"Prediction folder: {args.pred_folder}")
    print(f"Ground truth folder: {args.gt_folder}")
    print(f"Number of classes: {'Auto-detect' if args.num_classes is None else args.num_classes}")
    print(f"Ignore background: {args.ignore_background}")
    
    # Determine what to evaluate
    evaluate_sets = []
    
    if args.train_file:
        if not os.path.exists(args.train_file):
            raise ValueError(f"Train file does not exist: {args.train_file}")
        train_filenames = parse_file_list(args.train_file)
        evaluate_sets.append(('Train', train_filenames))
        print(f"Train file: {args.train_file} ({len(train_filenames)} images)")
    
    if args.val_file:
        if not os.path.exists(args.val_file):
            raise ValueError(f"Validation file does not exist: {args.val_file}")
        val_filenames = parse_file_list(args.val_file)
        evaluate_sets.append(('Validation', val_filenames))
        print(f"Validation file: {args.val_file} ({len(val_filenames)} images)")
    
    # If no file lists provided, evaluate all images in folders
    if not evaluate_sets:
        evaluate_sets.append(('All Images', None))
        print("No file lists provided - evaluating all matching images in folders")
    
    print("-" * 60)
    
    try:
        with open("miou_results.txt", "a") as f:
            for set_name, filename_list in evaluate_sets:
                f.write(f"\n{'='*20} {set_name.upper()} SET {'='*20}")
                
                miou, detailed_results = calculate_miou(
                    args.pred_folder, 
                    args.gt_folder, 
                    args.num_classes,
                    args.ignore_background,
                    filename_list
                )
                
                f.write("\n" + "="*50)
                f.write(f"{set_name.upper()} RESULTS")
                f.write("="*50)
                f.write(f"\nOverall mIoU: {miou:.4f}")
                f.write(f"Classes found: {detailed_results['summary']['classes_found']}")
                f.write(f"Number of classes: {detailed_results['summary']['num_classes']}")
                f.write(f"Images processed: {detailed_results['total_images']}")
                
                f.write(f"\n{set_name} per-class mean IoU:")
                for cls in sorted(detailed_results['per_class'].keys()):
                    class_miou = detailed_results['per_class'][cls]['mean_iou']
                    f.write(f"  Class {cls}: {class_miou:.4f}")
                
                if args.verbose:
                    f.write(f"\n{set_name} per-image results:")
                    for result in detailed_results['per_image']:
                        f.write(f"  {result['filename']}: {result['miou']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())