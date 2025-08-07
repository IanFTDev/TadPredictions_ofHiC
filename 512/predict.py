#!/usr/bin/env python3
"""
Prediction script for UNet segmentation models.
Supports both attention-enabled and regular UNet models.

Usage:
python predict.py --model_path output/best_model.pth --input_file test_files.txt --output_dir predictions
python predict.py --model_path output/best_model.pth --images image1.tif image2.tif --output_dir predictions
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Import your model
from model import UNet

def load_model(model_path, device, num_classes=2):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if model uses attention
    use_attention = checkpoint.get('use_attention', False)
    
    # Create model
    model = UNet(num_classes=num_classes, use_deconv=True, use_attention=use_attention)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Attention enabled: {use_attention}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if 'best_miou' in checkpoint:
        print(f"✓ Model best mIoU: {checkpoint['best_miou']:.4f}")
    
    return model

def load_image(image_path, target_size=(512, 512)):
    """Load and preprocess image"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        # Shape: (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict_single_image(model, image_tensor, device):
    """Make prediction on a single image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Determine square matrix dimensions from flattened size
        height = probabilities.shape[2]  # 512

        width = height
        

        #Make symmetric for each class
        symmetric_prob= torch.zeros_like(probabilities)
        for c in range(probabilities.shape[1]):  # for each class
            matrix_c = probabilities[0, c, :, :]
            
            # Average (i,j) with (j,i) - only works if height == width
            symmetric_matrix = (matrix_c + matrix_c.T) / 2
            symmetric_prob[0, c, :, :] = symmetric_matrix
        
        
        #probabilities = symmetric_matrix.view(1, probabilities.shape[1], height, width)
        
        predicted = torch.argmax(symmetric_prob, dim=1)
        
       
        
    # with torch.no_grad():
    #     outputs = model(image_tensor)
    #     # Apply softmax to get probabilities
    #     probabilities = F.softmax(outputs, dim=1)
        
    #     # Get predicted class
    #     predicted = torch.argmax(outputs, dim=1)
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

def save_prediction(prediction, probabilities, output_path, original_image_path=None):
    """Save prediction results"""
    # Remove batch dimension
    pred_mask = prediction[0]  # Shape: (H, W)
    
    
    # Save prediction mask as PNG
    pred_image = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
    pred_image.save(output_path)
    
    
    return {
        'prediction_path': output_path,

    }

def parse_input_file(file_path, dataset_root=None):
    """Parse input file containing image and mask pairs"""
    image_paths = []
    
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Split by spaces and filter out mask paths
    all_paths = content.split()
    
    for path in all_paths:
        if path.startswith('images') or path.startswith('original')  or path.startswith('differences')  and path.endswith('.tif'):
            # Handle both forward slashes and backslashes
            normalized_path = path.replace('\\', '/')
            if dataset_root:
                # Join with dataset root
                full_path = os.path.join(dataset_root, normalized_path)
                image_paths.append(full_path)
            else:
                image_paths.append(normalized_path)
    
    return image_paths

def calculate_iou(pred, target, num_classes=2):
    """Calculate IoU metrics"""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(float(iou))
    
    return ious

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained UNet model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_file', type=str,
                       help='Text file containing image paths')
    parser.add_argument('--images', nargs='+',
                       help='Individual image paths')
    parser.add_argument('--dataset_root', type=str,
                       help='Root directory of dataset (prepended to paths in input_file)')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for predictions')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate against ground truth masks (requires masks)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_file and not args.images:
        print("Error: Either --input_file or --images must be provided")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device, args.num_classes)
    
    # Get image paths
    if args.input_file:
        image_paths = parse_input_file(args.input_file, args.dataset_root)
        if args.dataset_root:
            print(f"Found {len(image_paths)} images in {args.input_file} with dataset root: {args.dataset_root}")
        else:
            print(f"Found {len(image_paths)} images in {args.input_file}")
    else:
        image_paths = args.images
        if args.dataset_root:
            # Prepend dataset root to individual images if they're relative paths
            image_paths = [os.path.join(args.dataset_root, img) if not os.path.isabs(img) else img 
                          for img in image_paths]
        print(f"Processing {len(image_paths)} images from command line")
    
    # Results storage
    results = {
        'model_path': args.model_path,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'predictions': []
    }
    
    evaluation_metrics = []
    
    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Making predictions")):
        # Load image
        image_tensor = load_image(image_path)
        
        if image_tensor is None:
            print(f"Skipping {image_path} due to loading error")
            continue
        
        # Make prediction
        prediction, probabilities = predict_single_image(model, image_tensor, device)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.tif")
        
        # Save results
        pred_info = save_prediction(prediction, probabilities, output_path, image_path)
        pred_info['input_image'] = image_path
        pred_info['image_index'] = i
        
        results['predictions'].append(pred_info)
        
        # Evaluate if requested and mask exists
        if args.evaluate:
            if args.dataset_root and not os.path.isabs(image_path):
                # Handle relative paths with dataset root
                relative_image_path = os.path.relpath(image_path, args.dataset_root)
                mask_path = os.path.join(args.dataset_root, relative_image_path.replace('images/', 'masks/'))
            else:
                # Handle absolute paths or no dataset root
                mask_path = image_path.replace('images/', 'masks/')
            
            if os.path.exists(mask_path):
                # Load ground truth mask
                gt_mask = Image.open(mask_path)
                if gt_mask.mode != 'L':
                    gt_mask = gt_mask.convert('L')
                gt_mask = gt_mask.resize((512, 512), Image.NEAREST)
                gt_array = np.array(gt_mask)
                
                # Normalize ground truth (assuming 0-255 range)
                gt_array = (gt_array > 127).astype(np.uint8)
                
                # Calculate IoU
                ious = calculate_iou(prediction[0], gt_array, args.num_classes)
                
                eval_result = {
                    'image': image_path,
                    'mask': mask_path,
                    'class_0_iou': ious[0],
                    'class_1_iou': ious[1],
                    'mean_iou': np.mean(ious)
                }
                evaluation_metrics.append(eval_result)
                pred_info['evaluation'] = eval_result
    
    # Save results summary
    results_path = os.path.join(args.output_dir, 'prediction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n✅ Prediction completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Processed {len(results['predictions'])} images")
    
    if evaluation_metrics:
        print(f"\n📈 Evaluation Results:")
        class_0_ious = [m['class_0_iou'] for m in evaluation_metrics]
        class_1_ious = [m['class_1_iou'] for m in evaluation_metrics]
        mean_ious = [m['mean_iou'] for m in evaluation_metrics]
        
        print(f"   Class 0 IoU: {np.mean(class_0_ious):.4f} ± {np.std(class_0_ious):.4f}")
        print(f"   Class 1 IoU: {np.mean(class_1_ious):.4f} ± {np.std(class_1_ious):.4f}")
        print(f"   Mean IoU: {np.mean(mean_ious):.4f} ± {np.std(mean_ious):.4f}")
    
    # Print file locations
    print(f"\n📄 Generated files:")
    print(f"   - Prediction masks: {args.output_dir}/*_prediction.png")
    print(f"   - Visualizations: {args.output_dir}/*_visualization.png")
    print(f"   - Probabilities: {args.output_dir}/*_probabilities.npy")
    print(f"   - Results summary: {results_path}")

if __name__ == '__main__':
    main()