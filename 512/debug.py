
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegmentationDataset
from model_128 import UNet

def analyze_predictions(model, dataset_root, file_path, device, num_samples=5):
    """
    Analyze what your model is actually predicting vs ground truth
    """
    # Load dataset
    dataset = SegmentationDataset(
        dataset_root=dataset_root,
        file_path=file_path,
        num_classes=2,
        target_size=(128, 128),
        mode='val'
    )
    
    model.eval()
    
    print("PREDICTION ANALYSIS")
    print("=" * 50)
    
    all_pred_class_dist = []
    all_true_class_dist = []
    all_class0_ious = []
    all_class1_ious = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, true_mask = dataset[i]
            
            # Add batch dimension and move to device
            image_batch = image.unsqueeze(0).to(device)
            true_mask_batch = true_mask.unsqueeze(0).to(device)
            
            # Get prediction
            logits = model(image_batch)
            pred_probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(logits, dim=1)
            
            # Convert to numpy
            pred_mask_np = pred_mask.cpu().numpy().squeeze()
            true_mask_np = true_mask.numpy()
            pred_probs_np = pred_probs.cpu().numpy().squeeze()
            
            # Calculate class distributions
            total_pixels = pred_mask_np.size
            
            # Predicted distribution
            pred_class0_count = np.sum(pred_mask_np == 0)
            pred_class1_count = np.sum(pred_mask_np == 1)
            pred_class0_pct = pred_class0_count / total_pixels * 100
            pred_class1_pct = pred_class1_count / total_pixels * 100
            
            # True distribution
            true_class0_count = np.sum(true_mask_np == 0)
            true_class1_count = np.sum(true_mask_np == 1)
            true_class0_pct = true_class0_count / total_pixels * 100
            true_class1_pct = true_class1_count / total_pixels * 100
            
            # Calculate IoUs manually
            # Class 0 IoU
            class0_intersection = np.sum((pred_mask_np == 0) & (true_mask_np == 0))
            class0_union = np.sum((pred_mask_np == 0) | (true_mask_np == 0))
            class0_iou = class0_intersection / class0_union if class0_union > 0 else 0
            
            # Class 1 IoU  
            class1_intersection = np.sum((pred_mask_np == 1) & (true_mask_np == 1))
            class1_union = np.sum((pred_mask_np == 1) | (true_mask_np == 1))
            class1_iou = class1_intersection / class1_union if class1_union > 0 else 0
            
            all_pred_class_dist.append([pred_class0_pct, pred_class1_pct])
            all_true_class_dist.append([true_class0_pct, true_class1_pct])
            all_class0_ious.append(class0_iou)
            all_class1_ious.append(class1_iou)
            
            print(f"\nSample {i+1}:")
            print(f"  GROUND TRUTH: Class 0: {true_class0_pct:.1f}%, Class 1: {true_class1_pct:.1f}%")
            print(f"  PREDICTED:    Class 0: {pred_class0_pct:.1f}%, Class 1: {pred_class1_pct:.1f}%")
            print(f"  CLASS 0 IoU: {class0_iou:.3f}")
            print(f"  CLASS 1 IoU: {class1_iou:.3f}")
            
            # Check prediction confidence
            avg_class0_conf = np.mean(pred_probs_np[0][pred_mask_np == 0]) if pred_class0_count > 0 else 0
            avg_class1_conf = np.mean(pred_probs_np[1][pred_mask_np == 1]) if pred_class1_count > 0 else 0
            print(f"  AVG CONFIDENCE: Class 0: {avg_class0_conf:.3f}, Class 1: {avg_class1_conf:.3f}")
            
            # Visualize first sample
            if i == 0:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Original image
                img_np = image.permute(1, 2, 0).numpy()
                axes[0,0].imshow(img_np[:,:,0], cmap='Reds')
                axes[0,0].set_title('Original HiC')
                axes[0,0].axis('off')
                
                # Ground truth
                axes[0,1].imshow(true_mask_np, cmap='viridis', vmin=0, vmax=1)
                axes[0,1].set_title(f'Ground Truth\nC0:{true_class0_pct:.1f}% C1:{true_class1_pct:.1f}%')
                axes[0,1].axis('off')
                
                # Prediction
                axes[0,2].imshow(pred_mask_np, cmap='viridis', vmin=0, vmax=1)
                axes[0,2].set_title(f'Prediction\nC0:{pred_class0_pct:.1f}% C1:{pred_class1_pct:.1f}%')
                axes[0,2].axis('off')
                
                # Prediction confidence for Class 0
                axes[1,0].imshow(pred_probs_np[0], cmap='Blues', vmin=0, vmax=1)
                axes[1,0].set_title('Class 0 Confidence')
                axes[1,0].axis('off')
                
                # Prediction confidence for Class 1
                axes[1,1].imshow(pred_probs_np[1], cmap='Reds', vmin=0, vmax=1)
                axes[1,1].set_title('Class 1 Confidence')
                axes[1,1].axis('off')
                
                # Difference map
                difference = (pred_mask_np == true_mask_np).astype(float)
                axes[1,2].imshow(difference, cmap='RdYlGn', vmin=0, vmax=1)
                axes[1,2].set_title(f'Correct Predictions\n{np.mean(difference)*100:.1f}% accurate')
                axes[1,2].axis('off')
                
                plt.tight_layout()
                plt.show()
    
    # Summary statistics
    avg_pred_class_dist = np.mean(all_pred_class_dist, axis=0)
    avg_true_class_dist = np.mean(all_true_class_dist, axis=0)
    avg_class0_iou = np.mean(all_class0_ious)
    avg_class1_iou = np.mean(all_class1_ious)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY ACROSS {num_samples} SAMPLES:")
    print(f"{'='*50}")
    print(f"AVERAGE GROUND TRUTH: Class 0: {avg_true_class_dist[0]:.1f}%, Class 1: {avg_true_class_dist[1]:.1f}%")
    print(f"AVERAGE PREDICTIONS:  Class 0: {avg_pred_class_dist[0]:.1f}%, Class 1: {avg_pred_class_dist[1]:.1f}%")
    print(f"AVERAGE CLASS 0 IoU: {avg_class0_iou:.3f}")
    print(f"AVERAGE CLASS 1 IoU: {avg_class1_iou:.3f}")
    
    # Diagnosis
    print(f"\n{'='*50}")
    print(f"DIAGNOSIS:")
    print(f"{'='*50}")
    
    if avg_pred_class_dist[1] > 80:  # Predicting >80% as Class 1
        print("🚨 ISSUE DETECTED: Model is over-predicting Class 1!")
        print("   Your model has learned to predict mostly Class 1 (TADs)")
        
        if avg_true_class_dist[1] > 70:
            print("   This might be partially correct if TADs really cover >70% of the image")
        else:
            print("   This is problematic - model is not learning proper boundaries")
            
        print("\n💡 SOLUTIONS:")
        print("   1. Increase Class 0 weight in loss function")
        print("   2. Use Focal Loss to handle class imbalance")
        print("   3. Reduce learning rate")
        print("   4. Add more regularization")
        print("   5. Check if labels are correct")
        
    elif abs(avg_pred_class_dist[1] - avg_true_class_dist[1]) < 10:
        print("✅ GOOD: Prediction distribution matches ground truth")
    else:
        print("⚠️  Prediction distribution doesn't match ground truth")

def check_model_bias(model, device):
    """Check if model has inherent bias towards one class"""
    model.eval()
    
    # Create a neutral test image (all zeros or noise)
    test_images = [
        torch.zeros(1, 3, 128, 128),  # All black
        torch.ones(1, 3, 128, 128),   # All white  
        torch.randn(1, 3, 128, 128),  # Random noise
    ]
    
    print(f"\n{'='*50}")
    print("MODEL BIAS TEST (neutral inputs):")
    print(f"{'='*50}")
    
    with torch.no_grad():
        for i, test_img in enumerate(test_images):
            test_img = test_img.to(device)
            logits = model(test_img)
            pred_probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(logits, dim=1)
            
            pred_class0_pct = torch.sum(pred_mask == 0).item() / pred_mask.numel() * 100
            pred_class1_pct = torch.sum(pred_mask == 1).item() / pred_mask.numel() * 100
            
            test_names = ["All Black", "All White", "Random Noise"]
            print(f"{test_names[i]}: Class 0: {pred_class0_pct:.1f}%, Class 1: {pred_class1_pct:.1f}%")
            
            if pred_class1_pct > 90:
                print(f"  🚨 Model is heavily biased towards Class 1!")

# Example usage
if __name__ == "__main__":
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # You'll need to load your actual model here
    model = UNet(num_classes=2, use_deconv=True, use_attention=False, dropout_rate=0.2)
    checkpoint = torch.load('output/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    analyze_predictions(
        model=model,
        dataset_root='4DNFISWHXA16_analysis',
        file_path='4DNFISWHXA16_analysis/train.txt',
        device=device,
        num_samples=3
    )
    
    check_model_bias(model, device)
    
    print("To use this analyzer:")
    print("1. Load your trained model")
    print("2. Call analyze_predictions() with your model")
    print("3. Check the visualization and statistics")