


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from torch.optim.lr_scheduler import LinearLR
import cv2
from scipy import ndimage
from dataset import SegmentationDataset
from model import UNet  # Your new attention-enhanced UNet



class EdgeWeightedLoss(nn.Module):
    """
    Custom loss that weights pixels higher at object edges and lower at centers/image edges
    Now includes symmetry enforcement for pairwise predictions
    """
    def __init__(self, base_weights=None, edge_weight=3.0, center_weight=0.5, 
                 image_edge_weight=0.3, edge_thickness=3, ignore_index=255, 
                 enforce_symmetry=False):
        super().__init__()
        self.base_weights = base_weights
        self.edge_weight = edge_weight
        self.center_weight = center_weight
        self.image_edge_weight = image_edge_weight
        self.edge_thickness = edge_thickness
        self.ignore_index = ignore_index
        self.enforce_symmetry = enforce_symmetry
        
    def create_edge_weight_map(self, masks):
        """
        Create weight map with higher weights at object edges
        Args:
            masks: Ground truth masks [B, H, W]
        Returns:
            weight_map: [B, H, W] with edge-based weights
        """
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
                
                
                # weight_map = weight_map * diagonal_weight_map
            
            # Reduce weights near image boundaries
            boundary_mask = np.zeros_like(mask, dtype=bool)
            boundary_width = 10
            boundary_mask[:boundary_width, :] = True
            boundary_mask[-boundary_width:, :] = True
            boundary_mask[:, :boundary_width] = True
            boundary_mask[:, -boundary_width:] = True
            
            # weight_map[boundary_mask] *= self.image_edge_weight
            
            weight_maps[b] = torch.from_numpy(weight_map).to(masks.device)
        
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
    
    def enforce_prediction_symmetry(self, predictions):
        """
        Enforce symmetry by averaging pred[i,j] and pred[j,i]
        Args:
            predictions: [B, C, H, W] logits
        Returns:
            symmetric_predictions: [B, C, H, W] with enforced symmetry
        """
        # Create symmetric predictions by averaging (i,j) and (j,i)
        symmetric_preds = (predictions + predictions.transpose(-2, -1)) / 2.0
        return symmetric_preds
    
    def get_upper_triangle_mask(self, height, width, device):
        """
        Create mask for upper triangle (including diagonal)
        Args:
            height, width: dimensions
            device: torch device
        Returns:
            mask: [H, W] boolean mask where True indicates upper triangle
        """
        # Create indices
        i_indices = torch.arange(height, device=device).view(-1, 1)
        j_indices = torch.arange(width, device=device).view(1, -1)
        
        # Upper triangle mask (including diagonal): i <= j
        upper_triangle_mask = i_indices <= j_indices
        return upper_triangle_mask
    
    def forward(self, predictions, targets):
        """
        Forward pass with edge-weighted loss and optional symmetry enforcement
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] ground truth
        """
        batch_size, num_classes, height, width = predictions.shape
        
        # Enforce symmetry if requested
        if self.enforce_symmetry:
            # Ensure square matrices for symmetry
            if height != width:
                raise ValueError(f"Symmetry enforcement requires square matrices, got {height}x{width}")
            
            # Average symmetric positions
            symmetric_predictions = self.enforce_prediction_symmetry(predictions)
            
            # Create upper triangle mask
            upper_triangle_mask = self.get_upper_triangle_mask(height, width, predictions.device)
            
            # Apply mask to keep only upper triangle
            predictions_masked = symmetric_predictions * upper_triangle_mask.unsqueeze(0).unsqueeze(0)
            targets_masked = targets * upper_triangle_mask.unsqueeze(0)
            
            # Create weight maps for upper triangle only
            weight_maps = self.create_edge_weight_map(targets_masked)
            weight_maps = weight_maps * upper_triangle_mask.unsqueeze(0)
            
            # Use masked versions for loss calculation
            predictions_for_loss = predictions_masked
            targets_for_loss = targets_masked
            valid_mask = (targets_for_loss != self.ignore_index) & upper_triangle_mask.unsqueeze(0)
            
        else:
            # Standard behavior without symmetry enforcement
            weight_maps = self.create_edge_weight_map(targets)
            predictions_for_loss = predictions
            targets_for_loss = targets
            valid_mask = (targets_for_loss != self.ignore_index)
        
        # Base cross entropy loss
        log_probs = F.log_softmax(predictions_for_loss, dim=1)
        
        # Gather log probabilities for target classes
        targets_clamped = targets_for_loss.clamp(0, num_classes-1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate weighted loss
        loss_per_pixel = -(targets_one_hot * log_probs).sum(dim=1)  # [B, H, W]
        
        # Apply class weights if provided
        if self.base_weights is not None:
            class_weights = self.base_weights.to(predictions.device)
            weight_per_pixel = (targets_one_hot * class_weights.view(1, -1, 1, 1)).sum(dim=1)
            loss_per_pixel *= weight_per_pixel
        
        # Apply edge weights
        weighted_loss = loss_per_pixel * weight_maps
        
        # Handle ignore index and compute final loss
        if valid_mask.sum() > 0:
            return weighted_loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)


class PolynomialLR:
    def __init__(self, optimizer, max_iters, power=1.0, end_lr=0):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power = power
        self.end_lr = end_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_iter = 0
    
    def step(self):
        self.current_iter += 1
        
        # Fix: Clamp the ratio to prevent going over 1.0
        ratio = min(self.current_iter / self.max_iters, 1.0)
        
        lr = (self.base_lr - self.end_lr) * (1 - ratio) ** self.power + self.end_lr
        
        # Ensure lr is a real number
        lr = max(float(lr), self.end_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def set_current_iter(self, current_iter):
        """Set the current iteration for resuming training"""
        self.current_iter = current_iter
        # Update learning rate to match current iteration
        ratio = min(self.current_iter / self.max_iters, 1.0)
        lr = (self.base_lr - self.end_lr) * (1 - ratio) ** self.power + self.end_lr
        lr = max(float(lr), self.end_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def load_checkpoint_with_attention_upgrade(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """
    Load checkpoint and handle upgrade from non-attention to attention model
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model state dict
    model_state = checkpoint['model_state_dict']
    current_model_state = model.state_dict()
    
    # Check if we're upgrading from non-attention to attention model
    attention_keys = [k for k in current_model_state.keys() if 'attention' in k or 'edge_attention' in k]
    checkpoint_attention_keys = [k for k in model_state.keys() if 'attention' in k or 'edge_attention' in k]
    
    if len(attention_keys) > 0 and len(checkpoint_attention_keys) == 0:
        print("🔄 Upgrading from non-attention to attention model...")
        
        # Load compatible weights
        compatible_state = {}
        for key in current_model_state.keys():
            if key in model_state:
                compatible_state[key] = model_state[key]
            else:
                # Initialize new attention weights
                compatible_state[key] = current_model_state[key]
                print(f"   Initializing new parameter: {key}")
        
        model.load_state_dict(compatible_state)
        print("✓ Model upgraded successfully with attention mechanism")
    
    elif len(attention_keys) == 0 and len(checkpoint_attention_keys) > 0:
        print("🔄 Downgrading from attention to non-attention model...")
        
        # Load only non-attention weights
        compatible_state = {}
        for key in current_model_state.keys():
            if key in model_state:
                compatible_state[key] = model_state[key]
            else:
                compatible_state[key] = current_model_state[key]
                print(f"   Skipping attention parameter: {key}")
        
        model.load_state_dict(compatible_state)
        print("✓ Model downgraded successfully (attention removed)")
    
    else:
        # Standard loading (same architecture)
        model.load_state_dict(model_state)
        print("✓ Model state loaded (same architecture)")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"⚠️ Could not load optimizer state: {e}")
            print("   Starting with fresh optimizer state")
    
    # Return training state
    resume_epoch = checkpoint.get('epoch', 0)
    resume_iter = checkpoint.get('iter', 0)
    best_miou = checkpoint.get('best_miou', 0.0)
    
    print(f"✓ Resuming from epoch {resume_epoch + 1}, iteration {resume_iter}")
    print(f"✓ Best mIoU so far: {best_miou:.4f}")
    
    return resume_epoch, resume_iter, best_miou

def find_latest_checkpoint(save_dir):
    """Find the latest checkpoint in the save directory"""
    checkpoint_pattern = os.path.join(save_dir, 'checkpoint_iter_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract iteration numbers and find the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0  # Perfect score if no pixels of this class
        else:
            iou = intersection / union
        ious.append(iou.item())
    return ious

def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device, epoch, writer, log_iters, start_iter=0):
    model.train()
    running_loss = 0.0
    running_iou = [0.0, 0.0]  # For 2 classes
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
        
        
        # Calculate metrics
        running_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            batch_ious = calculate_iou(preds, masks, num_classes=2)
            for j, iou in enumerate(batch_ious):
                running_iou[j] += iou
        
        # Logging
        global_step = start_iter + i + 1
        if (i + 1) % log_iters == 0:
            avg_loss = running_loss / (i + 1)
            avg_iou = [iou / (i + 1) for iou in running_iou]
            
            writer.add_scalar('Loss/Train', avg_loss, global_step)
            writer.add_scalar('IoU/Train_Class0', avg_iou[0], global_step)
            writer.add_scalar('IoU/Train_Class1', avg_iou[1], global_step)
            writer.add_scalar('IoU/Train_Mean', np.mean(avg_iou), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'mIoU': f'{np.mean(avg_iou):.4f}',
                'C1_IoU': f'{avg_iou[1]:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    return running_loss / len(dataloader), [iou / len(dataloader) for iou in running_iou]

def validate(model, dataloader, criterion, device, epoch, writer, global_step):
    model.eval()
    running_loss = 0.0
    running_iou = [0.0, 0.0]
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            batch_ious = calculate_iou(preds, masks, num_classes=2)
            for j, iou in enumerate(batch_ious):
                running_iou[j] += iou
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = [iou / len(dataloader) for iou in running_iou]

    writer.add_scalar('Loss/Val', avg_loss, global_step)
    writer.add_scalar('IoU/Val_Class0', avg_iou[0], global_step)
    writer.add_scalar('IoU/Val_Class1', avg_iou[1], global_step)
    writer.add_scalar('IoU/Val_Mean', np.mean(avg_iou), global_step)
    
    print(f'Validation - Loss: {avg_loss:.4f}, Class0 IoU: {avg_iou[0]:.4f}, Class1 IoU: {avg_iou[1]:.4f}, mIoU: {np.mean(avg_iou):.4f}')
    
    return avg_loss, avg_iou

    
# Updated training configuration for regularization
def get_regularized_training_config():
    return {
        'batch_size': 4,
        'iters': 45000,                    # Reduced from 20000
        'num_classes': 2,
        'learning_rate': 0.0005,           # Reduced from 0.001
        'weight_decay': 1e-4,              # L2 regularization
        'dropout_rate': 0.2,               # Dropout regularization
        'save_dir': 'output',
        'log_iters': 10,
        'save_interval': 500,
        'dataset_root': '4DNFISWHXA16_analysis',
        'train_path': '4DNFISWHXA16_analysis/train.txt',
        'val_path': '4DNFISWHXA16_analysis/val.txt',
        'resume': True,
        'checkpoint_path': None,
        'use_attention': False,
        'attention_learning_rate_factor': 0.1,
        'use_edge_weighting': True,
        'edge_weight': 8.0,          # High weight for object edges
        'center_weight': 0.5,        # Low weight for object centers
        'image_edge_weight': 0.3,    # Very low weight for image boundaries
        'edge_thickness': 3, 
        # Early stopping parameters
        'early_stopping_patience': 30,     # Stop if no improvement for 8 epochs
        'early_stopping_metric': 'val_iou',
        'min_improvement': 0.002,
    }

# Updated optimizer creation with weight decay
def create_regularized_optimizer(model, config):
    """Create optimizer with proper weight decay and different rates for attention"""
    if config['use_attention']:
        attention_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            if any(attention_keyword in name for attention_keyword in ['attention', 'edge_conv', 'W_g', 'W_x', 'psi']):
                attention_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {
                'params': base_params, 
                'lr': config['learning_rate'], 
                'weight_decay': config['weight_decay']
            },
            {
                'params': attention_params, 
                'lr': config['learning_rate'] * config['attention_learning_rate_factor'], 
                'weight_decay': config['weight_decay'] * 0.5  # Less regularization for attention
            }
        ])
        
        print(f"✓ Regularized optimizer with weight decay {config['weight_decay']}:")
        print(f"  Base model: LR={config['learning_rate']:.6f}, WD={config['weight_decay']:.6f}")
        print(f"  Attention layers: LR={config['learning_rate'] * config['attention_learning_rate_factor']:.6f}, WD={config['weight_decay'] * 0.5:.6f}")
        print(f"  Attention parameters: {len(attention_params)}")
        print(f"  Base parameters: {len(base_params)}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    return optimizer

# Early stopping class (same as before)
class EarlyStopping:
    def __init__(self, patience=15, min_improvement=0.002, metric='val_iou'):
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric
        self.best_score = float('inf') if 'loss' in metric else 0.0
        self.patience_counter = 0
        self.should_stop = False
        
    def __call__(self, current_score):
        if self.metric == 'val_loss':
            improved = current_score < (self.best_score - self.min_improvement)
        else:  # IoU metrics (higher is better)
            improved = current_score > (self.best_score + self.min_improvement)
        
        if improved:
            self.best_score = current_score
            self.patience_counter = 0
            return True  # Model improved
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
            return False  # Model didn't improve


def main():
    
    config = get_regularized_training_config()

    config.update({
        'dataset_root': '4DNFISWHXA16_analysis',
        'train_path': '4DNFISWHXA16_analysis/train.txt',
        'val_path': '4DNFISWHXA16_analysis/val.txt',
        # ... any other overrides
    })
    
    # Create output directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Datasets
    train_dataset = SegmentationDataset(
        dataset_root=config['dataset_root'],
        file_path=config['train_path'],
        num_classes=config['num_classes'],
        target_size=(512, 512),
        mode='train'
    )
    
    val_dataset = SegmentationDataset(
        dataset_root=config['dataset_root'],
        file_path=config['val_path'],
        num_classes=config['num_classes'],
        target_size=(512, 512),
        mode='val'
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model with attention
    model = UNet(num_classes=config['num_classes'], use_deconv=True, use_attention=config['use_attention'], dropout_rate=config['dropout_rate'])

    model = model.to(device)
    
    optimizer = create_regularized_optimizer(model, config) 
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_improvement=config['min_improvement'],
        metric=config['early_stopping_metric']
    )
    # Loss function
    
    if config.get('use_edge_weighting'):
        print("USING EDGE WEIGHTING")
        criterion = EdgeWeightedLoss(
            base_weights=torch.tensor([1.0,1.0]),
            edge_weight=config.get('edge_weight', 3.0),
            center_weight=config.get('center_weight', 0.5),
            image_edge_weight=config.get('image_edge_weight', 0.3),
            edge_thickness=config.get('edge_thickness', 3)
        )
    else:
        print("USING NORMAL WEIGHTS")
        class_weights = torch.tensor([0.60, 0.40])
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=255)
    
    # Optimizer with different learning rates for attention layers
    if config['use_attention']:
        attention_params = []
        base_params = []
        
        for name, param in model.named_parameters():
            if 'attention' in name:
                attention_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': config['learning_rate']},
            {'params': attention_params, 'lr': config['learning_rate'] * config['attention_learning_rate_factor']}
        ])
        
        print(f"✓ Using different learning rates:")
        print(f"  Base model: {config['learning_rate']:.6f}")
        print(f"  Attention layers: {config['learning_rate'] * config['attention_learning_rate_factor']:.6f}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    lr_scheduler = PolynomialLR(
        optimizer, 
        max_iters=config['iters'], 
        power=1.0
    )


    # Initialize training state
    start_epoch = 0
    start_iter = 0
    best_miou = 0.0
    
    # Resume training if requested
    if config['resume']:
        checkpoint_path = config['checkpoint_path']
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config['save_dir'])
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, start_iter, best_miou = load_checkpoint_with_attention_upgrade(
                checkpoint_path, model, optimizer, lr_scheduler
            )
            
            # Set the learning rate scheduler to the correct iteration
            #lr_scheduler.set_current_iter(start_iter)
            print(f"✓ Learning rate scheduler resumed at iteration {start_iter}")
            print(f"✓ Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print("No checkpoint found, starting from scratch")
            config['resume'] = False
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(config['save_dir'], 'logs'))
    
    # Calculate epochs
    steps_per_epoch = len(train_loader)
    total_epochs = (config['iters'] + steps_per_epoch - 1) // steps_per_epoch
    remaining_iters = config['iters'] - start_iter
    
    print(f'\nTraining Configuration:')
    print(f'- Model: UNet with {"attention" if config["use_attention"] else "no attention"}')
    print(f'- Total epochs: {total_epochs} ({config["iters"]} total iterations)')
    print(f'- Starting from epoch {start_epoch + 1}, iteration {start_iter}')
    print(f'- Remaining iterations: {remaining_iters}')
    print(f'- Steps per epoch: {steps_per_epoch}')
    print(f'- Class weights: [1.0, 2.0] (favoring class 1)')
    print(f'- Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Training loop
    current_iter = start_iter
    
    for epoch in range(start_epoch, total_epochs):
        if current_iter >= config['iters']:
            print(f"Reached maximum iterations ({config['iters']}). Training complete!")
            break
        if early_stopping.should_stop:
            if early_stopping.should_stop:
                print(f"Early stopping triggered after {early_stopping.patience_counter} epochs without improvement")
            break
            
        print(f'\nEpoch {epoch + 1}/{total_epochs}')
        
        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler, 
            device, epoch, writer, config['log_iters'], current_iter
        )
        
        current_iter += len(train_loader)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device, epoch, writer, current_iter)
        
        val_miou = np.mean(val_iou)
        
        # Save checkpoint
        if (current_iter % config['save_interval']) < len(train_loader) or current_iter >= config['iters']:
            checkpoint = {
                'epoch': epoch,
                'iter': current_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'best_miou': best_miou,
                'use_attention': config['use_attention']  # Store attention flag
            }
            
            save_path = os.path.join(config['save_dir'], f'checkpoint_iter_{current_iter}.pth')
            torch.save(checkpoint, save_path)
            print(f'Saved checkpoint: {save_path}')
            # Early stopping check
            if config['early_stopping_metric'] == 'val_loss':
                improved = early_stopping(val_loss)
            else:  # val_iou
                improved = early_stopping(np.mean(val_iou))
            
            if improved:
                print(f"✓ Model improved! {config['early_stopping_metric']}: {early_stopping.best_score:.4f}")
            # Save best model using weighted mIoU
            if config.get('use_edge_weighting'):
                weights = [1.0, 1.0]
            else:
                weights = [0.60, 0.40]  # Same as loss weights
            weighted_miou = sum(iou * weight for iou, weight in zip(val_iou, weights)) / sum(weights)
            if weighted_miou > best_miou:
                best_miou = weighted_miou
                checkpoint['best_miou'] = best_miou
                best_path = os.path.join(config['save_dir'], 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f'🎉 New best model saved with weighted mIoU: {best_miou:.4f}')
                print(f'   Class 0 IoU: {val_iou[0]:.4f}, Class 1 IoU: {val_iou[1]:.4f}')
    
    writer.close()
    print(f'\n🏁 Training completed! Best weighted mIoU: {best_miou:.4f}')

if __name__ == '__main__':
    main()


