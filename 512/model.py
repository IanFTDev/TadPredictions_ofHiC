import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

    

class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class EdgeAttention(nn.Module):
    """Edge-focused attention mechanism"""
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Sobel edge detection kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, x):
        # Calculate edge features using Sobel operators
        gray = torch.mean(x, dim=1, keepdim=True)  # Convert to grayscale
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # Generate attention weights
        attention = self.edge_conv(x)
        
        # Combine edge information with attention
        edge_enhanced_attention = attention * (1 + edge_magnitude)
        
        return x * edge_enhanced_attention

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv=True, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=2, use_deconv=True, use_attention=True, dropout_rate=0):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Encoder - with progressive dropout for deeper layers
        self.down1 = Down(3, 64)                              # 128x128 -> 128x128
        self.down2 = Down(64, 128)                                 # 128x128 -> 64x64
        self.down3 = Down(128, 256)           # 64x64 -> 32x32 (light dropout)
        self.down4 = Down(256, 512)                # 32x32 -> 16x16 (more dropout)
        
        
        self.bottle_neck = DoubleConv(512, 1024)                # 16x16 -> 8x8 (more dropout)
        

        # Decoder with attention (no dropout in decoder to preserve learned features)
        self.up1 = Up(1024, 512, use_deconv, use_attention)  # 8x8 -> 16x16
        self.up2 = Up(512, 256, use_deconv, use_attention)  # 16x16 -> 32x32
        self.up3 = Up(256, 128, use_deconv, use_attention)   # 32x32 -> 64x64
        self.up4 = Up(128, 64, use_deconv, use_attention)    # 64x64 -> 128x128
   
        # Output
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)


        self._initialize_weights()
    
    def forward(self, x):
        # Encoder
        x1, p1 = self.down1(x)      # 128x128x32
        x2, p2 = self.down2(p1)   # 64x64x64
        x3, p3 = self.down3(p2)   # 32x32x128
        x4, p4 = self.down4(p3)   # 16x16x256
            
        b = self.bottle_neck(p4)
        
        # Decoder with skip connections and attention
        y1 = self.up1(b, x4)  # 16x16x256
        y2 = self.up2(y1, x3)   # 32x32x128
        y3 = self.up3(y2, x2)   # 64x64x64
        y4 = self.up4(y3, x1)   # 128x128x32
        
        # Output
        logits = self.outc(y4)  # 128x128x num_classes
        return logits
    
    def _initialize_weights(self):
            """Initialize model weights using Kaiming normal initialization"""
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    try:
                        nn.init.zeros_(module.bias)
                    except AttributeError:
                        pass  # bias=False for this module

# Updated training configuration for regularization
def get_regularized_training_config():
    return {
        'batch_size': 4,
        'iters': 15000,                    # Reduced from 20000
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
        'use_attention': True,
        'attention_learning_rate_factor': 0.1,
        # Early stopping parameters
        'early_stopping_patience': 8,     # Stop if no improvement for 8 epochs
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
    def __init__(self, patience=8, min_improvement=0.002, metric='val_iou'):
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

# Example usage:
if __name__ == "__main__":
    # Test model with regularization
    config = get_regularized_training_config()
    
    # Create model with attention and dropout
    model = UNet(
        num_classes=config['num_classes'], 
        use_deconv=True, 
        use_attention=config['use_attention'],
        dropout_rate=config['dropout_rate']
    )
    
    # Test with 128x128 input
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create regularized optimizer
    optimizer = create_regularized_optimizer(model, config)
    print(f"✓ Optimizer created with {len(optimizer.param_groups)} parameter groups")
    
    # Test early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_improvement=config['min_improvement'],
        metric=config['early_stopping_metric']
    )
    print(f"✓ Early stopping: patience={early_stopping.patience}, metric={early_stopping.metric}")