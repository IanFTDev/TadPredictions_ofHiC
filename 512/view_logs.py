import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def debug_tensorboard_logs(logdir="output/logs"):
    """Debug TensorBoard logs to identify issues"""
    print("=== DEBUGGING TENSORBOARD LOGS ===")
    
    # Check if directory exists
    if not os.path.exists(logdir):
        print(f"❌ Log directory '{logdir}' does not exist!")
        return False
    
    print(f"✅ Log directory exists: {logdir}")
    
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                full_path = os.path.join(root, file)
                event_files.append(full_path)
                file_size = os.path.getsize(full_path)
                mod_time = os.path.getmtime(full_path)
                print(f"✅ Found event file: {file} (Size: {file_size} bytes, Modified: {mod_time})")
    
    if not event_files:
        print("❌ No TensorBoard event files found!")
        print("   Make sure your training script is actually writing to TensorBoard")
        return False
    
    # Sort by modification time to show chronological order
    event_files.sort(key=lambda x: os.path.getmtime(x))
    print(f"\n📅 Event files in chronological order:")
    for i, file in enumerate(event_files):
        print(f"   {i+1}. {os.path.basename(file)}")
    
    return True

def load_all_event_files(logdir):
    """Load and merge data from all event files in chronological order"""
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    # Sort by modification time (chronological order)
    event_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Dictionary to store all metrics
    all_metrics = defaultdict(list)
    
    for event_file in event_files:
        print(f"📖 Loading: {os.path.basename(event_file)}")
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    all_metrics[tag].append({
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time
                    })
        
        except Exception as e:
            print(f"❌ Error loading {event_file}: {e}")
            continue
    
    # Sort each metric by step to ensure proper chronological order
    for tag in all_metrics:
        all_metrics[tag].sort(key=lambda x: x['step'])
        
        # Remove duplicates (same step) - keep the latest wall_time
        unique_events = {}
        for event in all_metrics[tag]:
            step = event['step']
            if step not in unique_events or event['wall_time'] > unique_events[step]['wall_time']:
                unique_events[step] = event
        
        all_metrics[tag] = list(unique_events.values())
        all_metrics[tag].sort(key=lambda x: x['step'])
    
    return all_metrics

def plot_training_logs_complete_history(logdir="output/logs"):
    """Plot complete training history from all event files"""
    
    if not debug_tensorboard_logs(logdir):
        return
    
    print("\n=== LOADING COMPLETE TRAINING HISTORY ===")
    
    # Load all metrics from all event files
    all_metrics = load_all_event_files(logdir)
    
    if not all_metrics:
        print("❌ No metrics found in any event files!")
        return
    
    print(f"✅ Loaded complete history for {len(all_metrics)} metrics")
    for tag, events in all_metrics.items():
        print(f"   {tag}: {len(events)} data points (steps {events[0]['step']} to {events[-1]['step']})")
    
    # Create dynamic subplot layout based on available metrics
    available_plots = []
    
    # Check what metrics we actually have
    loss_metrics = [tag for tag in all_metrics.keys() if 'Loss' in tag]
    iou_metrics = [tag for tag in all_metrics.keys() if 'IoU' in tag and 'Mean' in tag]
    lr_metrics = [tag for tag in all_metrics.keys() if 'Learning_Rate' in tag or 'LR' in tag]
    class_iou_metrics = [tag for tag in all_metrics.keys() if 'IoU' in tag and 'Class' in tag]
    
    subplot_count = 0
    if loss_metrics: subplot_count += 1
    if iou_metrics: subplot_count += 1
    if lr_metrics: subplot_count += 1
    if class_iou_metrics: subplot_count += 1
    
    if subplot_count == 0:
        print("❌ No recognized metrics found to plot!")
        return
    
    # Create subplots
    rows = (subplot_count + 1) // 2
    cols = 2 if subplot_count > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if subplot_count == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Complete Training Progress History', fontsize=16)
    
    plot_idx = 0
    
    # Plot Loss
    if loss_metrics:
        ax = axes[plot_idx]
        colors = ['blue', 'red', 'green', 'orange']
        for i, metric in enumerate(loss_metrics):
            events = all_metrics[metric]
            if events:
                steps = [e['step'] for e in events]
                values = [e['value'] for e in events]
                ax.plot(steps, values, color=colors[i % len(colors)], 
                       label=metric.replace('Loss/', ''), linewidth=1.5)
        
        ax.set_title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot Mean IoU
    if iou_metrics:
        ax = axes[plot_idx]
        colors = ['green', 'orange', 'purple', 'brown']
        for i, metric in enumerate(iou_metrics):
            events = all_metrics[metric]
            if events:
                steps = [e['step'] for e in events]
                values = [e['value'] for e in events]
                ax.plot(steps, values, color=colors[i % len(colors)], 
                       label=metric.replace('IoU/', '').replace('_', ' '), linewidth=1.5)
        
        ax.set_title('Mean IoU')
        ax.set_xlabel('Step')
        ax.set_ylabel('IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot Learning Rate
    if lr_metrics:
        ax = axes[plot_idx]
        for metric in lr_metrics:
            events = all_metrics[metric]
            if events:
                steps = [e['step'] for e in events]
                values = [e['value'] for e in events]
                ax.plot(steps, values, 'm-', label='Learning Rate', linewidth=1.5)
        
        ax.set_title('Learning Rate')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        plot_idx += 1
    
    # Plot Class-specific IoU
    if class_iou_metrics:
        ax = axes[plot_idx]
        colors = ['blue', 'red', 'purple', 'orange', 'magenta']
        for i, metric in enumerate(class_iou_metrics):
            events = all_metrics[metric]
            if events:
                steps = [e['step'] for e in events]
                values = [e['value'] for e in events]
                label = metric.replace('IoU/', '').replace('_', ' ')
                ax.plot(steps, values, color=colors[i % len(colors)], 
                       label=label, linewidth=1.5)
        
        ax.set_title('Class-specific IoU')
        ax.set_xlabel('Step')
        ax.set_ylabel('IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add resume points as vertical lines
    if len(loss_metrics) > 0:
        # Find potential resume points (gaps in steps)
        metric = loss_metrics[0]
        events = all_metrics[metric]
        steps = [e['step'] for e in events]
        
        resume_points = []
        for i in range(1, len(steps)):
            if steps[i] - steps[i-1] > 100:  # Large gap indicates resume
                resume_points.append(steps[i])
        
        # Add vertical lines for resume points
        for ax_idx in range(plot_idx):
            for resume_step in resume_points:
                axes[ax_idx].axvline(x=resume_step, color='red', linestyle='--', 
                                   alpha=0.7, linewidth=1, label='Resume' if ax_idx == 0 else "")
        
        if resume_points and plot_idx > 0:
            axes[0].legend()
    
    plt.tight_layout()
    plt.savefig('complete_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Complete training progress plot saved as 'complete_training_progress.png'")
    
    # Print summary statistics
    print(f"\n📊 TRAINING SUMMARY:")
    for tag, events in all_metrics.items():
        if events:
            values = [e['value'] for e in events]
            print(f"   {tag}:")
            print(f"      Total data points: {len(events)}")
            print(f"      Steps: {events[0]['step']} → {events[-1]['step']}")
            print(f"      Min: {min(values):.4f}, Max: {max(values):.4f}, Final: {values[-1]:.4f}")

def check_training_script_issues():
    """Check for common issues in training script"""
    print("\n=== COMMON TRAINING SCRIPT ISSUES ===")
    print("Check these potential issues:")
    print("1. ❓ Is the SummaryWriter properly initialized?")
    print("   writer = SummaryWriter('output/logs')")
    print("2. ❓ Are you calling writer.close() at the end?")
    print("3. ❓ Are the scalar names consistent? Check for typos.")
    print("4. ❓ Is the training actually running? Check for early exits.")
    print("5. ❓ Are you flushing the writer periodically?")
    print("   Try adding: writer.flush() after logging")
    print("6. ❓ For resumed training, make sure global_step continues from checkpoint")

if __name__ == "__main__":
    print("🔍 Enhanced TensorBoard Log Analyzer - Complete History")
    print("=====================================================")
    
    # Try different common log directories
    possible_logdirs = [
        "output/logs",
        "logs", 
        "runs",
        "tensorboard_logs",
        "./output/logs"
    ]
    
    found_logs = False
    for logdir in possible_logdirs:
        if os.path.exists(logdir):
            print(f"🎯 Trying log directory: {logdir}")
            try:
                plot_training_logs_complete_history(logdir)
                found_logs = True
                break
            except Exception as e:
                print(f"❌ Failed to process {logdir}: {e}")
                continue
    
    if not found_logs:
        print("❌ No valid TensorBoard logs found in common directories")
        check_training_script_issues()
        
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Run your training script first")
        print("2. Check that 'output/logs' directory exists and has .tfevents files")
        print("3. Make sure TensorBoard is installed: pip install tensorboard")
        print("4. Try running: tensorboard --logdir=output/logs")
        print("5. For complete history, ensure all event files are in the same directory")