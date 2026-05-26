import glob
import os
import re

def main():
    print("======================================================================")
    print(" ⚡ RSNA Pneumonia Detection - Active Training Progress Monitor")
    print("======================================================================\n")
    
    # Path to the task log directory in App Data
    tasks_dir = r"C:\Users\SKV\.gemini\antigravity\brain\68baf366-3618-40e5-acaf-afe64f0a885e\.system_generated\tasks"
    
    if not os.path.exists(tasks_dir):
        print("❌ Error: Could not locate active task log directory.")
        return
        
    # Get all log files in the tasks directory
    log_files = glob.glob(os.path.join(tasks_dir, "*.log"))
    if not log_files:
        print("❌ No active training logs found.")
        return
        
    # Find the active log file (the one containing the RSNA training config)
    active_log = None
    for log_path in sorted(log_files, key=os.path.getmtime, reverse=True):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                head = f.read(1000)
                if "RSNA PNEUMONIA DETECTION - TRAINING" in head:
                    active_log = log_path
                    break
        except Exception:
            continue
            
    if not active_log:
        print("ℹ️ No active RSNA training session log found. Training might be inactive.")
        return
        
    # Read the active log
    try:
        with open(active_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        if not lines:
            print("⏳ Training log is empty. Starting up...")
            return
            
        # Parse total epochs configuration
        epochs_target = 10
        for line in lines[:50]:
            if "Target Epochs:" in line:
                try:
                    epochs_target = int(line.split("Target Epochs:")[-1].strip())
                except Exception:
                    pass
                    
        # Find the last progress bar line (from tqdm)
        progress_line = None
        for line in reversed(lines):
            if "Epoch" in line and "%" in line and "/" in line:
                progress_line = line
                break
                
        if not progress_line:
            # Check if there is a general status print
            for line in reversed(lines):
                if line.strip():
                    progress_line = line
                    break
                    
        if progress_line:
            # Clean up escape characters from tqdm formatting
            clean_line = re.sub(r'\x1b\[[0-9;]*[mK]', '', progress_line)
            clean_line = clean_line.replace('\r', '').strip()
            
            print("🟢 Current Training Status:")
            print("-" * 80)
            print(clean_line)
            print("-" * 80)
            
            # Format: Epoch 1 [Train]:  32%|###1      | 3366/10674 [3:02:56<2:01:20,  1.00it/s...]
            epoch_match = re.search(r'Epoch (\d+)', clean_line)
            percent_match = re.search(r'(\d+)%', clean_line)
            
            if epoch_match and percent_match:
                epoch = int(epoch_match.group(1))
                percent = int(percent_match.group(1))
                total_progress = ((epoch - 1) * 100 + percent) / epochs_target
                
                # Check for other stats in progress string
                stats = ""
                if "loss=" in clean_line:
                    loss_val = clean_line.split("loss=")[-1].split(",")[0]
                    stats += f"  - Current Loss: {loss_val}\n"
                if "gpu=" in clean_line:
                    gpu_mem = clean_line.split("gpu=")[-1].split("]")[0]
                    stats += f"  - GPU Memory: {gpu_mem}\n"
                
                print(f"📊 Progress Analytics:")
                print(f"  - Active Epoch: {epoch} of {epochs_target}")
                print(f"  - Current Epoch Completion: {percent}%")
                print(f"  - Total Pipeline Progress: {total_progress:.2f}% completed")
                if stats:
                    print(stats)
            else:
                # If we only have normal printed messages showing completed epochs
                completed_epoch = 0
                for line in reversed(lines):
                    if "Checkpoints updated" in line or "Saved best model" in line:
                        completed_match = re.search(r'checkpoint_epoch_(\d+)', line)
                        if completed_match:
                            completed_epoch = int(completed_match.group(1))
                            break
                if completed_epoch > 0:
                    print(f"📊 Completed Epochs: {completed_epoch} of {epochs_target}")
                    print(f"  - Total Pipeline Progress: {(completed_epoch / epochs_target) * 100:.2f}% completed")
        else:
            print("⏳ Initializing GPU training run. Please check back in a few seconds...")
            
    except Exception as e:
        print(f"❌ Error reading active log: {e}")

if __name__ == '__main__':
    main()
