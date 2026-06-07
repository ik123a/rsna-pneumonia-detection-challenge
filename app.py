import os
import io
import time
import json
import base64
import queue
import threading
import asyncio
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse

import config
import model as detection_model
from data_preparation import prepare_data

# Initialize FastAPI app
app = FastAPI(title="RSNA Pneumonia Detection Clinical Dashboard")

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_loaded_model = None
_loaded_model_device = None
model_lock = threading.Lock()

# Load dataset annotations globally for fast lookups
csv_path = config.TRAIN_LABELS
if os.path.exists(csv_path):
    print(f"Loading dataset labels from {csv_path}...")
    df_labels = pd.read_csv(csv_path)
    # Cache patient IDs
    available_patients = df_labels['patientId'].unique().tolist()
else:
    print(f"Warning: Labels file not found at {csv_path}")
    df_labels = pd.DataFrame()
    available_patients = []

def get_gpu_info():
    """Get system diagnostics regarding GPU support."""
    info = {"available": torch.cuda.is_available(), "name": "CPU", "count": 0, "vram_total": 0.0, "vram_allocated": 0.0}
    if info["available"]:
        info["name"] = torch.cuda.get_device_name(0)
        info["count"] = torch.cuda.device_count()
        info["vram_total"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        info["vram_allocated"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
    return info

def get_inference_model():
    """Thread-safe model loading / retrieval."""
    global _loaded_model, _loaded_model_device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with model_lock:
        if _loaded_model is not None and _loaded_model_device == device:
            return _loaded_model, device
            
        checkpoint_path = config.MODEL_SAVE_PATH
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Creating model with default weights.")
            model = detection_model.get_model(num_classes=2, pretrained=True, model_type='fasterrcnn')
        else:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = detection_model.get_model(num_classes=2, pretrained=False, model_type='fasterrcnn')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Clean module prefix from data parallel wrapping
            state_dict = checkpoint['model_state_dict']
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v
            model.load_state_dict(clean_state_dict)
            
        model.to(device)
        model.eval()
        _loaded_model = model
        _loaded_model_device = device
        return model, device

def extract_dicom_meta(dcm: pydicom.dataset.FileDataset) -> Dict[str, str]:
    """Helper to extract clinical parameters from DICOM tags."""
    meta = {}
    attributes = {
        'PatientID': 'Patient ID',
        'PatientSex': 'Patient Sex',
        'PatientAge': 'Patient Age',
        'ViewPosition': 'View Position',
        'StudyInstanceUID': 'Study UID',
        'SeriesInstanceUID': 'Series UID',
        'Modality': 'Modality',
        'BodyPartExamined': 'Body Part'
    }
    for attr, display in attributes.items():
        if hasattr(dcm, attr):
            meta[display] = str(getattr(dcm, attr))
        else:
            meta[display] = "N/A"
    return meta

def process_dicom(dcm_data: bytes):
    """Parses raw DICOM bytes into a standardized uint8 RGB image."""
    dcm = pydicom.dcmread(io.BytesIO(dcm_data))
    image = dcm.pixel_array
    
    # Normalize to 0-255
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0
        image = image.astype(np.uint8)
        
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
        
    return image, extract_dicom_meta(dcm)

def process_image(img_data: bytes):
    """Parses typical image formats (PNG/JPG) into an RGB image."""
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or corrupt image format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, {"Source": "Standard Image Upload"}

def convert_to_base64_png(image_rgb: np.ndarray, max_size: int = 800) -> str:
    """Resizes RGB image for web transfer and returns base64 string."""
    h, w, _ = image_rgb.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

@torch.no_grad()
def run_model_inference(model, device, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """Runs prediction on RGB image and outputs normalized box coordinates."""
    # Resize and normalize for model input
    pil_img = Image.fromarray(image_rgb)
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(pil_img).to(device)
    outputs = model([img_tensor])
    
    output = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        # Bounding box is in range [0, 512]
        x1, y1, x2, y2 = box
        
        # Normalize coordinates to [0, 1] relative to 512
        x1_norm = float(x1 / config.IMAGE_SIZE)
        y1_norm = float(y1 / config.IMAGE_SIZE)
        x2_norm = float(x2 / config.IMAGE_SIZE)
        y2_norm = float(y2 / config.IMAGE_SIZE)
        
        predictions.append({
            "box": [x1_norm, y1_norm, x2_norm, y2_norm],
            "score": float(score),
            "label": int(label)
        })
        
    return predictions

# Thread-safe Communication queues and Training Managers
update_queue = queue.Queue()

class GUIBackgroundTrainingManager:
    def __init__(self):
        self.is_training = False
        self.stop_requested = False
        self.thread = None
        self.status = {
            "state": "idle",
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "lr": 0.0,
            "message": "System idle.",
            "history": {"train_loss": [], "val_loss": [], "learning_rate": []}
        }
        self.active_websockets = []
        self.lock = threading.Lock()

    def send_log(self, text: str):
        update_queue.put({
            "type": "log",
            "timestamp": time.strftime("%H:%M:%S"),
            "message": text
        })

    def send_state_change(self, state: str, message: str):
        with self.lock:
            self.status["state"] = state
            self.status["message"] = message
        update_queue.put({
            "type": "state_change",
            "state": state,
            "message": message
        })

    def send_progress(self, epoch: int, step: int, total_steps: int, loss: float, lr: float):
        with self.lock:
            self.status["epoch"] = epoch
            self.status["step"] = step
            self.status["total_steps"] = total_steps
            self.status["loss"] = loss
            self.status["lr"] = lr
        update_queue.put({
            "type": "progress",
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "lr": lr
        })

    def send_epoch_end(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        with self.lock:
            self.status["epoch"] = epoch
            self.status["loss"] = train_loss
            self.status["val_loss"] = val_loss
            self.status["lr"] = lr
            self.status["history"]["train_loss"].append(train_loss)
            self.status["history"]["val_loss"].append(val_loss)
            self.status["history"]["learning_rate"].append(lr)
        update_queue.put({
            "type": "epoch_end",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "history": self.status["history"]
        })

training_manager = GUIBackgroundTrainingManager()

def training_thread_worker(params):
    """Executes the training cycle and communicates back to training_manager."""
    with training_manager.lock:
        training_manager.is_training = True
        training_manager.stop_requested = False
        training_manager.status["history"] = {"train_loss": [], "val_loss": [], "learning_rate": []}
        
    training_manager.send_state_change("training", "Initializing datasets...")
    training_manager.send_log("Starting training pipeline...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_manager.send_log(f"Target execution device: {device}")
        
        # Load datasets
        train_loader, val_loader, _, _ = prepare_data(
            csv_path=config.TRAIN_LABELS,
            train_dir=config.TRAIN_DIR,
            batch_size=params["batch_size"],
            split_ratio=config.TRAIN_VAL_SPLIT,
            use_augmentation=params["use_augmentation"],
            image_size=config.IMAGE_SIZE,
            num_workers=0,  # 0 workers avoids pickling/process issues on Windows in threads
            random_seed=config.RANDOM_SEED,
            persistent_workers=False,
            sample_size=params["sample_size"]
        )
        
        total_steps = len(train_loader)
        training_manager.send_log(f"DataLoader ready. Epoch steps: {total_steps}")
        
        # Load Model
        training_manager.send_log("Loading model architecture...")
        model = detection_model.get_model(num_classes=2, pretrained=True, model_type='fasterrcnn')
        detection_model.freeze_backbone(model, layers_to_freeze=2)
        model.to(device)
        
        optimizer = optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=params["lr"],
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        ) if params["scheduler"] == "plateau" else None
        
        scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and torch.cuda.is_available() else None
        best_val_loss = float('inf')
        
        training_manager.send_state_change("training", "Training started...")
        training_manager.send_log("Training loop started.")
        
        for epoch in range(1, params["epochs"] + 1):
            if training_manager.stop_requested:
                break
                
            model.train()
            total_loss = 0.0
            
            for step, (images, targets) in enumerate(train_loader):
                if training_manager.stop_requested:
                    break
                    
                optimizer.zero_grad(set_to_none=True)
                
                # Move to device
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                if config.USE_AMP and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                    
                loss_val = losses.item()
                total_loss += loss_val
                
                training_manager.send_progress(epoch, step + 1, total_steps, loss_val, optimizer.param_groups[0]['lr'])
                
            if training_manager.stop_requested:
                break
                
            avg_train_loss = total_loss / total_steps if total_steps > 0 else 0
            
            # Validation loss evaluation
            training_manager.send_log(f"Evaluating epoch {epoch} validation performance...")
            model.train()  # torchvision detection models require train mode to output validation loss
            val_loss = 0.0
            val_steps = len(val_loader)
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device, non_blocking=True) for img in images]
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    
                    if config.USE_AMP and scaler is not None:
                        with torch.cuda.amp.autocast():
                            loss_dict = model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                    else:
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            if scheduler is not None:
                scheduler.step(avg_val_loss)
                
            training_manager.send_epoch_end(epoch, avg_train_loss, avg_val_loss, current_lr)
            training_manager.send_log(f"Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Checkpoint save
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = config.MODEL_SAVE_PATH
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, checkpoint_path)
                training_manager.send_log(f"New best checkpoint saved (val_loss: {avg_val_loss:.4f})")
                
        if training_manager.stop_requested:
            training_manager.send_state_change("paused", "Training halted by user request.")
            training_manager.send_log("Training stopped.")
        else:
            training_manager.send_state_change("completed", "Training successfully complete!")
            training_manager.send_log("Process finished.")
            
    except Exception as e:
        import traceback
        training_manager.send_log(f"Training loop execution error:\n{traceback.format_exc()}")
        training_manager.send_state_change("error", f"Error: {str(e)}")
    finally:
        with training_manager.lock:
            training_manager.is_training = False
            
        # Try loading the newly trained model
        try:
            get_inference_model()
        except Exception:
            pass


async def websocket_event_broadcaster():
    """Background async loop matching FastAPI event thread, broadcasting queue data to clients."""
    while True:
        try:
            while not update_queue.empty():
                item = update_queue.get_nowait()
                # Broadcast to active clients
                disconnected_sockets = []
                for ws in list(training_manager.active_websockets):
                    try:
                        await ws.send_json(item)
                    except WebSocketDisconnect:
                        disconnected_sockets.append(ws)
                    except Exception:
                        disconnected_sockets.append(ws)
                        
                for ws in disconnected_sockets:
                    if ws in training_manager.active_websockets:
                        training_manager.active_websockets.remove(ws)
                update_queue.task_done()
        except Exception as e:
            print(f"Error in broadcaster: {e}")
        await asyncio.sleep(0.1)

# API ENDPOINTS

@app.on_event("startup")
def startup_event():
    """Run broadcaster on server startup."""
    asyncio.create_task(websocket_event_broadcaster())

@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.get("/api/diagnostics")
def diagnostics():
    """Returns general diagnostics about model checkpoints and hardware availability."""
    gpu = get_gpu_info()
    checkpoint_exists = os.path.exists(config.MODEL_SAVE_PATH)
    checkpoint_time = os.path.getmtime(config.MODEL_SAVE_PATH) if checkpoint_exists else 0
    
    return {
        "device": "GPU" if gpu["available"] else "CPU",
        "gpu_name": gpu["name"],
        "gpu_count": gpu["count"],
        "vram_total_gb": gpu["vram_total"],
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_modified": time.ctime(checkpoint_time) if checkpoint_time else "N/A",
        "checkpoint_size_mb": round(os.path.getsize(config.MODEL_SAVE_PATH) / 1024**2, 2) if checkpoint_exists else 0,
        "dataset_train_images_count": len(os.listdir(config.TRAIN_DIR)) if os.path.exists(config.TRAIN_DIR) else 0,
        "dataset_annotations_count": len(df_labels) if not df_labels.empty else 0
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Runs inference on uploaded chest X-ray file (DICOM or standard image)."""
    try:
        file_bytes = await file.read()
        filename = file.filename.lower()
        
        # 1. Parse Image
        if filename.endswith('.dcm'):
            image_rgb, metadata = process_dicom(file_bytes)
        else:
            image_rgb, metadata = process_image(file_bytes)
            
        # 2. Get Model
        model, device = get_inference_model()
        
        # 3. Predict
        predictions = run_model_inference(model, device, image_rgb)
        
        # 4. Convert Image to base64
        base64_image = convert_to_base64_png(image_rgb)
        
        return {
            "success": True,
            "filename": file.filename,
            "width": image_rgb.shape[1],
            "height": image_rgb.shape[0],
            "predictions": predictions,
            "image": base64_image,
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/samples")
def get_samples(limit: int = 100, target: Optional[int] = None):
    """Lists available validation patient IDs from labels database."""
    if df_labels.empty:
        return []
        
    # Drop duplicates for patient list
    unique_df = df_labels.drop_duplicates(subset=['patientId'])
    
    if target is not None:
        unique_df = unique_df[unique_df['Target'] == target]
        
    samples = unique_df.head(limit)[['patientId', 'Target']].to_dict(orient='records')
    return samples

@app.get("/api/sample/{patient_id}")
def get_sample_data(patient_id: str):
    """Loads DICOM, runs model inference, retrieves ground truth coordinates."""
    if df_labels.empty:
        raise HTTPException(status_code=404, detail="Dataset annotations not loaded.")
        
    dcm_path = os.path.join(config.TRAIN_DIR, f"{patient_id}.dcm")
    if not os.path.exists(dcm_path):
        raise HTTPException(status_code=404, detail=f"DICOM file not found for patient {patient_id}")
        
    try:
        # Load and process image
        with open(dcm_path, 'rb') as f:
            file_bytes = f.read()
        image_rgb, metadata = process_dicom(file_bytes)
        
        # Extract ground truth boxes
        rows = df_labels[df_labels['patientId'] == patient_id]
        ground_truths = []
        for _, row in rows.iterrows():
            if pd.notna(row['x']) and pd.notna(row['y']) and pd.notna(row['width']) and pd.notna(row['height']):
                # Original boxes are relative to 1024
                x = row['x'] / config.ORIGINAL_SIZE
                y = row['y'] / config.ORIGINAL_SIZE
                w = row['width'] / config.ORIGINAL_SIZE
                h = row['height'] / config.ORIGINAL_SIZE
                ground_truths.append([x, y, x + w, y + h])
                
        # Run inference
        model, device = get_inference_model()
        predictions = run_model_inference(model, device, image_rgb)
        
        base64_image = convert_to_base64_png(image_rgb)
        
        return {
            "patientId": patient_id,
            "target": int(rows.iloc[0]['Target']),
            "metadata": metadata,
            "image": base64_image,
            "width": image_rgb.shape[1],
            "height": image_rgb.shape[0],
            "ground_truths": ground_truths,
            "predictions": predictions
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/start")
def start_training(
    epochs: int = Form(5),
    batch_size: int = Form(2),
    lr: float = Form(1e-4),
    optimizer: str = Form("adamw"),
    scheduler: str = Form("plateau"),
    sample_size: int = Form(100),
    use_augmentation: bool = Form(True)
):
    """Launches custom training thread."""
    with training_manager.lock:
        if training_manager.is_training:
            return {"success": False, "message": "Training is already in progress."}
            
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "sample_size": sample_size,
            "use_augmentation": use_augmentation
        }
        
        training_manager.thread = threading.Thread(
            target=training_thread_worker,
            args=(params,),
            daemon=True
        )
        training_manager.thread.start()
        
    return {"success": True, "message": "Training started in background thread."}

@app.post("/api/train/stop")
def stop_training():
    """Gracefully requests training stoppage."""
    with training_manager.lock:
        if not training_manager.is_training:
            return {"success": False, "message": "No active training pipeline to stop."}
        training_manager.stop_requested = True
    return {"success": True, "message": "Stop requested. Halted epochs will complete cleanly."}

@app.get("/api/train/status")
def get_training_status():
    """Returns state variables of background manager."""
    with training_manager.lock:
        return {
            "is_training": training_manager.is_training,
            "status": training_manager.status
        }

@app.get("/api/metrics")
def get_metrics():
    """Loads metrics comparison charts if they exist in the output folder."""
    history_path = os.path.join(config.OUTPUT_DIR, 'history.json')
    history = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            pass
            
    # Check if we have pre-rendered PNGs and encode them to base64
    images = {}
    for filename, key in [
        ('comparison.png', 'comparison'),
        ('training_history.png', 'training_history'),
        ('precision_recall.png', 'precision_recall'),
        ('iou_distribution.png', 'iou_distribution')
    ]:
        path = os.path.join(config.OUTPUT_DIR, filename)
        if not os.path.exists(path):
            # Try workspace root fallback
            path = filename
            
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    img_bytes = f.read()
                b64_str = base64.b64encode(img_bytes).decode('utf-8')
                images[key] = f"data:image/png;base64,{b64_str}"
            except Exception:
                pass
                
    return {
        "history": history,
        "images": images
    }

@app.websocket("/api/train/ws")
async def websocket_training_endpoint(websocket: WebSocket):
    """Subscribes client websocket to training notifications."""
    await websocket.accept()
    training_manager.active_websockets.append(websocket)
    
    # Send initial state
    with training_manager.lock:
        initial_state = {
            "type": "state_change",
            "state": training_manager.status["state"],
            "message": training_manager.status["message"]
        }
        await websocket.send_json(initial_state)
        
        # If history already contains data, send it
        if training_manager.status["history"]["train_loss"]:
            await websocket.send_json({
                "type": "epoch_end",
                "epoch": training_manager.status["epoch"],
                "train_loss": training_manager.status["loss"],
                "val_loss": training_manager.status["val_loss"],
                "lr": training_manager.status["lr"],
                "history": training_manager.status["history"]
            })
            
    try:
        while True:
            # Maintain connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in training_manager.active_websockets:
            training_manager.active_websockets.remove(websocket)
    except Exception:
        if websocket in training_manager.active_websockets:
            training_manager.active_websockets.remove(websocket)

# Mount Frontend static files
# Make sure static directory exists before mounting
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Redirect roots to UI page."""
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    port = 5000
    host = "127.0.0.1"
    
    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception as e:
            print(f"Browser launch error: {e}")
            
    threading.Thread(target=open_browser, daemon=True).start()
    
    print(f"Server initializing on http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port)
