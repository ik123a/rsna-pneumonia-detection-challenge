// Global state variables
let currentTab = 'overview';
let systemStats = null;

// Inference State
let infCachedData = null; // Stores { image, width, height, predictions, metadata }
let infLoadedImage = null; // Image object for canvas drawing

// Database Browser State
let browserCachedData = null; // Stores current sample: { image, ground_truths, predictions, target, metadata }
let browserLoadedImage = null; // Image object
let showGroundTruth = true;
let showPredictions = true;

// Training State
let trainingSocket = null;
let lossChart = null;
let trainingHistory = { train_loss: [], val_loss: [], learning_rate: [] };

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initOverview();
    initInference();
    initBrowser();
    initTraining();
    initEvaluation();
});

// ==========================================
// 1. NAVIGATION & TAB MANAGEMENT
// ==========================================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pageTitle = document.getElementById('page-title');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = item.getAttribute('data-tab');
            if (tab === currentTab) return;
            
            // Toggle active classes in nav
            document.querySelector('.nav-item.active').classList.remove('active');
            item.classList.add('active');
            
            // Toggle active classes in panels
            document.querySelector('.tab-panel.active').classList.remove('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
            
            currentTab = tab;
            pageTitle.innerText = item.querySelector('span').innerText;
            
            // Load specific tab datasets
            onTabChanged(tab);
        });
    });
    
    // Check backend health
    checkBackendHealth();
    setInterval(checkBackendHealth, 10000);
}

async function checkBackendHealth() {
    const dot = document.getElementById('backend-dot');
    const text = document.getElementById('backend-status-text');
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            dot.className = 'status-dot';
            text.innerText = 'Online';
        } else {
            throw new Error();
        }
    } catch (e) {
        dot.className = 'status-dot offline';
        text.innerText = 'Offline';
    }
}

function onTabChanged(tab) {
    if (tab === 'overview') {
        loadSystemDiagnostics();
    } else if (tab === 'browser') {
        loadBrowserSamples();
    } else if (tab === 'training') {
        syncTrainingStatus();
    } else if (tab === 'evaluation') {
        loadEvaluationMetrics();
    }
}

// ==========================================
// 2. SYSTEM OVERVIEW / DIAGNOSTICS
// ==========================================
function initOverview() {
    loadSystemDiagnostics();
}

async function loadSystemDiagnostics() {
    try {
        const response = await fetch('/api/diagnostics');
        const data = await response.json();
        systemStats = data;
        
        // Update Overview widgets
        document.getElementById('diag-device').innerText = data.device;
        document.getElementById('diag-images').innerText = data.dataset_train_images_count.toLocaleString();
        document.getElementById('diag-ckpt-size').innerText = data.checkpoint_size_mb > 0 ? `${data.checkpoint_size_mb} MB` : 'N/A';
        
        // Update Details Table
        document.getElementById('diag-gpu-name').innerText = data.gpu_name || 'N/A';
        document.getElementById('diag-gpu-count').innerText = data.gpu_count;
        document.getElementById('diag-vram').innerText = data.vram_total_gb > 0 ? `${data.vram_total_gb} GB` : 'N/A';
        document.getElementById('diag-ckpt-time').innerText = data.checkpoint_modified;
        document.getElementById('diag-annotations').innerText = data.dataset_annotations_count.toLocaleString();
    } catch (e) {
        console.error("Error loading system diagnostics:", e);
    }
}

// ==========================================
// 3. DIAGNOSTIC TOOL (INFERENCE)
// ==========================================
function initInference() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const confSlider = document.getElementById('inf-conf-slider');
    const confVal = document.getElementById('inf-conf-val');
    
    // Upload drag and drop handlers
    dropArea.addEventListener('click', () => fileInput.click());
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.style.borderColor = 'var(--accent-purple)';
            dropArea.style.background = 'rgba(168, 85, 247, 0.05)';
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropArea.style.borderColor = 'rgba(255, 255, 255, 0.15)';
            dropArea.style.background = 'rgba(255, 255, 255, 0.01)';
        }, false);
    });
    
    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) uploadXrayFile(files[0]);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length) uploadXrayFile(fileInput.files[0]);
    });
    
    // Slider updates
    confSlider.addEventListener('input', () => {
        confVal.innerText = parseFloat(confSlider.value).toFixed(2);
        if (infCachedData) {
            drawInferenceCanvas();
            updateInferenceReport();
        }
    });
}

async function uploadXrayFile(file) {
    const loader = document.getElementById('inference-loader');
    loader.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (!data.success) {
            alert("Model prediction error: " + data.error);
            return;
        }
        
        // Cache coordinates & metadata
        infCachedData = {
            image: data.image,
            width: data.width,
            height: data.height,
            predictions: data.predictions,
            metadata: data.metadata
        };
        
        // Load image to draw on canvas
        infLoadedImage = new Image();
        infLoadedImage.onload = () => {
            drawInferenceCanvas();
            updateInferenceReport();
            loader.style.display = 'none';
        };
        infLoadedImage.src = data.image;
        
    } catch (e) {
        console.error("Upload error:", e);
        alert("Upload error. Check server log.");
        loader.style.display = 'none';
    }
}

function drawInferenceCanvas() {
    const canvas = document.getElementById('inference-canvas');
    const ctx = canvas.getContext('2d');
    const threshold = parseFloat(document.getElementById('inf-conf-slider').value);
    
    if (!infLoadedImage) return;
    
    // Size canvas to fit container retaining aspect ratio
    canvas.width = infCachedData.width;
    canvas.height = infCachedData.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. Draw base scan
    ctx.drawImage(infLoadedImage, 0, 0, canvas.width, canvas.height);
    
    // 2. Draw prediction boxes
    infCachedData.predictions.forEach((pred, index) => {
        if (pred.score < threshold) return;
        
        const [x1_n, y1_n, x2_n, y2_n] = pred.box;
        
        // Scale normalized box coordinates to display dimensions
        const x = x1_n * canvas.width;
        const y = y1_n * canvas.height;
        const w = (x2_n - x1_n) * canvas.width;
        const h = (y2_n - y1_n) * canvas.height;
        
        // Style box: Purple matching patient predictions
        ctx.strokeStyle = 'purple';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);
        
        // Label Text
        const text = `pneumonia ${pred.score.toFixed(2)}`;
        ctx.font = 'bold 16px Outfit, sans-serif';
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillStyle = 'purple';
        ctx.fillRect(x, y - 25, textWidth + 10, 25);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x + 5, y - 7);
    });
}

function updateInferenceReport() {
    const summaryBox = document.getElementById('inf-report-summary');
    const badge = document.getElementById('inf-report-badge');
    const title = document.getElementById('inf-report-title');
    const desc = document.getElementById('inf-report-desc');
    const boxList = document.getElementById('inf-box-list');
    const metaTable = document.getElementById('inf-meta-table');
    const threshold = parseFloat(document.getElementById('inf-conf-slider').value);
    
    if (!infCachedData) return;
    
    // Filter active predictions
    const activePreds = infCachedData.predictions.filter(p => p.score >= threshold);
    
    boxList.innerHTML = '';
    
    if (activePreds.length > 0) {
        summaryBox.className = 'report-summary-box positive';
        badge.className = 'clinical-badge positive';
        badge.innerText = 'PNEUMONIA DETECTED';
        badge.style.display = 'inline-block';
        title.innerText = `${activePreds.length} lesion(s) found`;
        desc.innerText = 'Model detected high density opacity corresponding to pneumonia consolidation. Bounding boxes are plotted above.';
        
        // Render box list
        activePreds.forEach((pred, index) => {
            const [x1, y1, x2, y2] = pred.box;
            const item = document.createElement('div');
            item.className = 'box-coordinate-item';
            item.innerHTML = `
                <span><span class="box-num">#${index+1}</span> Conf: ${(pred.score * 100).toFixed(0)}%</span>
                <span>[${(x1*100).toFixed(0)}%, ${(y1*100).toFixed(0)}% w: ${((x2-x1)*100).toFixed(0)}%]</span>
            `;
            boxList.appendChild(item);
        });
        boxList.style.display = 'flex';
    } else {
        summaryBox.className = 'report-summary-box negative';
        badge.className = 'clinical-badge negative';
        badge.innerText = 'NORMAL';
        badge.style.display = 'inline-block';
        title.innerText = 'No anomalies detected';
        desc.innerText = 'Scan shows clear fields, or anomalies present are below selected diagnostic confidence threshold.';
        boxList.style.display = 'none';
    }
    
    // Metadata Table
    metaTable.innerHTML = '';
    for (const [key, value] of Object.entries(infCachedData.metadata)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="label">${key}</td>
            <td class="value">${value}</td>
        `;
        metaTable.appendChild(row);
    }
    metaTable.style.display = 'table';
}

// ==========================================
// 4. DATABASE SAMPLE BROWSER
// ==========================================
function initBrowser() {
    const filter = document.getElementById('browser-filter');
    const confSlider = document.getElementById('browser-conf-slider');
    const confVal = document.getElementById('browser-conf-val');
    const toggleGT = document.getElementById('toggle-gt');
    const togglePred = document.getElementById('toggle-pred');
    
    filter.addEventListener('change', loadBrowserSamples);
    
    confSlider.addEventListener('input', () => {
        confVal.innerText = parseFloat(confSlider.value).toFixed(2);
        if (browserCachedData) {
            drawBrowserCanvas();
            updateBrowserReport();
        }
    });
    
    toggleGT.addEventListener('click', () => {
        showGroundTruth = !showGroundTruth;
        toggleGT.querySelector('.toggle-circle').className = `toggle-circle green ${showGroundTruth ? '' : 'inactive'}`;
        drawBrowserCanvas();
    });
    
    togglePred.addEventListener('click', () => {
        showPredictions = !showPredictions;
        togglePred.querySelector('.toggle-circle').className = `toggle-circle purple ${showPredictions ? '' : 'inactive'}`;
        drawBrowserCanvas();
    });
}

async function loadBrowserSamples() {
    const container = document.getElementById('sample-list-container');
    container.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem; padding:10px;">Loading list...</p>';
    
    const filterVal = document.getElementById('browser-filter').value;
    let url = '/api/samples?limit=100';
    if (filterVal === 'positive') url += '&target=1';
    if (filterVal === 'negative') url += '&target=0';
    
    try {
        const response = await fetch(url);
        const data = await response.json();
        
        container.innerHTML = '';
        if (data.length === 0) {
            container.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem; padding:10px;">No cases found.</p>';
            return;
        }
        
        data.forEach(sample => {
            const item = document.createElement('div');
            item.className = 'sample-list-item';
            item.setAttribute('data-id', sample.patientId);
            item.innerHTML = `
                <span class="sample-id">${sample.patientId.substring(0, 8)}...</span>
                <span class="sample-tag ${sample.Target === 1 ? 'positive' : 'negative'}">${sample.Target === 1 ? 'Pneumonia' : 'Normal'}</span>
            `;
            item.addEventListener('click', () => selectBrowserSample(sample.patientId, item));
            container.appendChild(item);
        });
        
        // Auto select first sample
        if (container.firstElementChild) container.firstElementChild.click();
        
    } catch (e) {
        console.error("List load error:", e);
        container.innerHTML = '<p style="color:var(--accent-red); font-size:0.85rem; padding:10px;">Load error.</p>';
    }
}

async function selectBrowserSample(patientId, listItem) {
    const loader = document.getElementById('browser-loader');
    loader.style.display = 'flex';
    
    // Toggle active list item styling
    const activeItem = document.querySelector('.sample-list-item.active');
    if (activeItem) activeItem.classList.remove('active');
    listItem.classList.add('active');
    
    try {
        const response = await fetch(`/api/sample/${patientId}`);
        if (!response.ok) throw new Error("Load failed");
        const data = await response.json();
        
        browserCachedData = data;
        
        // Load image to draw
        browserLoadedImage = new Image();
        browserLoadedImage.onload = () => {
            drawBrowserCanvas();
            updateBrowserReport();
            loader.style.display = 'none';
        };
        browserLoadedImage.src = data.image;
        
    } catch (e) {
        console.error("Error loading sample data:", e);
        alert("Error loading patient sample.");
        loader.style.display = 'none';
    }
}

function drawBrowserCanvas() {
    const canvas = document.getElementById('browser-canvas');
    const ctx = canvas.getContext('2d');
    const threshold = parseFloat(document.getElementById('browser-conf-slider').value);
    
    if (!browserLoadedImage) return;
    
    canvas.width = browserCachedData.width;
    canvas.height = browserCachedData.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 1. Draw image scan
    ctx.drawImage(browserLoadedImage, 0, 0, canvas.width, canvas.height);
    
    // 2. Draw ground truth boxes (Green)
    if (showGroundTruth && browserCachedData.ground_truths) {
        browserCachedData.ground_truths.forEach(box => {
            const [x1_n, y1_n, x2_n, y2_n] = box;
            const x = x1_n * canvas.width;
            const y = y1_n * canvas.height;
            const w = (x2_n - x1_n) * canvas.width;
            const h = (y2_n - y1_n) * canvas.height;
            
            ctx.strokeStyle = '#10b981'; // Green
            ctx.lineWidth = 3;
            ctx.setLineDash([6, 6]); // Dashed line for ground truth
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]); // Reset
            
            // Text Label
            ctx.fillStyle = '#10b981';
            ctx.font = '14px Outfit, sans-serif';
            ctx.fillRect(x, y - 22, 100, 22);
            ctx.fillStyle = 'white';
            ctx.fillText('Ground Truth', x + 5, y - 6);
        });
    }
    
    // 3. Draw predictions (Purple)
    if (showPredictions && browserCachedData.predictions) {
        browserCachedData.predictions.forEach(pred => {
            if (pred.score < threshold) return;
            
            const [x1_n, y1_n, x2_n, y2_n] = pred.box;
            const x = x1_n * canvas.width;
            const y = y1_n * canvas.height;
            const w = (x2_n - x1_n) * canvas.width;
            const h = (y2_n - y1_n) * canvas.height;
            
            ctx.strokeStyle = 'purple';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Text Label
            const text = `Pred: ${(pred.score*100).toFixed(0)}%`;
            ctx.fillStyle = 'purple';
            ctx.font = '14px Outfit, sans-serif';
            ctx.fillRect(x, y - 22, 90, 22);
            ctx.fillStyle = 'white';
            ctx.fillText(text, x + 5, y - 6);
        });
    }
}

function updateBrowserReport() {
    const summaryBox = document.getElementById('browser-report-summary');
    const badge = document.getElementById('browser-report-badge');
    const title = document.getElementById('browser-report-title');
    const desc = document.getElementById('browser-report-desc');
    const metaTable = document.getElementById('browser-meta-table');
    const threshold = parseFloat(document.getElementById('browser-conf-slider').value);
    
    if (!browserCachedData) return;
    
    const activePreds = browserCachedData.predictions.filter(p => p.score >= threshold);
    const hasGt = browserCachedData.ground_truths.length > 0;
    
    if (hasGt) {
        summaryBox.className = 'report-summary-box positive';
        badge.className = 'clinical-badge positive';
        badge.innerText = 'PNEUMONIA CASE';
        badge.style.display = 'inline-block';
        title.innerText = `Annotated scan (${browserCachedData.ground_truths.length} lesion targets)`;
        desc.innerText = `Medical experts marked ${browserCachedData.ground_truths.length} consolidations. Model predicted ${activePreds.length} detections at current threshold.`;
    } else {
        summaryBox.className = 'report-summary-box negative';
        badge.className = 'clinical-badge negative';
        badge.innerText = 'NORMAL CASE';
        badge.style.display = 'inline-block';
        title.innerText = 'Scan clear of consolidations';
        desc.innerText = `No annotations recorded. Model predicted ${activePreds.length} boxes at current filter settings.`;
    }
    
    // Fill Meta Table
    metaTable.innerHTML = '';
    for (const [key, value] of Object.entries(browserCachedData.metadata)) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="label">${key}</td>
            <td class="value">${value}</td>
        `;
        metaTable.appendChild(row);
    }
}

// ==========================================
// 5. TRAINING DASHBOARD
// ==========================================
function initTraining() {
    const btnStart = document.getElementById('btn-start-train');
    const btnStop = document.getElementById('btn-stop-train');
    
    btnStart.addEventListener('click', startTraining);
    btnStop.addEventListener('click', stopTraining);
    
    // Initialize Chart
    const ctx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    borderColor: '#a855f7', // Purple
                    backgroundColor: 'rgba(168, 85, 247, 0.05)',
                    borderWidth: 2,
                    data: [],
                    tension: 0.2
                },
                {
                    label: 'Val Loss',
                    borderColor: '#0ea5e9', // Blue
                    backgroundColor: 'rgba(14, 165, 233, 0.05)',
                    borderWidth: 2,
                    data: [],
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    title: { display: true, text: 'Loss Value', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
    
    syncTrainingStatus();
    connectWebSocket();
}

async function syncTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        const data = await response.json();
        
        toggleTrainingFormUI(data.is_training);
        
        if (data.is_training) {
            // Restore training logs/graphs
            const progress = document.getElementById('train-progress-block');
            progress.style.display = 'block';
            updateTrainingProgressUI(data.status);
        }
    } catch (e) {
        console.error("Error syncing training status:", e);
    }
}

function connectWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = `${wsProto}${window.location.host}/api/train/ws`;
    
    trainingSocket = new WebSocket(wsUrl);
    
    trainingSocket.onopen = () => {
        appendConsoleLog("WS-SYS", "WebSocket status pipeline connected.", "success");
    };
    
    trainingSocket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        if (msg.type === 'log') {
            appendConsoleLog("TRAIN", msg.message);
        } else if (msg.type === 'progress') {
            updateBatchProgressUI(msg);
        } else if (msg.type === 'epoch_end') {
            updateEpochChartUI(msg);
        } else if (msg.type === 'state_change') {
            appendConsoleLog("SYS", `State changed: ${msg.state.toUpperCase()} (${msg.message})`, msg.state === 'error' ? 'error' : 'success');
            toggleTrainingFormUI(msg.state === 'training');
        }
    };
    
    trainingSocket.onclose = () => {
        appendConsoleLog("WS-SYS", "WebSocket pipeline disconnected. Reconnecting in 3s...", "error");
        setTimeout(connectWebSocket, 3000);
    };
}

function appendConsoleLog(source, text, style = '') {
    const terminal = document.getElementById('terminal-output');
    const time = new Date().toLocaleTimeString();
    
    const line = document.createElement('div');
    line.className = `terminal-line ${style}`;
    line.innerHTML = `<span class="time">[${time}]</span> <strong style="color:var(--accent-purple-light)">[${source}]</strong> ${text}`;
    
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}

function toggleTrainingFormUI(isTraining) {
    const formInputs = document.querySelectorAll('#train-form input, #train-form select');
    const btnStart = document.getElementById('btn-start-train');
    const btnStop = document.getElementById('btn-stop-train');
    const progressBlock = document.getElementById('train-progress-block');
    
    formInputs.forEach(input => input.disabled = isTraining);
    btnStart.disabled = isTraining;
    btnStop.disabled = !isTraining;
    
    if (isTraining) {
        progressBlock.style.display = 'block';
    }
}

async function startTraining() {
    const form = document.getElementById('train-form');
    const formData = new FormData();
    formData.append('epochs', document.getElementById('train-epochs').value);
    formData.append('batch_size', document.getElementById('train-batch').value);
    formData.append('lr', document.getElementById('train-lr').value);
    formData.append('optimizer', document.getElementById('train-opt').value);
    formData.append('scheduler', document.getElementById('train-scheduler').value);
    formData.append('sample_size', document.getElementById('train-samples').value);
    formData.append('use_augmentation', document.getElementById('train-aug').checked);
    
    // Clear previous charts
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.data.datasets[1].data = [];
    lossChart.update();
    
    try {
        const response = await fetch('/api/train/start', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.success) {
            appendConsoleLog("API", "Training thread initialization sent.", "success");
            toggleTrainingFormUI(true);
        } else {
            alert("Could not start training: " + data.message);
        }
    } catch (e) {
        console.error("Start training error:", e);
    }
}

async function stopTraining() {
    try {
        const response = await fetch('/api/train/stop', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            appendConsoleLog("API", "Pause request dispatched to trainer. Current epoch will wrap up first.", "error");
        } else {
            alert(data.message);
        }
    } catch (e) {
        console.error(e);
    }
}

function updateBatchProgressUI(msg) {
    document.getElementById('progress-epoch-text').innerText = `Epoch ${msg.epoch}`;
    const pct = Math.round((msg.step / msg.total_steps) * 100);
    document.getElementById('progress-pct-text').innerText = `${pct}% (${msg.step}/${msg.total_steps})`;
    document.getElementById('progress-fill-bar').style.width = `${pct}%`;
    document.getElementById('progress-loss-val').innerText = msg.loss.toFixed(4);
}

function updateEpochChartUI(msg) {
    // Add epoch label
    if (!lossChart.data.labels.includes(`Ep ${msg.epoch}`)) {
        lossChart.data.labels.push(`Ep ${msg.epoch}`);
        lossChart.data.datasets[0].data.push(msg.train_loss);
        lossChart.data.datasets[1].data.push(msg.val_loss);
        lossChart.update();
    }
}

function updateTrainingProgressUI(status) {
    if (status.step > 0) {
        updateBatchProgressUI({
            epoch: status.epoch,
            step: status.step,
            total_steps: status.total_steps,
            loss: status.loss
        });
    }
    
    // Restore chart if history loaded
    if (status.history && status.history.train_loss.length > 0) {
        lossChart.data.labels = status.history.train_loss.map((_, i) => `Ep ${i+1}`);
        lossChart.data.datasets[0].data = status.history.train_loss;
        lossChart.data.datasets[1].data = status.history.val_loss;
        lossChart.update();
    }
}

// ==========================================
// 6. MODEL EVALUATION & COMPARISON
// ==========================================
function initEvaluation() {
    loadEvaluationMetrics();
}

async function loadEvaluationMetrics() {
    const container = document.getElementById('plots-container');
    container.innerHTML = '<div class="no-data-msg">Loading curves and comparison reports...</div>';
    
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        container.innerHTML = '';
        
        const images = data.images;
        if (Object.keys(images).length === 0) {
            container.innerHTML = '<div class="no-data-msg">No curves generated in outputs. Complete training or run comparison modes to generate charts.</div>';
            return;
        }
        
        // Define display titles for plots
        const plotTitles = {
            'comparison': 'Model Metric Improvements Bar Chart',
            'training_history': 'Training and Validation Loss Curves',
            'precision_recall': 'Precision-Recall Curve (IoU=0.5)',
            'iou_distribution': 'Detections IoU Distribution Histogram'
        };
        
        for (const [key, base64Str] of Object.entries(images)) {
            const card = document.createElement('div');
            card.className = 'glass-card rendered-plot-card';
            card.innerHTML = `
                <div class="card-title" style="border-bottom:none; margin-bottom:10px;">${plotTitles[key] || key}</div>
                <img src="${base64Str}" alt="${key}">
            `;
            container.appendChild(card);
        }
    } catch (e) {
        console.error("Evaluation load error:", e);
        container.innerHTML = '<div class="no-data-msg" style="color:var(--accent-red)">Error loading metrics data.</div>';
    }
}
