# Safety Monitor Edge AI

A "Production-Grade" Industrial Safety AI monitoring system, written entirely in **Rust** and optimized specifically for constrained Edge hardware.

## 🌟 Key Highlights

### 1. Robust Core Architecture (Asynchronous LIFO Pipeline)
- **Memory Safety:** Built natively on Rust, entirely eliminating memory leaks and Garbage Collector (GC) latency spikes typical in 24/7 environments.
- **LIFO Frame Dropping & Object Pooling:** Strictly decouples Camera capture, AI Inference, and UI rendering into distinct asynchronous crossbeam channels. Reuses memory buffers (`PooledFrame`) and drops stale frames rather than queueing them. This guarantees the AI evaluates the most recent real-world frame (Zero Queue Lag).

### 2. High-Performance Computation (Hardware-Accelerated)
- **Dynamic Execution Providers:** The `ort` v2 engine automatically analyzes the system and offloads operations to the most powerful hardware backend (prioritizing **TensorRT**, **CUDA**, and **OpenVINO** before falling back to CPU cores).
- **CPU Multi-threading:** Vision pre-processing (tensor conversion) is parallelized via `Rayon`, fully saturating multi-core CPUs instead of bottlenecking the main loop sequentially.
- **INT8 Standardized:** Natively supports computing Quantized (`QDQ`) models, slashing RAM bandwidth requirements by 4x compared to FP32 models.

### 3. Industrial Safety Logic (Fault Tolerance)
- **Auto-Reconnect (Resilience):** Eliminates silent crashing (`Panic`) when industrial cameras are unpredictably disconnected. An internal retry/timeout loop guarantees the unit stays "Always Alive", auto-repairing stream pipelines when the hardware is plugged back in.
- **AI Debouncing & Cooldown:** Fights AI noise through a dedicated State Machine. A violation (e.g., No Helmet) must persist for at least **5 consecutive frames** before asserting an alarm, neutralizing false-positives. Once triggered, it enforces a **5-second punishment cooldown**, trapping the red HUD alarm on-screen for 5s even if the worker rapidly covers up the violation.
- **Automated Audit Trails:** The exact millisecond the AI locks onto a rule breakage, a detached background thread securely saves the pristine evidence frame down to the `/violations` directory attached with absolute timestamps, without pausing the video feed.

---

## 📂 Project Structure

```text
safety_monitor_edge/
├── models/
│   ├── best_640x384.onnx        # Base YOLOv8 FP32 ONNX model
│   └── best_640x384_int8.onnx   # Quantized INT8 model for Edge devices
├── src/
│   ├── alarm.rs       # State Machine handling Debouncing, Cooldown & Alerts
│   ├── camera.rs      # Robust Nokhwa camera wrapper with Auto-Reconnect fallback
│   ├── engine.rs      # ORT v2 AI Engine setup (TensorRT/CUDA/OpenVINO routing)
│   ├── main.rs        # Core LIFO Architecture & Thread Orchestrator
│   ├── processing.rs  # Rayon parallel Preprocessing & NMS Post-processing
│   ├── types.rs       # Internal shared data structures (BoundingBox, Timings)
│   └── visualizer.rs  # Dynamic OSD UI Renderer (RustType Font blending & Canvas)
├── tools/
│   └── quantize.py    # Python script for Dynamic INT8 ONNX Quantization
├── violations/        # Auto-generated directory storing incident audit snapshots
├── Cargo.toml         # Rust dependencies and Manifest
└── README.md          # You are here!
```

---

## 🚀 Usage

### 1. Local Quantization (INT8)
*(Optional)* Compress your FP32 ONNX models down to Int8 using the provided script:
```bash
python tools/quantize.py models/best_640x384.onnx models/best_640x384_int8.onnx
```

### 2. Run the Engine
Run by Best Model (INT8, 640x384):
```bash
cargo run --release
```
Run Another Model: Replace <your_model_name.onnx> with your model name. And <your_input_w>x<your_input_h> with your model input size.
```bash
MODEL_PATH=models/<your_model_name.onnx> INPUT_W=<your_input_w> INPUT_H=<your_input_h> cargo run --release
```
