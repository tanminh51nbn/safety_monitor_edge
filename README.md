# 🛡️ Safety Monitor Edge AI

![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-v2-blue.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An industrial-grade, production-ready **Safety Monitoring Edge AI**, built entirely in **Rust**. This system processes live camera feeds (USB & IP Cameras) to detect safety violations (e.g., workers not wearing hard hats) and seamlessly drives physical alarms and real-time Telegram alerts.

It is engineered from the ground up to solve the three biggest problems of Edge Computing: **RAM Leaks, Network Thread Spam, and Frame Lag Queuing**.

---

## 🌟 Key Features & Selling Points

### 🚀 **1. Hyper-Optimized Concurrency (Zero-Lag LIFO)**
Unlike typical Python scripts that choke and queue frames when the AI lags, this system utilizes a **Last-In-First-Out (LIFO) Bounded Channel** architecture using `crossbeam`. It forcefully drops stale frames to ensure the AI always processes the absolute *most recent* real-world timestamp. Zero queuing lag.

### 🧠 **2. Hardware-Accelerated INT8 AI**
Driven by `ort` v2, the engine probes your hardware and automatically attaches to the most potent Execution Provider (`TensorRT`, `CUDA`, `DirectML`, or `OpenVINO`). The model provided is **INT8 Quantized (QDQ)**, radically slashing RAM bandwidth by 400% compared to traditional FP32 models.

### 🔌 **3. Seamless IoT & Cloud Connectivity**
* **Direct Hardware GPIO:** Natively triggers physical 220V Relays/Sirens on Linux Edge devices (Raspberry Pi/Jetson) via `sysfs_gpio`. Fallbacks gracefully to Console Logging on Windows.
* **Bounded Telegram Alerts:** Instantly sends photo evidence to your Telegram via Webhooks. Protected by a **Bounded Message Queue** – even if your factory's WiFi goes down for a week, the system will *never* leak threads or crash from API blocking limit. It just gracefully drops over-queued messages to protect the AI's core RAM.

### 🛡️ **4. Industrial Fault-Tolerance**
* **Auto-Reconnect Watchdog:** If a rat chews your USB cable or the RTSP IP stream drops, the software doesn't `panic!`. The camera pipeline will gracefully sleep and infinitely attempt to reconnect until the hardware signal returns.
* **Debounce State Machine:** Eliminates false positives! A worker must be violating safety rules for **5 consecutive frames** before the alarm is tripped, followed by an enforced **5-second Cooldown** constraint to prevent spamming the alarm.

---

## 📂 Project Architecture

```text
safety_monitor_edge/
├── models/
│   ├── best_640x384.onnx        # Base YOLOv8 FP32 ONNX model
│   └── best_640x384_int8.onnx   # Quantized INT8 model for Edge devices
├── src/
│   ├── alarm.rs       # State Machine: Debouncing & Cooldown constraints
│   ├── bot_alert.rs   # Telegram REST webhook integration (Photo + Caption)
│   ├── camera.rs      # Abstraction for USB (nokhwa) & IP Camera (RTSP via OpenCV)
│   ├── engine.rs      # ORT v2 AI Engine setup (TensorRT/OpenVINO routing)
│   ├── iot_gpio.rs    # Pin 18 Relay execution for native hardware sirens
│   ├── main.rs        # Core LIFO Thread Orchestrator & Bounded Queues
│   ├── processing.rs  # Rayon parallel Preprocessing & NMS Post-processing
│   └── visualizer.rs  # Dynamic OSD UI Renderer (RustType Font blending)
├── tools/
│   └── quantize.py    # Python script for Dynamic INT8 ONNX Quantization
├── violations/        # Auto-generated directory storing incident audit snapshots
└── .env               # Configuration files for your hardware/network configs
```

---

## 🚀 Quick Start & Usage

### ⚙️ 1. Setup Environment
Create a `.env` file in the root folder of your project to tell the AI how to behave. 

```env
# CAMERA INPUT: 
# Put 0 for Local Webcam. Put "rtsp://..." for IP Camera Stream.
CAMERA_SOURCE=0

# TELEGRAM BOT INTEGRATION:
# Included is a public testing bot (@SafetyMonitorAlert_bot)
TELEGRAM_BOT_TOKEN=8629365297:AAG7rglUDNSbS9yAbSM2CCgPnsD5KM0DjNk

# Replace this with your own Personal Chat ID (Find it via @userinfobot)
TELEGRAM_CHAT_ID=your_private_chat_id_here

# IoT HARDWARE: Pin connected to your Siren/Relay (Default 18 for Linux)
GPIO_ALARM_PIN=18
```
> **Privacy Note:** We heavily enforce **1-on-1 Direct Messaging** for Telegram testing. Avoid adding the testing bot to public supergroups, so your webcam snapshots remain entirely private to your own Inbox.

### 🔨 2. Run the Engine (Standard Mode)
The default configuration automatically loads the highly optimized INT8 Quantized model (`640x384`).

```bash
cargo run --release
```

### 🛰️ 3. Enable IP Camera (RTSP Mode)
If you want to pull feeds from networked factory cameras (RTSP/H.264), you need OpenCV installed on your host system. Enable the feature flag to compile the OpenCV C++ bindings:

```bash
cargo run --release --features rtsp
```

### 🎛️ 4. Advanced Run (Custom AI Models)
You can inject your own proprietary ONNX models and specify the internal tensor constraints dynamically through terminal environment variables:

```bash
# Windows PowerShell
$env:MODEL_PATH="models/your_custom_model.onnx"; $env:INPUT_W="640"; $env:INPUT_H="640"; cargo run --release

# Linux / Mac Bash
MODEL_PATH=models/your_custom_model.onnx INPUT_W=640 INPUT_H=640 cargo run --release
```

### 🗜️ 5. Optional: Quantize your own Model
If you bring your own FP32 model and want to compress it to Edge-friendly INT8 logic, run our provided Python utility (Requires `onnxruntime` in Python):
```bash
python tools/quantize.py models/best_640x384.onnx models/best_640x384_int8.onnx
```

---
**Made with ❤️ in Rust.** Welcome to the future of Industrial Vision.
