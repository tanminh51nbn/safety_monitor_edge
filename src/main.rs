// File: src/main.rs
mod camera;
mod engine;
mod processing;
mod types;
mod visualizer;

use crossbeam_channel::{Receiver, Sender};
use image::{ImageBuffer, Rgb};
use minifb::{Key, Window, WindowOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use sysinfo::System;

use std::sync::{Arc, Mutex};
use std::thread;

const MODEL_PATH: &str = "models/best_640x384.onnx";
const INPUT_W: u32 = 640;
const INPUT_H: u32 = 384;
const TARGET_CAMERA_FPS: f64 = 24.0;

#[derive(Clone)]
struct FramePacket {
    image: PooledFrame,
    captured_at: Instant,
}

#[derive(Clone)]
struct PooledFrame {
    image: Arc<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    return_tx: Sender<Arc<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
}

impl PooledFrame {
    fn as_ref(&self) -> &ImageBuffer<Rgb<u8>, Vec<u8>> {
        &self.image
    }
}

impl Drop for PooledFrame {
    fn drop(&mut self) {
        if Arc::strong_count(&self.image) == 1 {
            let _ = self.return_tx.try_send(Arc::clone(&self.image));
        }
    }
}

#[derive(Default, Clone)]
struct Stat {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn update(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        self.sum += value;
        self.count += 1;
    }

    fn avg(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Default, Clone)]
struct Summary {
    camera_fps: Stat,
    ai_fps: Stat,
    preprocess_ms: Stat,
    inference_ms: Stat,
    postprocess_ms: Stat,
    total_ms: Stat,
    e2e_ms: Stat,
    rss_mb: Stat,
    cpu_pct: Stat,
}

fn main() {
    println!("=== HỆ THỐNG GIÁM SÁT AN TOÀN EDGE AI ===");

    // 1. KHỞI TẠO MẮT (CAMERA)
    println!("Đang khởi động Camera...");
    let mut edge_cam = camera::EdgeCamera::new().expect("Lỗi: Không thể khởi tạo Camera!");

    let first_frame = edge_cam
        .capture_frame()
        .expect("Lỗi: Không thể chụp ảnh mồi!");
    let width = first_frame.width() as usize; // Thường là 1280
    let height = first_frame.height() as usize; // Thường là 720

    // 2. KHỞI TẠO ĐA LUỒNG (LIFO Producer-Consumer Pattern)
    println!("Đang tạo Kiến trúc Đa Luồng cho AI Engine...");
    let latest_frame: Arc<Mutex<Option<FramePacket>>> = Arc::new(Mutex::new(None));
    let latest_boxes = Arc::new(Mutex::new(Vec::new()));
    let summary: Arc<Mutex<Summary>> = Arc::new(Mutex::new(Summary::default()));
    let dropped_frames = Arc::new(Mutex::new(0_u64));
    let processed_frames = Arc::new(Mutex::new(0_u64));
    let skipped_frames = Arc::new(Mutex::new(0_u64));
    let running = Arc::new(AtomicBool::new(true));

    let (notify_tx, notify_rx) = crossbeam_channel::bounded(1);
    let (render_tx, render_rx): (Sender<FramePacket>, Receiver<FramePacket>) =
        crossbeam_channel::bounded(1);
    let (boxes_tx, boxes_rx) = crossbeam_channel::bounded(1);
    let (pool_tx, pool_rx): (
        Sender<Arc<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
        Receiver<Arc<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    ) = crossbeam_channel::bounded(4);

    let worker_frame = latest_frame.clone();
    let worker_boxes = latest_boxes.clone();
    let worker_summary = summary.clone();
    let worker_processed = processed_frames.clone();
    let worker_boxes_tx = boxes_tx.clone();

    // Khởi chạy Worker Thread (Chỉ làm nhiệm vụ AI ngầm)
    thread::spawn(move || {
        println!(
            "Worker (AI): Đang nạp mô hình AI ({}) vào RAM...",
            MODEL_PATH
        );
        println!("Worker (AI): Input size = {}x{}", INPUT_W, INPUT_H);
        let mut ai_engine = engine::AIEngine::new(MODEL_PATH, INPUT_W, INPUT_H)
            .expect("Lỗi: Không tìm thấy file mô hình hoặc lỗi nạp ORT!");
        println!("Worker (AI): Đã sẵn sàng suy luận!");

        let mut ai_frames: u32 = 0;
        let mut ai_timer = Instant::now();

        // Vòng lặp suy luận chạy độc lập
        while let Ok(_) = notify_rx.recv() {
            // Lấy ảnh mới nhất ra khỏi bộ nhớ chung (đồng thời dọn trống luôn - LIFO)
            let frame = worker_frame.lock().unwrap().take();

            if let Some(packet) = frame {
                // Chạy AI trên ảnh, không lo bị chặn UI
                if let Ok(output) = ai_engine.process_frame(packet.image.as_ref()) {
                    *worker_boxes.lock().unwrap() = output.boxes;
                    let _ = worker_boxes_tx.try_send(worker_boxes.lock().unwrap().clone());
                    *worker_processed.lock().unwrap() += 1;
                    let e2e_ms = packet.captured_at.elapsed().as_secs_f64() * 1000.0;

                    let mut stats = worker_summary.lock().unwrap();
                    stats.preprocess_ms.update(output.timings.preprocess_ms);
                    stats.inference_ms.update(output.timings.inference_ms);
                    stats.postprocess_ms.update(output.timings.postprocess_ms);
                    stats.total_ms.update(output.timings.total_ms);
                    stats.e2e_ms.update(e2e_ms);
                }

                ai_frames += 1;
                let elapsed = ai_timer.elapsed().as_secs_f64();
                if elapsed >= 1.0 {
                    let ai_fps = ai_frames as f64 / elapsed;
                    worker_summary.lock().unwrap().ai_fps.update(ai_fps);
                    ai_frames = 0;
                    ai_timer = Instant::now();
                }
            }
        }
    });

    println!(">> HỆ THỐNG ĐÃ SẴN SÀNG! ĐANG CHẠY REAL-TIME (Tối ưu Multi-Threading & NMS)...");

    let mut frame_counter: u32 = 0;
    let mut fps_timer = Instant::now();
    let mut stats_timer = Instant::now();
    let mut frame_pacer = Instant::now();
    let mut sys = System::new();
    let pid = sysinfo::get_current_pid().expect("Lỗi: Không lấy được PID hiện tại!");

    let render_running = running.clone();
    let render_thread = thread::spawn(move || {
        let mut last_boxes: Vec<types::BoundingBox> = Vec::new();
        let mut display_buffer: Vec<u32> = vec![0; width * height];
        let mut window = Window::new(
            "Giám Sát An Toàn - Edge AI (Bấm ESC để tắt)",
            width,
            height,
            WindowOptions::default(),
        )
        .expect("Lỗi: Không thể tạo cửa sổ!");
        window.limit_update_rate(Some(Duration::from_micros(16_600)));

        while render_running.load(Ordering::Relaxed) && window.is_open() {
            while let Ok(boxes) = boxes_rx.try_recv() {
                last_boxes = boxes;
            }

            let frame = match render_rx.recv_timeout(Duration::from_millis(5)) {
                Ok(packet) => packet,
                Err(_) => {
                    if window.is_key_down(Key::Escape) {
                        render_running.store(false, Ordering::Relaxed);
                    }
                    continue;
                }
            };

            visualizer::fill_display_buffer_with_boxes(
                frame.image.as_ref(),
                &last_boxes,
                &mut display_buffer,
            );
            window
                .update_with_buffer(&display_buffer, width, height)
                .unwrap();

            if window.is_key_down(Key::Escape) {
                render_running.store(false, Ordering::Relaxed);
            }
        }

        render_running.store(false, Ordering::Relaxed);
    });

    // VÒNG LẶP CHÍNH (Producer/Camera)
    let mut frame_index: u64 = 0;
    while running.load(Ordering::Relaxed) {
        // BƯỚC A: Chụp ảnh từ Camera
        if let Ok(image_rgb) = edge_cam.capture_frame() {
            let captured_at = Instant::now();
            let mut shared_image = if let Ok(buffer) = pool_rx.try_recv() {
                buffer
            } else {
                Arc::new(ImageBuffer::new(width as u32, height as u32))
            };
            let can_reuse = Arc::get_mut(&mut shared_image);
            if let Some(buffer) = can_reuse {
                let src = image_rgb.as_raw();
                let dst = buffer.as_flat_samples_mut().samples;
                if dst.len() == src.len() {
                    dst.copy_from_slice(src);
                } else {
                    *buffer = ImageBuffer::from_raw(width as u32, height as u32, src.to_vec())
                        .expect("Lỗi: Không thể tạo buffer ảnh");
                }
            } else {
                shared_image = Arc::new(image_rgb);
            }
            // Đẩy ảnh mới nhất vào Share State (bỏ khung cũ nếu tồn tại)
            *latest_frame.lock().unwrap() = Some(FramePacket {
                image: PooledFrame {
                    image: Arc::clone(&shared_image),
                    return_tx: pool_tx.clone(),
                },
                captured_at,
            });
            // Cố gắng đánh thức Worker (Bỏ qua nếu kênh full, Worker vẫn đang mải chạy)
            frame_index += 1;
            if frame_index % 12 == 0 {
                if notify_tx.try_send(()).is_err() {
                    *dropped_frames.lock().unwrap() += 1;
                }
            } else {
                *skipped_frames.lock().unwrap() += 1;
            }

            let _ = render_tx.try_send(FramePacket {
                image: PooledFrame {
                    image: shared_image,
                    return_tx: pool_tx.clone(),
                },
                captured_at,
            });

            frame_counter += 1;
            let fps_elapsed = fps_timer.elapsed().as_secs_f64();
            if fps_elapsed >= 1.0 {
                let camera_fps = frame_counter as f64 / fps_elapsed;
                summary.lock().unwrap().camera_fps.update(camera_fps);
                frame_counter = 0;
                fps_timer = Instant::now();
            }

            if stats_timer.elapsed() >= Duration::from_secs(1) {
                sys.refresh_process(pid);
                if let Some(process) = sys.process(pid) {
                    let rss_mb = process.memory() as f64 / (1024.0 * 1024.0);
                    let cpu = process.cpu_usage();
                    let mut stats = summary.lock().unwrap();
                    stats.rss_mb.update(rss_mb);
                    stats.cpu_pct.update(cpu as f64);
                }
                stats_timer = Instant::now();
            }

            let target_frame_ms = 1000.0 / TARGET_CAMERA_FPS;
            let elapsed_ms = frame_pacer.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms < target_frame_ms {
                thread::sleep(Duration::from_millis((target_frame_ms - elapsed_ms) as u64));
            }
            frame_pacer = Instant::now();
        } else {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    println!("Đã tắt luồng video an toàn!");
    running.store(false, Ordering::Relaxed);
    let _ = render_thread.join();

    let stats = summary.lock().unwrap().clone();
    println!("\n=== TÓM TẮT HIỆU NĂNG (avg/min/max) ===");
    println!(
        "Camera FPS: {:.2}/{:.2}/{:.2}",
        stats.camera_fps.avg(),
        stats.camera_fps.min,
        stats.camera_fps.max
    );
    println!(
        "AI FPS: {:.2}/{:.2}/{:.2}",
        stats.ai_fps.avg(),
        stats.ai_fps.min,
        stats.ai_fps.max
    );
    println!("\n");
    println!(
        "Latency preprocess: {:.2}/{:.2}/{:.2} (ms)",
        stats.preprocess_ms.avg(),
        stats.preprocess_ms.min,
        stats.preprocess_ms.max
    );
    println!(
        "Latency inference: {:.2}/{:.2}/{:.2} (ms)",
        stats.inference_ms.avg(),
        stats.inference_ms.min,
        stats.inference_ms.max
    );
    println!(
        "Latency postprocess: {:.2}/{:.2}/{:.2} (ms)",
        stats.postprocess_ms.avg(),
        stats.postprocess_ms.min,
        stats.postprocess_ms.max
    );
    println!(
        "Latency total: {:.2}/{:.2}/{:.2} (ms)",
        stats.total_ms.avg(),
        stats.total_ms.min,
        stats.total_ms.max
    );
    println!(
        "Latency end-to-end: {:.2}/{:.2}/{:.2} (ms)",
        stats.e2e_ms.avg(),
        stats.e2e_ms.min,
        stats.e2e_ms.max
    );
    println!("\n");
    println!(
        "RSS: {:.2}/{:.2}/{:.2} (MB)",
        stats.rss_mb.avg(),
        stats.rss_mb.min,
        stats.rss_mb.max
    );
    println!(
        "CPU: {:.2}/{:.2}/{:.2} (%)",
        stats.cpu_pct.avg(),
        stats.cpu_pct.min,
        stats.cpu_pct.max
    );
    println!("\n");
    println!(
        "Frames processed: {} | Frames dropped: {} | Frames skipped: {}",
        *processed_frames.lock().unwrap(),
        *dropped_frames.lock().unwrap(),
        *skipped_frames.lock().unwrap()
    );
    println!("\n=== KẾT THÚC CHƯƠNG TRÌNH ===");
}
