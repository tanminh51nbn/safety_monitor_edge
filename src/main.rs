// File: src/main.rs
mod camera;
mod engine;
mod types; // Gọi module Não bộ AI

use image::Rgb;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use sysinfo::System;

use std::sync::{Arc, Mutex};
use std::thread;

const MODEL_PATH: &str = "models/best_640x384.onnx";
const INPUT_W: u32 = 640;
const INPUT_H: u32 = 384;

#[derive(Clone)]
struct FramePacket {
    image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    captured_at: Instant,
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

    let (notify_tx, notify_rx) = crossbeam_channel::bounded(1);

    let worker_frame = latest_frame.clone();
    let worker_boxes = latest_boxes.clone();
    let worker_summary = summary.clone();

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
                if let Ok(output) = ai_engine.process_frame(&packet.image) {
                    *worker_boxes.lock().unwrap() = output.boxes;
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

    // 3. KHỞI TẠO GIAO DIỆN (UI)
    let mut window = Window::new(
        "Giám Sát An Toàn - Edge AI (Bấm ESC để tắt)",
        width,
        height,
        WindowOptions::default(),
    )
    .expect("Lỗi: Không thể tạo cửa sổ!");
    window.limit_update_rate(Some(Duration::from_micros(16_600))); // ~60 FPS

    let mut display_buffer: Vec<u32> = vec![0; width * height];

    println!(">> HỆ THỐNG ĐÃ SẴN SÀNG! ĐANG CHẠY REAL-TIME (Tối ưu Multi-Threading & NMS)...");

    let mut frame_counter: u32 = 0;
    let mut fps_timer = Instant::now();
    let mut stats_timer = Instant::now();
    let mut sys = System::new();
    let pid = sysinfo::get_current_pid().expect("Lỗi: Không lấy được PID hiện tại!");

    // VÒNG LẶP CHÍNH (UI THREAD) - Không bao giờ bị chặn
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // BƯỚC A: Chụp ảnh từ Camera
        if let Ok(mut image_rgb) = edge_cam.capture_frame() {
            let captured_at = Instant::now();
            // Đẩy ảnh mới nhất vào Share State (bỏ khung cũ nếu tồn tại)
            *latest_frame.lock().unwrap() = Some(FramePacket {
                image: image_rgb.clone(),
                captured_at,
            });
            // Cố gắng đánh thức Worker (Bỏ qua nếu kênh full, Worker vẫn đang mải chạy)
            let _ = notify_tx.try_send(());

            // Trích xuất kết quả nhận diện (Boxes) GẦN NHẤT
            let current_boxes = latest_boxes.lock().unwrap().clone();

            // BƯỚC C: Vẽ Bounding Box lên ảnh
            for bbox in current_boxes {
                // Chọn màu: Mũ bảo hiểm (0) -> Xanh lá | Không mũ (1) -> Đỏ
                let color = if bbox.class_id == 0 {
                    Rgb([0, 255, 0])
                } else {
                    Rgb([255, 0, 0])
                };

                // Lấy y nguyên tọa độ vì AIEngine (Letterbox) đã giải max ngược về đúng không gian thật
                let real_x = bbox.x as i32;
                let real_y = bbox.y as i32;
                let real_w = bbox.width as u32;
                let real_h = bbox.height as u32;

                // Giới hạn tọa độ để không bị văng lỗi khi vẽ tràn viền
                let rx = real_x.max(0);
                let ry = real_y.max(0);

                // Vẽ khung với độ dày 3 pixel (Bằng cách lặp vẽ 3 hình chữ nhật lồng nhau)
                for thickness in 0..3 {
                    let t_rect = Rect::at(rx - thickness, ry - thickness).of_size(
                        real_w + (thickness as u32) * 2,
                        real_h + (thickness as u32) * 2,
                    );
                    draw_hollow_rect_mut(&mut image_rgb, t_rect, color);
                }
            }

            // BƯỚC D: Thuật toán Bitwise ghép RGB (3 byte) thành u32 (4 byte) để đẩy lên màn hình
            for (i, pixel) in image_rgb.pixels().enumerate() {
                let r = pixel[0] as u32;
                let g = pixel[1] as u32;
                let b = pixel[2] as u32;
                display_buffer[i] = (r << 16) | (g << 8) | b;
            }

            // Render khung hình lên Window
            window
                .update_with_buffer(&display_buffer, width, height)
                .unwrap();

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
        }
    }

    println!("Đã tắt luồng video an toàn!");

    let stats = summary.lock().unwrap().clone();
    println!("=== TÓM TẮT HIỆU NĂNG ===");
    println!(
        "Camera FPS avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.camera_fps.avg(),
        stats.camera_fps.min,
        stats.camera_fps.max
    );
    println!(
        "AI FPS avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.ai_fps.avg(),
        stats.ai_fps.min,
        stats.ai_fps.max
    );
    println!(
        "Latency preprocess ms avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.preprocess_ms.avg(),
        stats.preprocess_ms.min,
        stats.preprocess_ms.max
    );
    println!(
        "Latency inference ms avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.inference_ms.avg(),
        stats.inference_ms.min,
        stats.inference_ms.max
    );
    println!(
        "Latency postprocess ms avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.postprocess_ms.avg(),
        stats.postprocess_ms.min,
        stats.postprocess_ms.max
    );
    println!(
        "Latency total ms avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.total_ms.avg(),
        stats.total_ms.min,
        stats.total_ms.max
    );
    println!(
        "Latency end-to-end ms avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.e2e_ms.avg(),
        stats.e2e_ms.min,
        stats.e2e_ms.max
    );
    println!(
        "RSS MB avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.rss_mb.avg(),
        stats.rss_mb.min,
        stats.rss_mb.max
    );
    println!(
        "CPU % avg/min/max: {:.2}/{:.2}/{:.2}",
        stats.cpu_pct.avg(),
        stats.cpu_pct.min,
        stats.cpu_pct.max
    );
}
