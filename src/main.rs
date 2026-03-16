// File: src/main.rs
mod types;
mod camera;
mod engine; // Gọi module Não bộ AI

use minifb::{Key, Window, WindowOptions};
use std::time::Duration;
use image::Rgb;
use imageproc::rect::Rect;
use imageproc::drawing::draw_hollow_rect_mut;

use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    println!("=== HỆ THỐNG GIÁM SÁT AN TOÀN EDGE AI ===");
    
    // 1. KHỞI TẠO MẮT (CAMERA)
    println!("Đang khởi động Camera...");
    let mut edge_cam = camera::EdgeCamera::new().expect("Lỗi: Không thể khởi tạo Camera!");

    let first_frame = edge_cam.capture_frame().expect("Lỗi: Không thể chụp ảnh mồi!");
    let width = first_frame.width() as usize;   // Thường là 1280
    let height = first_frame.height() as usize; // Thường là 720
    
    // 2. KHỞI TẠO ĐA LUỒNG (LIFO Producer-Consumer Pattern)
    println!("Đang tạo Kiến trúc Đa Luồng cho AI Engine...");
    let latest_frame = Arc::new(Mutex::new(None));
    let latest_boxes = Arc::new(Mutex::new(Vec::new()));
    
    let (notify_tx, notify_rx) = crossbeam_channel::bounded(1);
    
    let worker_frame = latest_frame.clone();
    let worker_boxes = latest_boxes.clone();

    // Khởi chạy Worker Thread (Chỉ làm nhiệm vụ AI ngầm)
    thread::spawn(move || {
        println!("Worker (AI): Đang nạp mô hình AI (best.onnx) vào RAM...");
        let mut ai_engine = engine::AIEngine::new("models/best.onnx")
            .expect("Lỗi: Không tìm thấy file mô hình tại 'models/best.onnx' hoặc lỗi nạp ORT!");
        println!("Worker (AI): Đã sẵn sàng suy luận!");

        // Vòng lặp suy luận chạy độc lập
        while let Ok(_) = notify_rx.recv() {
            // Lấy ảnh mới nhất ra khỏi bộ nhớ chung (đồng thời dọn trống luôn - LIFO)
            let frame = {
                worker_frame.lock().unwrap().take()
            };
            
            if let Some(img) = frame {
                // Chạy AI trên ảnh, không lo bị chặn UI
                if let Ok(boxes) = ai_engine.process_frame(&img) {
                    *worker_boxes.lock().unwrap() = boxes;
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
    ).expect("Lỗi: Không thể tạo cửa sổ!");
    window.limit_update_rate(Some(Duration::from_micros(16_600))); // ~60 FPS

    let mut display_buffer: Vec<u32> = vec![0; width * height];

    println!(">> HỆ THỐNG ĐÃ SẴN SÀNG! ĐANG CHẠY REAL-TIME (Tối ưu Multi-Threading & NMS)...");

    // VÒNG LẶP CHÍNH (UI THREAD) - Không bao giờ bị chặn
    while window.is_open() && !window.is_key_down(Key::Escape) {
        
        // BƯỚC A: Chụp ảnh từ Camera
        if let Ok(mut image_rgb) = edge_cam.capture_frame() {
            
            // Đẩy ảnh mới nhất vào Share State (bỏ khung cũ nếu tồn tại)
            *latest_frame.lock().unwrap() = Some(image_rgb.clone());
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
                    let t_rect = Rect::at(rx - thickness, ry - thickness)
                        .of_size(real_w + (thickness as u32)*2, real_h + (thickness as u32)*2);
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
            window.update_with_buffer(&display_buffer, width, height).unwrap();
        }
    }
    
    println!("Đã tắt luồng video an toàn!");
}