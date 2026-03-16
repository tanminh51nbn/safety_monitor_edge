// File: src/engine.rs
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel; // Đã cập nhật đường dẫn API mới của ort 2.0
use ort::inputs;
use ndarray::Array4;
use image::{ImageBuffer, Rgb, imageops::FilterType};
use std::error::Error;
use crate::types::BoundingBox;

pub struct AIEngine {
    session: Session,
}

impl AIEngine {
    // 1. NẠP MÔ HÌNH (Load Model)
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        // Cấu hình ORT Session để tận dụng tối đa sức mạnh CPU
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)? // Dùng trực tiếp Enum đã cập nhật
            .with_intra_threads(4)? // Sử dụng 4 luồng CPU để chạy suy luận
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    // Hàm thực thi toàn bộ luồng từ ảnh thô -> Danh sách Khung nhận diện
    pub fn process_frame(&mut self, img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<BoundingBox>, Box<dyn Error>> {
        
        // 2. TIỀN XỬ LÝ (Preprocessing) TỐI ƯU SỬ DỤNG KỸ THUẬT LETTERBOX PADDING
        // 2.1. Tính toán tỷ lệ bù viền để tránh bị vỡ Tỷ lệ Hình ảnh (Aspect Ratio Distortion)
        let orig_w = img.width() as f32;
        let orig_h = img.height() as f32;
        let target_size = 640.0;
        let scale = (target_size / orig_w).min(target_size / orig_h);
        
        let new_unpad_w = (orig_w * scale).round() as u32;
        let new_unpad_h = (orig_h * scale).round() as u32;

        let resized_img = image::imageops::resize(img, new_unpad_w, new_unpad_h, FilterType::Triangle);

        // Tính khoảng cách viền đen cần bù
        let pad_w = (target_size as u32 - new_unpad_w) / 2;
        let pad_h = (target_size as u32 - new_unpad_h) / 2;

        // 2.2. Khởi tạo mảng Tensor (1, 3, 640, 640) có giá trị nền là Màu xám (YOLO chuẩn cần là RGB 114)
        let mut input_tensor = Array4::<f32>::from_elem((1, 3, 640, 640), 114.0 / 255.0);
        
        // 2.3. Chép ảnh đã Resize vào chính giữa nền xám đó (HWC -> CHW / Chuẩn hóa 0.0-1.0)
        for (x, y, pixel) in resized_img.enumerate_pixels() {
            let px = x as usize + pad_w as usize;
            let py = y as usize + pad_h as usize;
            input_tensor[[0, 0, py, px]] = pixel[0] as f32 / 255.0; // Kênh Đỏ (R)
            input_tensor[[0, 1, py, px]] = pixel[1] as f32 / 255.0; // Kênh Xanh lá (G)
            input_tensor[[0, 2, py, px]] = pixel[2] as f32 / 255.0; // Kênh Xanh dương (B)
        }

        // 3. CHẠY SUY LUẬN (Inference)
        // [FIX CHÍNH] Tạo Tensor chuẩn của ort 2.0 một cách tường minh từ mảng ndarray
        let input_ort = ort::value::Tensor::from_array(input_tensor)?;
        
        // Chèn input_ort vào macro inputs! (macro trả về Vec, không dùng dấu `?` trong ruột macro)
        let outputs = self.session.run(inputs!["images" => input_ort])?;
        
        // Trích xuất kết quả từ cổng đầu ra "output0"
        // Ở bản ort 2.0, hàm này trả về Tuple: (Shape, Dữ liệu thô &[f32])
        let (_, raw_data) = outputs["output0"].try_extract_tensor::<f32>()?;
        
        // Tái tạo lại mảng 3 chiều (1 Batch, 6 Channel, 8400 Anchor) từ dữ liệu thô
        let output_view = ndarray::ArrayView3::from_shape((1, 6, 8400), raw_data)
            .expect("Lỗi: Không thể ép kiểu dữ liệu đầu ra về đúng kích thước của YOLOv8");
        
        // 4. HẬU XỬ LÝ (Post-processing)
        let mut boxes = Vec::new();
        let num_anchors = output_view.shape()[2]; // 8400

        for i in 0..num_anchors {
            let score_helmet = output_view[[0, 4, i]];
            let score_no_helmet = output_view[[0, 5, i]];
            
            // Tìm Class có điểm cao nhất
            let (class_id, max_score) = if score_helmet > score_no_helmet {
                (0, score_helmet)
            } else {
                (1, score_no_helmet)
            };

            // Sửa NGƯỠNG ĐỘ TIN CẬY: Hạ từ 0.5 xuống 0.3 (Tiêu chuẩn mặc định chạy Camera thực tế của YOLOv8)
            // Nhờ cài NMS (Gộp khung), chúng ta có thể xài ngưỡng thấp mà không lo bị loạn nhiều khung giả!
            if max_score > 0.3 {
                let cx = output_view[[0, 0, i]]; // Tọa độ tâm X (trên canvas 640x640)
                let cy = output_view[[0, 1, i]]; // Tọa độ tâm Y 
                let w = output_view[[0, 2, i]];  // Chiều rộng
                let h = output_view[[0, 3, i]];  // Chiều cao

                // 4.1 TRẢ LẠI TỌA ĐỘ ẢNH GỐC: Vứt bỏ phần viền và nhân ngược tỷ lệ Scale
                let real_cx = (cx - pad_w as f32) / scale;
                let real_cy = (cy - pad_h as f32) / scale;
                let real_w = w / scale;
                let real_h = h / scale;

                boxes.push(BoundingBox {
                    x: real_cx - real_w / 2.0, 
                    y: real_cy - real_h / 2.0,
                    width: real_w,
                    height: real_h,
                    class_id,
                    confidence: max_score,
                });
            }
        }

        // 5. NMS (Non-Maximum Suppression) để gộp các khung hình đè nhau
        let final_boxes = apply_nms(boxes, 0.45); // Ngưỡng IoU: Khung nào đè lên khung kia >= 45% sẽ bị loại

        Ok(final_boxes)
    }
}

// --- CÁC HÀM TIỆN ÍCH CHO VIỆC LỌC KHUNG HÌNH ---

/// Hàm thực thi Thuật toán NMS (Non-Maximum Suppression)
fn apply_nms(mut boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    // B1: Sắp xếp danh sách hộp giảm dần theo độ tự tin (Confidence)
    boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected_boxes: Vec<BoundingBox> = Vec::new();

    // B2: Duyệt qua từng hộp
    for current_box in boxes {
        let mut keep = true;
        
        for selected_box in &selected_boxes {
            // Chỉ so sánh các khung có cùng nhãn (Ví dụ chỉ so sánh Cùng Helmet với nhau)
            if current_box.class_id != selected_box.class_id {
                continue;
            }

            // Tính độ giao nhau (IoU)
            let iou = calculate_iou(&current_box, selected_box);
            
            // B3: Nếu độ giao nhau lớn hơn ngưỡng quy định thì vứt bỏ khung hiện tại
            if iou > iou_threshold {
                keep = false;
                break;
            }
        }
        
        if keep {
            selected_boxes.push(current_box);
        }
    }

    selected_boxes
}

/// Hàm tính toán độ đè lên nhau giữa 2 khung hình (Intersection over Union)
fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    // 1. Tìm hình chữ nhật giao nhau
    let x1 = box1.x.max(box2.x);
    let y1 = box1.y.max(box2.y);
    let x2 = (box1.x + box1.width).min(box2.x + box2.width);
    let y2 = (box1.y + box1.height).min(box2.y + box2.height);

    // Tính diện tích phần giao (Intersection)
    let intersection_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    
    // 2. Tính diện tích 2 hình chữ nhật ban đầu
    let box1_area = box1.width * box1.height;
    let box2_area = box2.width * box2.height;
    
    // 3. Tính diện tích tổng (Union)
    let union_area = box1_area + box2_area - intersection_area;
    
    // 4. Tính tỉ lệ IoU
    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}