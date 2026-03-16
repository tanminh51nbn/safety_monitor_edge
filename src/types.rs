// File: src/types.rs
// Định nghĩa các cấu trúc dữ liệu dùng chung cho toàn dự án

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,         // Tọa độ góc trên bên trái (trục X)
    pub y: f32,         // Tọa độ góc trên bên trái (trục Y)
    pub width: f32,     // Chiều rộng khung
    pub height: f32,    // Chiều cao khung
    pub class_id: usize,// ID của nhãn (0: Helmet, 1: No_Helmet)
    pub confidence: f32,// Độ tự tin của AI (0.0 -> 1.0)
}

// Danh sách các nhãn dựa theo file ONNX của bạn (Opset 12)
// pub const CLASS_NAMES: [&str; 2] = ["Helmet", "No_Helmet"];