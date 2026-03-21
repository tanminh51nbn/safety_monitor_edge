use image::{ImageBuffer, Rgb};

use crate::types::BoundingBox;

use rusttype::{Font, Scale, point};

pub fn fill_display_buffer(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, buffer: &mut [u32]) {
    for (i, pixel) in image.pixels().enumerate() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        buffer[i] = (r << 16) | (g << 8) | b;
    }
}

pub fn fill_display_buffer_with_boxes(
    image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    boxes: &[BoundingBox],
    buffer: &mut [u32],
    is_alarming: bool,
    font: Option<&Font>,
) {
    fill_display_buffer(image, buffer);

    let width = image.width() as i32;
    let height = image.height() as i32;

    for bbox in boxes {
        let (color, label) = if bbox.class_id == 0 {
            (0x00FF00, "Helmet")
        } else {
            (0xFF0000, "No Helmet")
        };

        let x = bbox.x.max(0.0) as i32;
        let y = bbox.y.max(0.0) as i32;
        let w = bbox.width.max(0.0) as i32;
        let h = bbox.height.max(0.0) as i32;

        if w == 0 || h == 0 {
            continue;
        }

        let x2 = (x + w).min(width - 1);
        let y2 = (y + h).min(height - 1);

        for t in 0..3 {
            let top = (y - t).max(0);
            let bottom = (y2 + t).min(height - 1);
            let left = (x - t).max(0);
            let right = (x2 + t).min(width - 1);

            for xx in left..=right {
                let top_idx = (top as usize) * (width as usize) + (xx as usize);
                let bottom_idx = (bottom as usize) * (width as usize) + (xx as usize);
                buffer[top_idx] = color;
                buffer[bottom_idx] = color;
            }

            for yy in top..=bottom {
                let left_idx = (yy as usize) * (width as usize) + (left as usize);
                let right_idx = (yy as usize) * (width as usize) + (right as usize);
                buffer[left_idx] = color;
                buffer[right_idx] = color;
            }
        }

        // Vẽ Text trực tiếp trên hộp (OSD)
        if let Some(f) = font {
            let text = format!("{} {:.0}%", label, bbox.confidence * 100.0);
            draw_text_to_buffer(buffer, width as usize, height as usize, &text, x, (y - 25).max(0), f, 24.0, color);
        }
    }

    // VẼ CẢNH BÁO ĐỎ VIỀN QUANH MÀN HÌNH NẾU ĐANG BÁO ĐỘNG
    if is_alarming {
        let border_thickness = 15;
        for y in 0..height {
            for x in 0..width {
                if x < border_thickness || x >= width - border_thickness || y < border_thickness || y >= height - border_thickness {
                    buffer[(y * width + x) as usize] = 0xFF0000;
                }
            }
        }
        
        // Vẽ chữ Báo động siêu to giữa màn hình trên
        if let Some(f) = font {
            draw_text_to_buffer(buffer, width as usize, height as usize, "! ALARM: VIOLATION DETECTED !", 20, 30, f, 45.0, 0xFF0000);
        }
    }
}

pub fn draw_text_to_buffer(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    text: &str,
    x: i32,
    y: i32,
    font: &Font,
    font_scale: f32,
    color_u32: u32,
) {
    let scale = Scale::uniform(font_scale);
    let v_metrics = font.v_metrics(scale);
    let offset = point(x as f32, y as f32 + v_metrics.ascent);

    for glyph in font.layout(text, scale, offset) {
        if let Some(bb) = glyph.pixel_bounding_box() {
            glyph.draw(|gx, gy, v| {
                if v > 0.5 {
                    let px = bb.min.x + gx as i32;
                    let py = bb.min.y + gy as i32;
                    if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                        let idx = (py as usize) * width + (px as usize);
                        buffer[idx] = color_u32;
                    }
                }
            });
        }
    }
}
