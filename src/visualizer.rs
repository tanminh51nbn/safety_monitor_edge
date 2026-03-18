use image::{ImageBuffer, Rgb};

use crate::types::BoundingBox;

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
) {
    fill_display_buffer(image, buffer);

    let width = image.width() as i32;
    let height = image.height() as i32;

    for bbox in boxes {
        let color = if bbox.class_id == 0 {
            0x00FF00
        } else {
            0xFF0000
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
    }
}
