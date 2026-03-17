use image::{ImageBuffer, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::types::BoundingBox;

pub fn draw_boxes(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, boxes: &[BoundingBox]) {
    for bbox in boxes {
        let color = if bbox.class_id == 0 {
            Rgb([0, 255, 0])
        } else {
            Rgb([255, 0, 0])
        };

        let real_x = bbox.x as i32;
        let real_y = bbox.y as i32;
        let real_w = bbox.width as u32;
        let real_h = bbox.height as u32;

        let rx = real_x.max(0);
        let ry = real_y.max(0);

        for thickness in 0..3 {
            let t_rect = Rect::at(rx - thickness, ry - thickness).of_size(
                real_w + (thickness as u32) * 2,
                real_h + (thickness as u32) * 2,
            );
            draw_hollow_rect_mut(image, t_rect, color);
        }
    }
}

pub fn fill_display_buffer(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, buffer: &mut [u32]) {
    for (i, pixel) in image.pixels().enumerate() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        buffer[i] = (r << 16) | (g << 8) | b;
    }
}
