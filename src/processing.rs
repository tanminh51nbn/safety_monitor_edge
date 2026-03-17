use fast_image_resize as fir;
use image::{ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView3};
use rayon::prelude::*;
use std::num::NonZeroU32;

use crate::types::BoundingBox;

pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_w: u32,
    pub pad_h: u32,
}

pub struct PreprocessOutput {
    pub tensor: Array4<f32>,
    pub meta: LetterboxMeta,
}

pub fn preprocess(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    input_w: u32,
    input_h: u32,
) -> PreprocessOutput {
    let orig_w_u32 = img.width();
    let orig_h_u32 = img.height();
    let orig_w = orig_w_u32 as f32;
    let orig_h = orig_h_u32 as f32;
    let target_w = input_w as f32;
    let target_h = input_h as f32;
    let scale = (target_w / orig_w).min(target_h / orig_h);

    let new_unpad_w = (orig_w * scale).round() as u32;
    let new_unpad_h = (orig_h * scale).round() as u32;

    let src_raw = img.as_raw().to_vec();
    let src_image = fir::Image::from_vec_u8(
        NonZeroU32::new(orig_w_u32).expect("Lỗi: Chiều rộng ảnh nguồn không hợp lệ"),
        NonZeroU32::new(orig_h_u32).expect("Lỗi: Chiều cao ảnh nguồn không hợp lệ"),
        src_raw,
        fir::PixelType::U8x3,
    )
    .expect("Lỗi: Không thể tạo buffer ảnh nguồn");
    let mut dst_image = fir::Image::new(
        NonZeroU32::new(new_unpad_w).expect("Lỗi: Chiều rộng ảnh resize không hợp lệ"),
        NonZeroU32::new(new_unpad_h).expect("Lỗi: Chiều cao ảnh resize không hợp lệ"),
        fir::PixelType::U8x3,
    );
    let mut resizer = fir::Resizer::new(fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom));
    resizer
        .resize(&src_image.view(), &mut dst_image.view_mut())
        .expect("Lỗi: Resize thất bại");
    let resized_data = dst_image.buffer();

    let pad_w = (input_w - new_unpad_w) / 2;
    let pad_h = (input_h - new_unpad_h) / 2;

    let mut input_tensor =
        Array4::<f32>::from_elem((1, 3, input_h as usize, input_w as usize), 114.0 / 255.0);

    let resized_w = new_unpad_w as usize;
    let resized_h = new_unpad_h as usize;
    let input_w = input_w as usize;
    let input_h = input_h as usize;
    let pad_w_usize = pad_w as usize;
    let pad_h_usize = pad_h as usize;
    let tensor_slice = input_tensor
        .as_slice_mut()
        .expect("Lỗi: Tensor không liên tục trong bộ nhớ");
    let hw = input_h * input_w;
    let data = resized_data;
    let row_stride = resized_w * 3;

    let (r_channel, rest) = tensor_slice.split_at_mut(hw);
    let (g_channel, b_channel) = rest.split_at_mut(hw);
    let valid_y_start = pad_h_usize;
    let valid_y_end = pad_h_usize + resized_h;

    r_channel
        .par_chunks_mut(input_w)
        .zip(g_channel.par_chunks_mut(input_w))
        .zip(b_channel.par_chunks_mut(input_w))
        .enumerate()
        .for_each(|(y, ((r_row, g_row), b_row))| {
            if y < valid_y_start || y >= valid_y_end {
                return;
            }
            let src_y = y - valid_y_start;
            let src_row = &data[src_y * row_stride..(src_y + 1) * row_stride];
            for x in 0..resized_w {
                let src_idx = x * 3;
                let dst_idx = pad_w_usize + x;
                r_row[dst_idx] = src_row[src_idx] as f32 / 255.0;
                g_row[dst_idx] = src_row[src_idx + 1] as f32 / 255.0;
                b_row[dst_idx] = src_row[src_idx + 2] as f32 / 255.0;
            }
        });

    PreprocessOutput {
        tensor: input_tensor,
        meta: LetterboxMeta {
            scale,
            pad_w,
            pad_h,
        },
    }
}

pub fn postprocess(
    output_view: &ArrayView3<f32>,
    meta: &LetterboxMeta,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Vec<BoundingBox> {
    let mut boxes = Vec::new();
    let num_anchors = output_view.shape()[2];

    for i in 0..num_anchors {
        let score_helmet = output_view[[0, 4, i]];
        let score_no_helmet = output_view[[0, 5, i]];

        let (class_id, max_score) = if score_helmet > score_no_helmet {
            (0, score_helmet)
        } else {
            (1, score_no_helmet)
        };

        if max_score > conf_threshold {
            let cx = output_view[[0, 0, i]];
            let cy = output_view[[0, 1, i]];
            let w = output_view[[0, 2, i]];
            let h = output_view[[0, 3, i]];

            let real_cx = (cx - meta.pad_w as f32) / meta.scale;
            let real_cy = (cy - meta.pad_h as f32) / meta.scale;
            let real_w = w / meta.scale;
            let real_h = h / meta.scale;

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

    apply_nms(boxes, iou_threshold)
}

fn apply_nms(mut boxes: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected_boxes: Vec<BoundingBox> = Vec::new();

    for current_box in boxes {
        let mut keep = true;

        for selected_box in &selected_boxes {
            if current_box.class_id != selected_box.class_id {
                continue;
            }

            let iou = calculate_iou(&current_box, selected_box);

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

fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1 = box1.x.max(box2.x);
    let y1 = box1.y.max(box2.y);
    let x2 = (box1.x + box1.width).min(box2.x + box2.width);
    let y2 = (box1.y + box1.height).min(box2.y + box2.height);

    let intersection_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);

    let box1_area = box1.width * box1.height;
    let box2_area = box2.width * box2.height;

    let union_area = box1_area + box2_area - intersection_area;

    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}
