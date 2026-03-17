use image::{imageops::FilterType, ImageBuffer, Rgb};
use ndarray::{Array4, ArrayView3};

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
    let orig_w = img.width() as f32;
    let orig_h = img.height() as f32;
    let target_w = input_w as f32;
    let target_h = input_h as f32;
    let scale = (target_w / orig_w).min(target_h / orig_h);

    let new_unpad_w = (orig_w * scale).round() as u32;
    let new_unpad_h = (orig_h * scale).round() as u32;

    let resized_img = image::imageops::resize(img, new_unpad_w, new_unpad_h, FilterType::Triangle);

    let pad_w = (input_w - new_unpad_w) / 2;
    let pad_h = (input_h - new_unpad_h) / 2;

    let mut input_tensor =
        Array4::<f32>::from_elem((1, 3, input_h as usize, input_w as usize), 114.0 / 255.0);

    for (x, y, pixel) in resized_img.enumerate_pixels() {
        let px = x as usize + pad_w as usize;
        let py = y as usize + pad_h as usize;
        input_tensor[[0, 0, py, px]] = pixel[0] as f32 / 255.0;
        input_tensor[[0, 1, py, px]] = pixel[1] as f32 / 255.0;
        input_tensor[[0, 2, py, px]] = pixel[2] as f32 / 255.0;
    }

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
