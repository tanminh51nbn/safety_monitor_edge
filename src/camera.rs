use image::{ImageBuffer, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::Camera;
use std::error::Error;

pub struct EdgeCamera {
    cam: Camera,
}

impl EdgeCamera {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let index = CameraIndex::Index(0);
        let resolutions = [Resolution::new(640, 360), Resolution::new(640, 480)];
        let formats = [FrameFormat::YUYV, FrameFormat::MJPEG];

        for req_res in resolutions {
            for frame_format in formats {
                let target_format = CameraFormat::new(req_res, frame_format, 30);
                let format =
                    RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(target_format));
                if let Ok(mut cam) = Camera::new(index.clone(), format) {
                    if cam.open_stream().is_ok() {
                        println!(
                            "Camera format selected: {}x{} @ {}fps ({:?})",
                            req_res.width(),
                            req_res.height(),
                            30,
                            frame_format
                        );
                        return Ok(Self { cam });
                    }
                }
            }
        }

        Err("Lỗi: Không thể khởi tạo Camera với độ phân giải yêu cầu".into())
    }

    pub fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let frame = self.cam.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let fixed_image = image::imageops::flip_horizontal(&decoded);

        Ok(fixed_image)
    }
}
