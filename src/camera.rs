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
        let req_res = Resolution::new(1280, 720);
        let target_format = CameraFormat::new(req_res, FrameFormat::MJPEG, 30);
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(target_format));
        let mut cam = Camera::new(index, format)?;
        cam.open_stream()?;

        Ok(Self { cam })
    }

    pub fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let frame = self.cam.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let fixed_image = image::imageops::flip_horizontal(&decoded);

        Ok(fixed_image)
    }
}
