use image::{ImageBuffer, Rgb};
use std::error::Error;

// Lớp linh hoạt hỗ trợ tất cả các thể loại Stream hình ảnh
pub trait VideoProvider {
    fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>>;
}

// Xưởng sinh luồng Video (Factory)
pub fn create_camera(source: &str) -> Result<Box<dyn VideoProvider>, Box<dyn Error>> {
    if source.starts_with("rtsp://") || source.starts_with("http://") {
        #[cfg(feature = "rtsp")]
        {
            return Ok(Box::new(RtspCamera::new(source)?));
        }
        #[cfg(not(feature = "rtsp"))]
        {
            return Err("Tính năng RTSP & OpenCV chưa được bật. Hãy biên dịch bằng 'cargo run --features rtsp'.\n Hoặc quay về chạy Camera cổng USB bằng cách đặt CAMERA_SOURCE=0".into());
        }
    }

    // Default to USB Camera
    let index = source.parse::<u32>().unwrap_or(0);
    Ok(Box::new(UsbCamera::new(index)?))
}

pub struct UsbCamera {
    cam: nokhwa::Camera,
}

impl UsbCamera {
    pub fn new(camera_index: u32) -> Result<Self, Box<dyn Error>> {
        use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
        use nokhwa::pixel_format::RgbFormat;
        
        let index = CameraIndex::Index(camera_index);
        let resolutions = [Resolution::new(640, 360), Resolution::new(640, 480)];
        let formats = [FrameFormat::YUYV, FrameFormat::MJPEG];

        for req_res in resolutions {
            for frame_format in formats {
                let target_format = CameraFormat::new(req_res, frame_format, 30);
                let format =
                    RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(target_format));
                if let Ok(mut cam) = nokhwa::Camera::new(index.clone(), format) {
                    if cam.open_stream().is_ok() {
                        println!(
                            ">> Camera USB format selected: {}x{} @ {}fps ({:?})",
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
        Err("Lỗi: Không thể khởi tạo Camera USB. Bạn quên cắm vào máy tính chăng?".into())
    }
}

impl VideoProvider for UsbCamera {
    fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        use nokhwa::pixel_format::RgbFormat;
        let frame = self.cam.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let fixed_image = image::imageops::flip_horizontal(&decoded); // Gương mặt đối kháng
        Ok(fixed_image)
    }
}

#[cfg(feature = "rtsp")]
pub struct RtspCamera {
    cap: opencv::videoio::VideoCapture,
}

#[cfg(feature = "rtsp")]
impl RtspCamera {
    pub fn new(url: &str) -> Result<Self, Box<dyn Error>> {
        use opencv::videoio::{VideoCapture, CAP_ANY};
        let cap = VideoCapture::from_file(url, CAP_ANY)?;
        if !cap.is_opened()? {
            return Err("Không thể bóc tách luồng RTSP (Xem lại mật khẩu Camera hoặc Check IP LAN)".into());
        }
        println!(">> Khởi tạo IP Camera / RTSP Link hoàn tất!");
        Ok(Self { cap })
    }
}

#[cfg(feature = "rtsp")]
impl VideoProvider for RtspCamera {
    fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        use opencv::prelude::*;
        let mut frame = opencv::core::Mat::default();
        self.cap.read(&mut frame)?;
        if frame.empty() {
            return Err("Bị chặn/Rớt mạng luồng RTSP stream".into());
        }
        
        // Hoán đổi BGR (OpenCV) Sang RGB (Ảnh thô cho Inference)
        let mut rgb_frame = opencv::core::Mat::default();
        opencv::imgproc::cvt_color(&frame, &mut rgb_frame, opencv::imgproc::COLOR_BGR2RGB, 0)?;
        let size = rgb_frame.size()?;
        let width = size.width as u32;
        let height = size.height as u32;
        let data = rgb_frame.data_bytes()?.to_vec();
        
        // Ép kiểu Mat của thư viện C++ sang bộ đệm bộ nhớ RUST Buffer
        let img = ImageBuffer::from_raw(width, height, data).ok_or("Lỗi ép kiểu Byte Buffer RTSP")?;
        Ok(img)
    }
}
