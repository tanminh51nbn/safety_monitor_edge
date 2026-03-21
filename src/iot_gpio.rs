#[cfg(target_os = "linux")]
use sysfs_gpio::{Direction, Pin};

pub struct AlarmRelay {
    #[cfg(target_os = "linux")]
    pin: Option<Pin>,
    last_state: bool,
}

impl AlarmRelay {
    pub fn new(pin_num: u64) -> Self {
        #[cfg(target_os = "linux")]
        {
            let pin = Pin::new(pin_num);
            // Cố gắng mở kênh xuất tín hiệu vào Kernel Linux OS
            let pin_opt = if pin.with_exported(|| {
                pin.set_direction(Direction::Out)?; // Rút dây ra Output
                Ok(())
            }).is_ok() {
                println!(">> [GPIO STARTUP] Đã link vật lý Pin {} với bảng điều khiển Relays Còi rú.", pin_num);
                Some(pin)
            } else {
                println!(">> [GPIO WARN] Pin {} không khả dụng. Từ chối quyền hoặc cắm sai.", pin_num);
                None
            };
            Self { pin: pin_opt, last_state: false }
        }
        #[cfg(not(target_os = "linux"))]
        {
            println!(">> [GPIO SIMULATION] Máy hiện tại là Window. Sẽ chỉ giả lập Console text Out cho Pin {}.", pin_num);
            Self { last_state: false }
        }
    }

    pub fn update(&mut self, is_alarming: bool) {
        if self.last_state == is_alarming {
            return; // Đã chốt logic thì không đổi liên tục gây cháy Mạch
        }
        
        self.last_state = is_alarming;
        
        #[cfg(target_os = "linux")]
        {
            if let Some(pin) = &self.pin {
                let val = if is_alarming { 1 } else { 0 };
                let _ = pin.set_value(val); // Sút áp lực điện cực cao kích nổ Relay 220V!!
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            if is_alarming {
                println!(">> 🚨 [HARDWARE PIN HIGH] CÒI VẬT LÝ NHÀ XƯỞNG BẬT !!!!");
            } else {
                println!(">> 🔇 [HARDWARE PIN LOW] Còi ở xưởng đã ngắt.");
            }
        }
    }
}
