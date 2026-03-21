use std::time::{Duration, Instant};

pub struct AlarmStateMachine {
    violation_count: u32,
    pub is_alarming: bool,
    alarm_start: Option<Instant>,
    cooldown_duration: Duration,
    debounce_frames: u32,
}

impl AlarmStateMachine {
    pub fn new(debounce_frames: u32, cooldown_secs: u64) -> Self {
        Self {
            violation_count: 0,
            is_alarming: false,
            alarm_start: None,
            cooldown_duration: Duration::from_secs(cooldown_secs),
            debounce_frames,
        }
    }

    pub fn update(&mut self, has_violation: bool) -> bool {
        let mut trigger_snapshot = false;

        if has_violation {
            self.violation_count += 1;
            if self.violation_count >= self.debounce_frames && !self.is_alarming {
                self.is_alarming = true;
                self.alarm_start = Some(Instant::now());
                trigger_snapshot = true; // Chớp đúng khung đầu tiên vi phạm
            }
            if self.is_alarming {
                // Gia hạn Cooldown liên tục chừng nào còn thấy vi phạm
                self.alarm_start = Some(Instant::now());
            }
        } else {
            self.violation_count = 0;
            if self.is_alarming {
                if let Some(start) = self.alarm_start {
                    if start.elapsed() >= self.cooldown_duration {
                        self.is_alarming = false; // Tắt còi sau khi hết Cooldown
                    }
                }
            }
        }

        trigger_snapshot
    }
}
