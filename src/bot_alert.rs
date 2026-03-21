use reqwest::blocking::multipart;

pub fn send_telegram_alert(bot_token: &str, chat_id: &str, image_path: &str, caption: &str) {
    if bot_token.is_empty() {
        return; // Bỏ qua nếu cấu hình rỗng
    }
    
    let url = format!("https://api.telegram.org/bot{}/sendPhoto", bot_token);
    
    // Gói HTTP Multipart-Data (Bao gồm File ảnh thực tế chụp tại RAM và Text chú thích)
    if let Ok(form) = multipart::Form::new()
        .text("chat_id", chat_id.to_string())
        .text("caption", caption.to_string())
        .file("photo", image_path)
    {
        println!(">> [-BOT-] Đang tải và gửi ảnh tới Telegram của Quản đốc...");
        let client = reqwest::blocking::Client::new();
        match client.post(&url).multipart(form).send() {
            Ok(resp) => {
                if resp.status().is_success() {
                    println!(">> [🚀 TELEGRAM ALARM SENT] ✅");
                } else {
                    println!(">> [Lỗi TELEGRAM] API Key hoặc Chat ID không chính xác (HTTP {:?})", resp.status());
                }
            },
            Err(e) => println!(">> Cảnh báo: Gửi Telegram thất bại do mất liên kết Wifi/Internet: {}", e),
        }
    }
}
