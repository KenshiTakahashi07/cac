import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

def custom_loss(y_true, y_pred):
    # Chỉ tính loss cho các box thực sự (không phải padding)
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    # Mở rộng mask để match với kích thước của y_true và y_pred
    mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
    # Tính MSE loss và áp dụng mask
    loss = tf.reduce_mean(tf.square(y_true - y_pred) * mask)
    return loss

# Load model đã train với custom loss function
model = load_model("traffic_sign_model_v1.h5", custom_objects={'custom_loss': custom_loss})

class_names = {
    0: 'Giới hạn tốc độ (20km/h)',
    1: 'Giới hạn tốc độ (30km/h)', 
    2: 'Giới hạn tốc độ (50km/h)', 
    3: 'Giới hạn tốc độ (60km/h)', 
    4: 'Giới hạn tốc độ (70km/h)', 
    5: 'Giới hạn tốc độ (80km/h)', 
    6: 'Hết giới hạn tốc độ (80km/h)', 
    7: 'Giới hạn tốc độ (100km/h)', 
    8: 'Giới hạn tốc độ (120km/h)', 
    9: 'Cấm vượt', 
    10: 'Cấm xe tải vượt', 
    11: 'Giao nhau với đường ưu tiên', 
    12: 'Đường ưu tiên', 
    13: 'Nhường đường', 
    14: 'Dừng lại', 
    15: 'Cấm đi', 
    16: 'Cấm xe tải', 
    17: 'Cấm đi ngược chiều', 
    18: 'Nguy hiểm', 
    19: 'Đoạn cong nguy hiểm bên trái', 
    20: 'Đoạn cong nguy hiểm bên phải', 
    21: 'Đoạn cong kép', 
    22: 'Đường gồ ghề', 
    23: 'Đường trơn trượt', 
    24: 'Đường thu hẹp về bên phải', 
    25: 'Công trường', 
    26: 'Đèn tín hiệu', 
    27: 'Người đi bộ', 
    28: 'Trẻ em', 
    29: 'Xe đạp', 
    30: 'Tuyết/Băng', 
    31: 'Động vật hoang dã', 
    32: 'Hết tất cả giới hạn tốc độ', 
    33: 'Bắt buộc rẽ phải', 
    34: 'Bắt buộc rẽ trái', 
    35: 'Bắt buộc đi thẳng', 
    36: 'Bắt buộc thẳng hoặc phải', 
    37: 'Bắt buộc thẳng hoặc trái', 
    38: 'Rẽ phải', 
    39: 'Rẽ trái', 
    40: 'Bắt buộc vòng xuyến', 
    41: 'Hết cấm vượt', 
    42: 'Hết cấm xe tải vượt'
}

def detect_traffic_signs(image):
    """Phát hiện vùng có khả năng là biển báo trong ảnh"""
    # Chuyển sang ảnh HSV để dễ dàng phát hiện màu sắc
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa range màu cho biển báo (đỏ và xanh dương)
    # Màu đỏ trong HSV có 2 range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # Màu xanh dương
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # Tạo mask cho từng màu
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Kết hợp các mask
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_blue)
    
    # Áp dụng morphology để loại bỏ nhiễu
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc và trả về các vùng có khả năng là biển báo
    sign_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Lọc bỏ các vùng quá nhỏ
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            # Kiểm tra tỷ lệ khung hình (biển báo thường có tỷ lệ gần 1:1)
            if 0.7 <= aspect_ratio <= 1.3:
                sign_regions.append((x, y, w, h))
    
    return sign_regions

def enhance_image(image):
    """Cải thiện chất lượng ảnh"""
    # Cân bằng histogram để cải thiện độ tương phản
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Giảm nhiễu
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    return enhanced

def preprocess_frame(frame, size=(50, 50)):
    """Tiền xử lý frame cho model"""
    # Cải thiện chất lượng ảnh
    enhanced = enhance_image(frame)
    
    # Phát hiện các vùng biển báo
    sign_regions = detect_traffic_signs(enhanced)
    
    if not sign_regions:
        return None, None
    
    # Lấy vùng biển báo lớn nhất
    largest_region = max(sign_regions, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_region
    
    # Cắt vùng biển báo
    sign = enhanced[y:y+h, x:x+w]
    
    # Resize về kích thước phù hợp với model
    sign_resized = cv2.resize(sign, size)
    
    # Chuẩn hóa ảnh
    sign_normalized = sign_resized.astype('float32') / 255.0
    
    return sign_normalized, largest_region

def put_vietnamese_text(img, text, position, font_size=32, color=(0,255,0)):
    try:
        # Sử dụng font có sẵn trong Windows
        font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Đường dẫn mặc định đến Arial font trong Windows
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except OSError:
            # Nếu không tìm thấy Arial, thử dùng font Times New Roman
            font_path = "C:\\Windows\\Fonts\\times.ttf"
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Lỗi khi vẽ text: {str(e)}")
        return img  # Trả về ảnh gốc nếu có lỗi

def main():
    CONFIDENCE_THRESHOLD = 0.9  # Ngưỡng xác suất tối thiểu để chấp nhận dự đoán
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    cv2.namedWindow("Traffic Sign Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Sign Recognition", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera")
            break

        # Tiền xử lý frame
        processed_sign, region = preprocess_frame(frame)
        
        if processed_sign is not None:
            # Dự đoán
            input_img = np.expand_dims(processed_sign, axis=0)
            preds = model.predict(input_img)
            class_id = np.argmax(preds)
            confidence = preds[0][class_id]

            # Vẽ khung cho biển báo được phát hiện
            x, y, w, h = region
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Hiển thị kết quả nếu độ tin cậy vượt ngưỡng
            if confidence >= CONFIDENCE_THRESHOLD:
                if class_id < len(class_names):
                    label = f"{class_names[class_id]} - {confidence*100:.2f}%"
                else:
                    label = f"Biển báo khác - {confidence*100:.2f}%"
            else:
                label = f"Không rõ biển báo ({confidence*100:.2f}%)"

            frame = put_vietnamese_text(frame, label, (10, 30), font_size=32, color=(0,255,0))
        else:
            # Hiển thị thông báo nếu không tìm thấy biển báo
            frame = put_vietnamese_text(frame, "Không tìm thấy biển báo", (10, 30), font_size=32, color=(0,255,0))

        cv2.imshow("Traffic Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()