import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import os

# Custom loss function (phải định nghĩa lại giống như trong file train)
def custom_loss(y_true, y_pred):
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
    loss = tf.reduce_mean(tf.square(y_true - y_pred) * mask)
    return loss

# Kiểm tra và load model
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model {MODEL_PATH}")
    print("Vui lòng đảm bảo file model đã được train và lưu đúng vị trí")
    exit(1)

try:
    model = load_model(MODEL_PATH, custom_objects={'custom_loss': custom_loss})
    print("Đã load model thành công!")
except Exception as e:
    print(f"Lỗi khi load model: {str(e)}")
    exit(1)

class_names = {
    0: 'Đường người đi bộ cắt ngang',
    1: 'Đường giao nhau (ngã ba bên phải)',
    2: 'Cấm đi ngược chiều',
    3: 'Phải đi vòng sang bên phải',
    4: 'Giao nhau với đường đồng cấp',
    5: 'Giao nhau với đường không ưu tiên',
    6: 'Chỗ ngoặt nguy hiểm vòng bên trái',
    7: 'Cấm rẽ trái',
    8: 'Bến xe buýt',
    9: 'Nơi giao nhau chạy theo vòng xuyến',
    10: 'Cấm dừng và đỗ xe',
    11: 'Chỗ quay xe',
    12: 'Biển gộp làn đường theo phương tiện',
    13: 'Đi chậm',
    14: 'Cấm xe tải',
    15: 'Đường bị thu hẹp về phía phải',
    16: 'Giới hạn chiều cao',
    17: 'Cấm quay đầu',
    18: 'Cấm ô tô khách và ô tô tải',
    19: 'Cấm rẽ phải và quay đầu',
    20: 'Cấm ô tô',
    21: 'Đường bị thu hẹp về phía trái',
    22: 'Gồ giảm tốc phía trước',
    23: 'Cấm xe hai và ba bánh',
    24: 'Kiểm tra',
    25: 'Chỉ dành cho xe máy',
    26: 'Chướng ngoại vật phía trước',
    27: 'rẻ em',
    28: 'Xe tải và công Xe tải và xe công',
    29: 'Cấm mô tô và xe máy',
    30: 'Chỉ dành cho xe tải',
    31: 'Đường có camera giám sát',
    32: 'Cấm rẽ phải',
    33: 'Cấm xe sơ-mi rơ-moóc',
    34: 'Cấm rẽ trái và phải',
    35: 'Cấm đi thẳng và rẽ phải',
    36: 'Đường giao nhau (ngã ba bên trái)',
    37: 'Giới hạn tốc độ (50km/h)',
    38: 'Giới hạn tốc độ (60km/h)',
    39: 'Giới hạn tốc độ (80km/h)',
    40: 'Giới hạn tốc độ (40km/h)',
    41: 'Các xe chỉ được rẽ trái',
    42: 'Chiều cao tĩnh không thực tế',
    43: 'Nguy hiểm khác',
    44: 'Đường một chiều',
    45: 'Cấm đỗ xe',
    46: 'Cấm ô tô quay đầu xe (được rẽ trái)',
    47: 'Giao nhau với đường sắt có rào chắn',
    48: 'Cấm rẽ trái và quay đầu xe',
    49: 'Chỗ ngoặt nguy hiểm vòng bên phải',
    50: 'Chú ý chướng ngại vật vòng tránh sang bên phải'
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

def preprocess_frame(frame, size=(224, 224)):
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
    
    # Resize về kích thước phù hợp với model (224x224)
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
    CONFIDENCE_THRESHOLD = 0.7  # Ngưỡng xác suất tối thiểu để chấp nhận dự đoán
    
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
            class_id = np.argmax(preds[0])  # Lấy index của class có xác suất cao nhất
            confidence = float(preds[0][class_id].tolist()[0])  # Lấy phần tử đầu tiên của list và chuyển thành float

            # Vẽ khung cho biển báo được phát hiện
            x, y, w, h = region
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Hiển thị kết quả nếu độ tin cậy vượt ngưỡng
            if confidence >= CONFIDENCE_THRESHOLD:
                label = f"{class_names[class_id]} - {confidence*100:.2f}%"
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
