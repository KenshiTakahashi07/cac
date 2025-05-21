import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Đường dẫn dữ liệu
IMAGE_DIR = 'data_train/archive/images'
LABEL_DIR = 'data_train/archive/labels'
CLASSES_FILE = 'data_train/archive/classes.txt'
IMG_SIZE = 224
BATCH_SIZE = 16
MAX_BOXES = 10  # Số lượng box tối đa cho mỗi ảnh

# Đọc danh sách class
with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f.readlines()]
NUM_CLASSES = len(CLASSES)

def load_yolo_label(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, x, y, w, h = map(float, parts)
            x_min = int((x - w/2) * img_w)
            y_min = int((y - h/2) * img_h)
            x_max = int((x + w/2) * img_w)
            y_max = int((y + h/2) * img_h)
            boxes.append([cls, x_min, y_min, x_max, y_max])
    return np.array(boxes)

def pad_boxes(boxes, max_boxes=MAX_BOXES):
    """Pad boxes array to fixed size"""
    if len(boxes) == 0:
        return np.zeros((max_boxes, 5))
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]
    else:
        padding = np.zeros((max_boxes - len(boxes), 5))
        boxes = np.vstack([boxes, padding])
    return boxes

def data_generator(image_dir, label_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    while True:
        np.random.shuffle(image_files)
        for i in range(0, len(image_files), batch_size):
            batch_imgs = []
            batch_boxes = []
            
            for img_file in image_files[i:i+batch_size]:
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
                
                # Đọc và xử lý ảnh
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                img = cv2.resize(img, (img_size, img_size))
                
                # Đọc và xử lý nhãn
                boxes = load_yolo_label(label_path, w, h)
                
                # Chuyển box về tỉ lệ mới
                scale_x, scale_y = img_size / w, img_size / h
                if len(boxes) > 0:  # Kiểm tra nếu có box
                    boxes = boxes.astype(np.float32)
                    boxes[:, 1] = boxes[:, 1] * scale_x
                    boxes[:, 2] = boxes[:, 2] * scale_y
                    boxes[:, 3] = boxes[:, 3] * scale_x
                    boxes[:, 4] = boxes[:, 4] * scale_y
                
                # Pad boxes để có kích thước cố định
                boxes = pad_boxes(boxes)
                
                batch_imgs.append(img / 255.0)
                batch_boxes.append(boxes)
            
            yield np.array(batch_imgs), np.array(batch_boxes)

# Xây dựng mô hình SSD nhỏ gọn
def build_ssd_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    # Input layer
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    # Backbone: MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone
    
    # Feature extraction
    x = base_model(inputs)
    
    # SSD head (ví dụ đơn giản: 1 scale)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output: 5 values per box (x_min, y_min, x_max, y_max, confidence)
    outputs = layers.Dense(MAX_BOXES * 5, activation='sigmoid')(x)
    outputs = layers.Reshape((MAX_BOXES, 5))(outputs)
    
    model = keras.Model(inputs, outputs)
    return model

# Khởi tạo model
model = build_ssd_model()

# Compile model với custom loss function
def custom_loss(y_true, y_pred):
    # Chỉ tính loss cho các box thực sự (không phải padding)
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    # Mở rộng mask để match với kích thước của y_true và y_pred
    mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
    # Tính MSE loss và áp dụng mask
    loss = tf.reduce_mean(tf.square(y_true - y_pred) * mask)
    return loss

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=custom_loss,
    metrics=['accuracy']
)

# Huấn luyện model
train_gen = data_generator(IMAGE_DIR, LABEL_DIR)
model.fit(
    train_gen,
    steps_per_epoch=len(os.listdir(IMAGE_DIR)) // BATCH_SIZE,
    epochs=10,
    verbose=1
)

# Lưu model
model.save('model.h5')
