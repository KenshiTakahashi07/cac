import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục dữ liệu
DATA_DIR = "GTSRB/Final_Training/Images"
IMG_SIZE = 50  # Kích thước ảnh đầu vào

# 1. Đọc dữ liệu và nhãn
images = []
labels = []

for class_id in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_id)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        if img_name.endswith('.ppm') or img_name.endswith('.png') or img_name.endswith('.jpg'):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(int(class_id))

images = np.array(images)
labels = np.array(labels)

# 2. Chuẩn hóa dữ liệu
images = images.astype('float32') / 255.0
labels = to_categorical(labels, num_classes=len(np.unique(labels)))

# 3. Chia train/test
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# 4. Xây dựng mô hình CNN đơn giản
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Huấn luyện mô hình
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# 6. Lưu mô hình
model.save("traffic_sign_model_v1.h5")
print("Đã lưu mô hình vào traffic_sign_model_v1.h5")