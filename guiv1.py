import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from test_model_v1 import model, preprocess_frame, put_vietnamese_text, class_names
import threading

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Biển Báo Giao Thông")
        self.root.configure(bg='#f0f0f0')
        
        # Biến để theo dõi trạng thái camera
        self.is_camera_running = False
        self.cap = None
        
        # Frame chính
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Frame bên trái (cho ảnh và nút)
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame cho các nút
        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Style cho các nút
        button_style = {
            'font': ('Arial', 10, 'bold'),
            'bg': '#2196F3',
            'fg': 'white',
            'relief': tk.FLAT,
            'padx': 20,
            'pady': 8
        }
        
        # Nút chọn ảnh
        self.select_button = tk.Button(button_frame, text="📂 Chọn Ảnh", 
                                     command=self.select_image, **button_style)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Nút camera
        self.camera_button = tk.Button(button_frame, text="📹 Mở Camera",
                                     command=self.toggle_camera, **button_style)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        # Frame hiển thị ảnh với viền
        self.image_frame = tk.Frame(left_frame, bg='white', bd=1, relief=tk.SOLID)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label hiển thị ảnh
        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(padx=2, pady=2)
        
        # Frame bên phải (thông tin)
        info_frame = tk.Frame(main_frame, bg='white', bd=1, relief=tk.SOLID, width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        info_frame.pack_propagate(False)  # Giữ kích thước cố định
        
        # Tiêu đề panel thông tin
        title_frame = tk.Frame(info_frame, bg='#2196F3', height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="THÔNG TIN NHẬN DIỆN",
                fg='white', bg='#2196F3',
                font=('Arial', 12, 'bold')).pack(pady=12)
        
        # Nội dung thông tin
        content_frame = tk.Frame(info_frame, bg='white', padx=15, pady=15)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label thông tin
        label_style = {'bg': 'white', 'font': ('Arial', 11), 'anchor': 'w'}
        
        self.sign_name_label = tk.Label(content_frame,
                                      text="🚸 Tên biển báo: Chưa phát hiện",
                                      wraplength=250, **label_style)
        self.sign_name_label.pack(fill=tk.X, pady=(0, 10))
        
        self.confidence_label = tk.Label(content_frame,
                                       text="💡 Độ tin cậy: -",
                                       **label_style)
        self.confidence_label.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(content_frame,
                                   text="📡 Trạng thái: Đang chờ",
                                   **label_style)
        self.status_label.pack(fill=tk.X, pady=(0, 10))
        
        # Kích thước hiển thị
        self.display_width = 800
        self.display_height = 600
        
        # Thiết lập kích thước cửa sổ
        self.root.geometry("1200x700")
        self.root.minsize(1200, 700)

    def update_info(self, class_id=None, confidence=None, status="Đang chờ"):
        if class_id is not None and class_id < len(class_names):
            self.sign_name_label.config(text=f"🚸 Tên biển báo: {class_names[class_id]}")
        else:
            self.sign_name_label.config(text="🚸 Tên biển báo: Chưa phát hiện")
            
        if confidence is not None:
            self.confidence_label.config(text=f"💡 Độ tin cậy: {confidence*100:.2f}%")
        else:
            self.confidence_label.config(text="💡 Độ tin cậy: -")
            
        self.status_label.config(text=f"📡 Trạng thái: {status}")

    def select_image(self):
        if self.is_camera_running:
            self.toggle_camera()
            
        file_path = filedialog.askopenfilename(
            title="Chọn Ảnh Biển Báo",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.gif *.ppm")]
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                processed_sign, region = preprocess_frame(image)
                
                if processed_sign is not None:
                    input_img = np.expand_dims(processed_sign, axis=0)
                    preds = model.predict(input_img)
                    class_id = np.argmax(preds)
                    confidence = preds[0][class_id]
                    
                    x, y, w, h = region
                    cv2.rectangle(image, (x, y), (x+w, y+h), (33, 150, 243), 3)
                    
                    self.update_info(class_id, confidence, "Đã phát hiện")
                else:
                    self.update_info(status="Không tìm thấy biển báo")
                
                self.show_image(image)

    def toggle_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_camera_running = True
                self.camera_button.config(text="⏹️ Tắt Camera", bg='#f44336')
                self.update_camera()
                self.update_info(status="Camera đang hoạt động")
            else:
                self.update_info(status="Không thể mở camera!")
        else:
            self.is_camera_running = False
            self.camera_button.config(text="📹 Mở Camera", bg='#2196F3')
            if self.cap:
                self.cap.release()
            self.update_info(status="Đang chờ")

    def update_camera(self):
        if self.is_camera_running:
            ret, frame = self.cap.read()
            if ret:
                processed_sign, region = preprocess_frame(frame)
                
                if processed_sign is not None:
                    input_img = np.expand_dims(processed_sign, axis=0)
                    preds = model.predict(input_img)
                    class_id = np.argmax(preds)
                    confidence = preds[0][class_id]
                    
                    x, y, w, h = region
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (33, 150, 243), 3)
                    
                    self.update_info(class_id, confidence, "Đang phát hiện realtime")
                else:
                    self.update_info(status="Đang tìm kiếm biển báo...")
                
                self.show_image(frame)
                
            self.root.after(10, self.update_camera)

    def show_image(self, cv_image):
        aspect_ratio = cv_image.shape[1] / cv_image.shape[0]
        new_width = min(self.display_width, int(self.display_height * aspect_ratio))
        new_height = min(self.display_height, int(new_width / aspect_ratio))
        
        resized = cv2.resize(cv_image, (new_width, new_height))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
        
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop() 