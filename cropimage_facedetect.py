import os
from tkinter import Tk, Canvas, Button, Label, filedialog, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO

class ImageCropperWithFaceDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Accurate Crop Tool with Face Detection")

        # Canvasとコントロール用フレームの作成
        self.canvas = Canvas(root, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.control_frame = Frame(root, bg="white", height=50)
        self.control_frame.pack(fill="x", side="bottom")

        # ボタンとステータスラベル
        self.open_folder_btn = Button(self.control_frame, text="Open Folder", command=self.open_folder)
        self.save_btn = Button(self.control_frame, text="Crop & Save", command=self.crop_and_save, state="disabled")
        self.status_label = Label(self.control_frame, text="No folder selected", bg="white")

        self.open_folder_btn.pack(side="left", padx=10)
        self.save_btn.pack(side="left", padx=10)
        self.status_label.pack(side="left", padx=10)

        # 属性
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None
        self.tk_image = None
        self.image_offset = [0, 0]  # 画像のオフセット位置
        self.scale = 1.0  # 画像の拡大縮小スケール
        self.crop_size = 512
        self.output_size = 1024

        # YOLOモデルのロード
        self.model_path = os.path.join(os.getcwd(), "face_yolov8m.pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Please place the YOLOv8 model in the current directory.")
        self.model = YOLO(self.model_path)

        # マウスイベントのバインド
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # 拡大縮小
        self.canvas.bind("<B1-Motion>", self.on_drag)  # ドラッグ
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)  # ドラッグ開始

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            self.current_image_index = 0
            if self.image_list:
                os.makedirs("output", exist_ok=True)
                self.load_image()
                self.update_ui(folder_loaded=True)
            else:
                self.update_ui(folder_loaded=False)

    def update_ui(self, folder_loaded=False, images_remaining=True):
        """UIの状態を更新"""
        if folder_loaded:
            self.status_label.config(text=f"{len(self.image_list)} images loaded.")
            self.save_btn.config(state="normal" if images_remaining else "disabled")
        else:
            self.status_label.config(text="No folder selected")
            self.save_btn.config(state="disabled")
        self.root.update_idletasks()

    def load_image(self):
        if self.current_image_index < len(self.image_list):
            image_path = self.image_list[self.current_image_index]
            self.original_image = Image.open(image_path).convert("RGBA")
            self.image_offset = [0, 0]
            self.scale = 1.0

            # 顔検出
            face_center = self.detect_face(image_path)
            if face_center:
                print(f"Face detected at: {face_center}")
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                self.image_offset[0] = -(face_center[0] * self.scale) + canvas_width // 2
                self.image_offset[1] = -(face_center[1] * self.scale) + canvas_height // 2
            else:
                print("No face detected.")

            self.display_image()
            self.update_ui(folder_loaded=True, images_remaining=True)
        else:
            self.update_ui(folder_loaded=True, images_remaining=False)

    def detect_face(self, image_path):
        results = self.model.predict(image_path, conf=0.25)
        detections = results[0].boxes.xyxy.tolist() if results else []
        if detections:
            x1, y1, x2, y2 = detections[0]
            face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            return face_center
        return None

    def display_image(self):
        """Canvasに画像を描画"""
        if self.original_image is None:
            return

        self.canvas.delete("all")

        scaled_width = int(self.original_image.width * self.scale)
        scaled_height = int(self.original_image.height * self.scale)
        resized_image = self.original_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        self.canvas.create_image(self.image_offset[0], self.image_offset[1], image=self.tk_image, anchor="nw")

        # 赤枠を描画
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        crop_left = (canvas_width - self.crop_size) // 2
        crop_top = (canvas_height - self.crop_size) // 2
        crop_right = crop_left + self.crop_size
        crop_bottom = crop_top + self.crop_size

        self.canvas.create_rectangle(crop_left, crop_top, crop_right, crop_bottom, outline="red", width=2)

    def on_drag_start(self, event):
        """ドラッグ操作の開始"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        """ドラッグ操作で画像を移動"""
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.image_offset[0] += dx
        self.image_offset[1] += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.display_image()

    def on_zoom(self, event):
        """マウスホイールで画像を拡大縮小"""
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= zoom_factor
        self.display_image()

    def crop_and_save(self):
        if self.original_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        crop_left_canvas = (canvas_width - self.crop_size) // 2
        crop_top_canvas = (canvas_height - self.crop_size) // 2

        crop_left_image = (crop_left_canvas - self.image_offset[0]) / self.scale
        crop_top_image = (crop_top_canvas - self.image_offset[1]) / self.scale
        crop_right_image = crop_left_image + self.crop_size / self.scale
        crop_bottom_image = crop_top_image + self.crop_size / self.scale

        crop_box = (
            int(crop_left_image),
            int(crop_top_image),
            int(crop_right_image),
            int(crop_bottom_image)
        )

        cropped_image = self.original_image.crop(crop_box)
        resized_image = cropped_image.resize((self.output_size, self.output_size), Image.LANCZOS)

        output_path = os.path.join("output", f"cropped_{self.current_image_index + 1}.png")
        resized_image.save(output_path)
        print(f"Saved: {output_path}")

        self.current_image_index += 1
        self.load_image()

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x800")  # 初期ウィンドウサイズを指定（幅1200px、高さ800px）
    app = ImageCropperWithFaceDetection(root)
    root.mainloop()
