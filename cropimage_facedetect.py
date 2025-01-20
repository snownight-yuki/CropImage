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
        self.save_btn = Button(self.control_frame, text="Crop & Save", command=self.crop_selected_face, state="disabled")
        self.status_label = Label(self.control_frame, text="No folder selected", bg="white")

        self.open_folder_btn.pack(side="left", padx=10)
        self.save_btn.pack(side="left", padx=10)
        self.status_label.pack(side="left", padx=10)

        # 属性
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None
        self.tk_image = None
        self.image_offset = [0, 0]
        self.scale = 1.0
        self.output_size = 1024
        self.face_boxes = []  # 顔検出の座標リスト（画像内座標）
        self.selected_face_index = None

        # ドラッグ操作用
        self.drag_start_x = 0
        self.drag_start_y = 0

        # クロップ枠移動用の属性
        self.is_moving_crop = False
        self.crop_drag_start_x = 0
        self.crop_drag_start_y = 0
        self.original_box_coords = None  # 移動開始時の枠の位置（画像内座標）

        # リサイズ状態の属性
        self.is_resizing = False
        self.resize_handle_size = 10
        self.active_handle_index = None   # 0:左上, 1:右上, 2:左下, 3:右下
        self.fixed_point = None            # リサイズ中に固定する対角の点（画像内座標）

        # YOLOモデルのロード
        self.model_path = os.path.join(os.getcwd(), "yolov11n-face.pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Please place the YOLOv11 model in the current directory.")
        self.model = YOLO(self.model_path)

        # マウスイベントのバインド
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def open_folder(self):
        """フォルダを選択し、最初の画像をロード"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ]
            self.current_image_index = 0
            if self.image_list:
                os.makedirs("output", exist_ok=True)
                self.load_image()
                self.update_ui(folder_loaded=True)
            else:
                self.update_ui(folder_loaded=False)

    def update_ui(self, folder_loaded=False):
        if folder_loaded:
            self.status_label.config(text=f"{len(self.image_list)} images loaded.")
            self.save_btn.config(state="normal" if self.face_boxes else "disabled")
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
            self.detect_faces(image_path)
        else:
            self.update_ui(folder_loaded=False)

    def display_image(self):
        if self.original_image is None:
            return
        self.canvas.delete("all")
        scaled_width = int(self.original_image.width * self.scale)
        scaled_height = int(self.original_image.height * self.scale)
        resized_image = self.original_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(
            self.image_offset[0], self.image_offset[1],
            image=self.tk_image, anchor="nw", tags="image"
        )
        # 顔検出枠の描画（青枠）
        for idx, (x1, y1, x2, y2) in enumerate(self.face_boxes):
            self.draw_face_box(idx, x1, y1, x2, y2)
        # 選択領域（赤枠とリサイズハンドル）の描画
        if self.selected_face_index is not None:
            x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
            self.canvas.create_rectangle(
                x1 * self.scale + self.image_offset[0],
                y1 * self.scale + self.image_offset[1],
                x2 * self.scale + self.image_offset[0],
                y2 * self.scale + self.image_offset[1],
                outline="red", width=3, tags="highlight"
            )
            self.add_resize_handles(
                x1 * self.scale + self.image_offset[0],
                y1 * self.scale + self.image_offset[1],
                x2 * self.scale + self.image_offset[0],
                y2 * self.scale + self.image_offset[1]
            )

    def detect_faces(self, image_path):
        """顔を検出してCanvasに描画"""
        self.face_boxes = []
        self.selected_face_index = None
        self.display_image()  # 画像描画

        # 顔検出
        results = self.model.predict(image_path, conf=0.3)
        detections = results[0].boxes.xyxy.tolist() if results else []

        if detections:
            for idx, (x1, y1, x2, y2) in enumerate(detections):
                face_height = y2 - y1
                padding_top = face_height*0.3
                padding_bottom = face_height*0.3
                padding_sides = face_height*0.3

                x1 -= padding_sides*1.8
                y1 -= padding_top*1.8
                x2 += padding_sides
                y2 += padding_bottom

                # 正方形に補正（長いほうに合わせる）
                side = max(x2 - x1, y2 - y1)
                self.face_boxes.append((x1, y1, x1 + side, y1 + side))
                self.draw_face_box(idx, x1, y1, x1 + side, y1 + side)

            # 検出結果が1件のみなら自動的に選択状態にする
            if len(detections) == 1:
                self.selected_face_index = 0
                # 選択状態となるように再描画
                self.display_image()
        else:
            self.add_default_box()

    def draw_face_box(self, idx, x1, y1, x2, y2):
        x1_canvas = x1 * self.scale + self.image_offset[0]
        y1_canvas = y1 * self.scale + self.image_offset[1]
        x2_canvas = x2 * self.scale + self.image_offset[0]
        y2_canvas = y2 * self.scale + self.image_offset[1]
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="blue", width=2, tags=f"visual_{idx}"
        )
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="", fill="", tags=f"clickable_{idx}"
        )
        self.canvas.tag_bind(f"clickable_{idx}", "<Button-1>", lambda event, i=idx: self.select_face(i))

    def add_default_box(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # デフォルトは正方形 512x512
        x1_canvas = (canvas_width - 512) // 2
        y1_canvas = (canvas_height - 512) // 2
        x2_canvas = x1_canvas + 512
        y2_canvas = y1_canvas + 512
        # ※ここでは画面上のキャンバス座標をそのまま画像内座標とする（scale=1, offset=0想定）
        self.face_boxes = [(x1_canvas, y1_canvas, x2_canvas, y2_canvas)]
        self.selected_face_index = 0
        self.canvas.create_rectangle(
            x1_canvas, y1_canvas, x2_canvas, y2_canvas,
            outline="red", width=3, tags="default"
        )
        self.canvas.tag_bind("default", "<Button-1>", lambda event: self.select_face(0))

    def select_face(self, index):
        self.selected_face_index = index
        self.display_image()

    def add_resize_handles(self, x1, y1, x2, y2):
        handle_size = self.resize_handle_size
        handles = [
            (x1, y1),  # 左上: handle_index=0
            (x2, y1),  # 右上: handle_index=1
            (x1, y2),  # 左下: handle_index=2
            (x2, y2)   # 右下: handle_index=3
        ]
        for idx, (hx, hy) in enumerate(handles):
            tag = f"resize_handle_{idx}"
            self.canvas.create_rectangle(
                hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size,
                fill="red", outline="black", tags=tag
            )
            self.canvas.tag_bind(tag, "<ButtonPress-1>", lambda event, i=idx: self.start_resize(i))
            self.canvas.tag_bind(tag, "<ButtonRelease-1>", self.end_resize)

    def start_resize(self, handle_index):
        """リサイズ開始時に、操作しているハンドルに対し対角の固定点を記録する"""
        if self.selected_face_index is None:
            return
        self.is_resizing = True
        self.active_handle_index = handle_index
        # 現在の枠（画像内座標）
        x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
        if handle_index == 0:        # 左上を操作 → 右下固定
            self.fixed_point = (x2, y2)
        elif handle_index == 1:      # 右上を操作 → 左下固定
            self.fixed_point = (x1, y2)
        elif handle_index == 2:      # 左下を操作 → 右上固定
            self.fixed_point = (x2, y1)
        elif handle_index == 3:      # 右下を操作 → 左上固定
            self.fixed_point = (x1, y1)

    def end_resize(self, event):
        self.is_resizing = False
        self.active_handle_index = None
        self.fixed_point = None

    def on_drag_start(self, event):
        # まず、リサイズ操作中でなければ、クリック位置が赤い選択枠内かどうかチェック
        if not self.is_resizing and self.selected_face_index is not None:
            x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
            # キャンバス上での赤枠の座標
            box_left = x1 * self.scale + self.image_offset[0]
            box_top = y1 * self.scale + self.image_offset[1]
            box_right = x2 * self.scale + self.image_offset[0]
            box_bottom = y2 * self.scale + self.image_offset[1]
            if box_left <= event.x <= box_right and box_top <= event.y <= box_bottom:
                # クリック位置が赤枠内部なら、クロップ枠移動モードにする
                self.is_moving_crop = True
                self.crop_drag_start_x = event.x
                self.crop_drag_start_y = event.y
                self.original_box_coords = (x1, y1, x2, y2)
                return  # ここで終了し、画像移動処理は行わない

        # リサイズ中以外の場合、通常は画像移動用のドラッグ開始
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        if self.is_resizing and self.active_handle_index is not None and self.selected_face_index is not None:
            # リサイズ中の処理（正方形固定）
            # 現在のマウス位置→画像内座標
            current_img_x = (event.x - self.image_offset[0]) / self.scale
            current_img_y = (event.y - self.image_offset[1]) / self.scale

            fixed_x, fixed_y = self.fixed_point
            # 固定点からの差分
            dx = current_img_x - fixed_x
            dy = current_img_y - fixed_y
            side = max(abs(dx), abs(dy))
            # 各軸の符号（固定点から見たドラッグの方向）を保持
            new_x = fixed_x + (side if dx >= 0 else -side)
            new_y = fixed_y + (side if dy >= 0 else -side)
            # 新しい枠は固定点と新たな点の対角の矩形
            x1_new = min(fixed_x, new_x)
            y1_new = min(fixed_y, new_y)
            x2_new = max(fixed_x, new_x)
            y2_new = max(fixed_y, new_y)
            # 最小サイズチェック
            min_size = 20
            if (x2_new - x1_new) < min_size or (y2_new - y1_new) < min_size:
                return
            self.face_boxes[self.selected_face_index] = (x1_new, y1_new, x2_new, y2_new)
            self.display_image()
        elif self.is_moving_crop and self.selected_face_index is not None:
            # クロップ枠移動中の場合
            dx = (event.x - self.crop_drag_start_x) / self.scale  # 画像内座標での差分
            dy = (event.y - self.crop_drag_start_y) / self.scale
            orig_x1, orig_y1, orig_x2, orig_y2 = self.original_box_coords
            new_coords = (orig_x1 + dx, orig_y1 + dy, orig_x2 + dx, orig_y2 + dy)
            self.face_boxes[self.selected_face_index] = new_coords
            self.display_image()
        else:
            # 画像移動処理
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.image_offset[0] += dx
            self.image_offset[1] += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.display_image()

    def on_mouse_release(self, event):
        if self.is_resizing:
            self.end_resize(event)
        if self.is_moving_crop:
            self.is_moving_crop = False
            self.original_box_coords = None

    def on_zoom(self, event):
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= zoom_factor
        self.display_image()

    def crop_selected_face(self):
        """選択した顔領域をクロップし保存"""
        if self.selected_face_index is None or self.original_image is None:
            return

        x1, y1, x2, y2 = self.face_boxes[self.selected_face_index]
        cropped_image = self.original_image.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # PIL の ImageFilter を使用（ファイル先頭に「from PIL import ImageFilter」がない場合は追加してください）
        from PIL import ImageFilter

        # 1. クロップ直後に DETAIL フィルタを適用して微細部をやや強調する
        detail_enhanced = cropped_image.filter(ImageFilter.DETAIL)
        
        # 2. LANCZOS により固定サイズにリサイズ
        resized_image = detail_enhanced.resize((self.output_size, self.output_size), Image.LANCZOS)
        
        # 3. リサイズ後にも軽く DETAIL フィルタを適用（必要に応じて）
        final_image = resized_image.filter(ImageFilter.DETAIL)
        
        output_path = os.path.join("output", f"cropped_{self.current_image_index + 1}.png")
        final_image.save(output_path)
        print(f"Saved: {output_path}")

        # 次の画像へ
        self.current_image_index += 1
        if self.current_image_index < len(self.image_list):
            self.load_image()
        else:
            self.update_ui(folder_loaded=False)

if __name__ == "__main__":
    root = Tk()
    root.geometry("1200x800")
    app = ImageCropperWithFaceDetection(root)
    root.mainloop()
