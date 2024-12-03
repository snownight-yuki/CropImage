import os
from tkinter import Tk, Canvas, Button, Label, filedialog
from PIL import Image, ImageTk


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Accurate Crop Tool - Save as 1024x1024")

        # UI Elements
        self.canvas = Canvas(root, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.control_frame = Label(self.root, bg="white", height=2)
        self.control_frame.pack(fill="x")

        self.open_folder_btn = Button(self.control_frame, text="Open Folder", command=self.open_folder)
        self.open_folder_btn.pack(side="left", padx=10)

        self.save_btn = Button(self.control_frame, text="Crop & Save", command=self.crop_and_save, state="disabled")
        self.save_btn.pack(side="left", padx=10)

        self.status_label = Label(self.control_frame, text="No folder selected", bg="white")
        self.status_label.pack(side="left", padx=10)

        # Attributes
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None
        self.tk_image = None
        self.image_offset = [0, 0]  # Offset of image top-left corner
        self.scale = 1.0
        self.crop_size = 512  # Fixed red frame size for display
        self.output_size = 1024  # Final output size

        # Event Bindings
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<MouseWheel>", self.on_zoom)

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            self.current_image_index = 0
            if self.image_list:
                os.makedirs("output", exist_ok=True)
                self.status_label.config(text=f"{len(self.image_list)} images loaded.")
                self.load_image()
                self.save_btn.config(state="normal")
            else:
                self.status_label.config(text="No images found in selected folder.")

    def load_image(self):
        image_path = self.image_list[self.current_image_index]
        self.original_image = Image.open(image_path).convert("RGBA")
        self.image_offset = [0, 0]  # Reset offset
        self.scale = 1.0  # Reset scale
        self.display_image()

    def display_image(self):
        self.canvas.delete("all")

        # Calculate scaled dimensions
        scaled_width = int(self.original_image.width * self.scale)
        scaled_height = int(self.original_image.height * self.scale)
        resized_image = self.original_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # Draw image on canvas
        self.canvas.create_image(self.image_offset[0], self.image_offset[1], image=self.tk_image, anchor="nw")

        # Draw red crop frame
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        crop_left = (canvas_width - self.crop_size) // 2
        crop_top = (canvas_height - self.crop_size) // 2
        crop_right = crop_left + self.crop_size
        crop_bottom = crop_top + self.crop_size

        self.canvas.create_rectangle(crop_left, crop_top, crop_right, crop_bottom, outline="red", width=2)

    def on_resize(self, event):
        self.display_image()

    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.image_offset[0] += dx
        self.image_offset[1] += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.display_image()

    def on_zoom(self, event):
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.scale *= zoom_factor
        self.display_image()

    def crop_and_save(self):
        # Get crop box coordinates in Canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        crop_left_canvas = (canvas_width - self.crop_size) // 2
        crop_top_canvas = (canvas_height - self.crop_size) // 2

        # Convert Canvas coordinates to original image coordinates
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

        # Crop the image
        cropped_image = self.original_image.crop(crop_box)

        # Resize the cropped image to 1024x1024
        resized_image = cropped_image.resize((self.output_size, self.output_size), Image.LANCZOS)

        # Save the cropped image
        output_path = os.path.join("output", f"cropped_{self.current_image_index + 1}.png")
        resized_image.save(output_path)
        print(f"Saved: {output_path}")

        # Move to next image
        self.current_image_index += 1
        if self.current_image_index < len(self.image_list):
            self.load_image()
        else:
            self.status_label.config(text="All images processed!")
            self.save_btn.config(state="disabled")


if __name__ == "__main__":
    root = Tk()
    app = ImageCropper(root)
    root.mainloop()
