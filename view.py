
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, List, Optional

from PIL import Image, ImageTk


class EmotionApp(tk.Tk):

    DISPLAY_W = 640
    DISPLAY_H = 480

    def __init__(self):
        super().__init__()
        self.title("Розпізнавання емоційного стану людини")
        self.geometry("980x620")
        self.minsize(860, 560)
        self.configure(bg="#f2f2f2")

        self.on_select_image: Optional[Callable[[str], None]] = None
        self.on_select_video: Optional[Callable[[str], None]] = None
        self.on_toggle_camera: Optional[Callable[[], None]] = None
        self.on_stop: Optional[Callable[[], None]] = None

        self._tk_image = None 

        self._build_widgets()

    def _build_widgets(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(
            toolbar, text="Обрати фото", command=self._handle_select_image
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            toolbar, text="Обрати відео", command=self._handle_select_video
        ).pack(side=tk.LEFT, padx=4)

        self.camera_button = ttk.Button(
            toolbar, text="Увімкнути камеру", command=self._handle_toggle_camera
        )
        self.camera_button.pack(side=tk.LEFT, padx=4)

        self.stop_button = ttk.Button(
            toolbar, text="Зупинити", command=self._handle_stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=4)

        main_area = ttk.Frame(self, padding=(10, 0, 10, 10))
        main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        image_frame = ttk.LabelFrame(main_area, text="Зображення")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.image_label = ttk.Label(
            image_frame,
            background="#1e1e1e",
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._show_placeholder("Оберіть фото, відео або увімкніть камеру")

        results_frame = ttk.LabelFrame(main_area, text="Результат розпізнавання", width=300)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y)
        results_frame.pack_propagate(False)

        ttk.Separator(results_frame).pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(results_frame, text="Реакція робота:", font=("", 10, "bold")).pack(anchor=tk.W, padx=10)
        self.reaction_var = tk.StringVar(value="—")
        ttk.Label(results_frame, textvariable=self.reaction_var, wraplength=270, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(2, 10))

        self.main_emotion_var = tk.StringVar(value="—")
        self.main_confidence_var = tk.StringVar(value="")

        ttk.Label(results_frame, text="Домінуюча емоція:", font=("", 10, "bold")).pack(
            anchor=tk.W, padx=10, pady=(12, 0)
        )
        ttk.Label(
            results_frame, textvariable=self.main_emotion_var, font=("", 16, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(0, 2))
        ttk.Label(results_frame, textvariable=self.main_confidence_var).pack(
            anchor=tk.W, padx=10, pady=(0, 10)
        )

        ttk.Separator(results_frame).pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(results_frame, text="Усі знайдені обличчя:", font=("", 10, "bold")).pack(
            anchor=tk.W, padx=10, pady=(4, 4)
        )

        self.results_list = tk.Listbox(results_frame, height=14, activestyle="none")
        self.results_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="Готово до роботи.")
        status_bar = ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    def show_reaction(self, text: str) -> None:
        self.reaction_var.set(text)

        
    def _handle_select_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Оберіть зображення",
            filetypes=[
                ("Зображення", "*.jpg *.jpeg *.png *.bmp"),
                ("Усі файли", "*.*"),
            ],
        )
        if path and self.on_select_image:
            self.on_select_image(path)

    def _handle_select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Оберіть відеофайл",
            filetypes=[
                ("Відео", "*.mp4 *.avi *.mov *.mkv"),
                ("Усі файли", "*.*"),
            ],
        )
        if path and self.on_select_video:
            self.on_select_video(path)

    def _handle_toggle_camera(self) -> None:
        if self.on_toggle_camera:
            self.on_toggle_camera()

    def _handle_stop(self) -> None:
        if self.on_stop:
            self.on_stop()

    def set_running_state(self, running: bool, is_camera: bool = False) -> None:
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)
        if is_camera:
            self.camera_button.config(
                text="Вимкнути камеру" if running else "Увімкнути камеру"
            )

    def show_image(self, pil_image: Image.Image) -> None:
        fitted = self._fit_to_display(pil_image)
        self._tk_image = ImageTk.PhotoImage(fitted)
        self.image_label.configure(image=self._tk_image, text="")

    def show_results(self, detections, main_label: str, main_conf: Optional[float]) -> None:

        self.main_emotion_var.set(main_label)
        self.main_confidence_var.set(
            f"Впевненість: {main_conf * 100:.1f}%" if main_conf is not None else ""
        )

        self.results_list.delete(0, tk.END)
        if not detections:
            self.results_list.insert(tk.END, "Обличчя не знайдено")
            return

        for i, det in enumerate(detections, start=1):
            bar_len = int(round(det.confidence * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            self.results_list.insert(
                tk.END, f"{i}. {det.label_ua:<12} {bar} {det.confidence * 100:5.1f}%"
            )

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def show_warning(self, title: str, message: str) -> None:
        messagebox.showwarning(title, message)

    def _show_placeholder(self, text: str) -> None:
        self.image_label.configure(image="", text=text, foreground="#aaaaaa")

    def _fit_to_display(self, pil_image: Image.Image) -> Image.Image:
        img = pil_image.copy()
        img.thumbnail((self.DISPLAY_W, self.DISPLAY_H), Image.LANCZOS)
        return img
