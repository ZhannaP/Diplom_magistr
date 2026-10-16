from __future__ import annotations

import threading
import time

import cv2
from PIL import Image

from model import EmotionModel
from view import EmotionApp
from robot_reaction import RobotReactionManager


class EmotionController:
    def __init__(
        self,
        model: EmotionModel,
        view: EmotionApp,
        reaction: RobotReactionManager | None = None,
    ):
        self.model = model
        self.view = view
        self.reaction = reaction or RobotReactionManager()

        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_camera = False

        self.view.on_select_image = self.handle_select_image
        self.view.on_select_video = self.handle_select_video
        self.view.on_toggle_camera = self.handle_toggle_camera
        self.view.on_stop = self.handle_stop

    def load_model(self) -> None:
        """Завантажує навчену модель. Викликається перед показом вікна."""
        self.view.set_status("Завантаження моделі…")
        try:
            self.model.load()
            self.view.set_status("Модель завантажено. Готово до роботи.")
        except Exception as exc:
            self.view.set_status("Помилка завантаження моделі.")
            self.view.show_error(
                "Не вдалося завантажити модель",
                f"{exc}\n\n"
                f"Перевірте шлях до файлу ваг у main.py (WEIGHTS_PATH).",
            )

    def handle_select_image(self, path: str) -> None:
        self._stop_stream_if_running()

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            self.view.show_warning(
                "Помилка відкриття файлу", f"Не вдалося відкрити зображення:\n{path}"
            )
            return

        self.view.set_status(f"Обробка зображення: {path}")
        try:
            result = self._predict_safely(image_bgr)
        except Exception as exc:
            self.view.show_error("Помилка розпізнавання", str(exc))
            self.view.set_status("Помилка під час розпізнавання.")
            return

        self._render_result(result, stream=False)
        if result.face_found:
            self.view.set_status("Розпізнавання завершено.")
        else:
            self.view.set_status("Обличчя на зображенні не знайдено.")

    def handle_select_video(self, path: str) -> None:
        self._start_stream(source=path, is_camera=False)

    def handle_toggle_camera(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive() and self._is_camera:
            self.handle_stop()
        else:
            self._start_stream(source=0, is_camera=True)

    def handle_stop(self) -> None:
        self._stop_stream_if_running()
        self.view.set_status("Зупинено користувачем.")


    def _start_stream(self, source, is_camera: bool) -> None:
        if not self.model.is_loaded:
            self.view.show_warning("Модель не завантажена", "Спочатку дочекайтесь завантаження моделі.")
            return

        self._stop_stream_if_running()

        self._stop_event.clear()
        self._is_camera = is_camera
        self.view.set_running_state(running=True, is_camera=is_camera)
        self.view.set_status("Відкриття камери…" if is_camera else "Відтворення відео…")

        self._stream_thread = threading.Thread(
            target=self._stream_loop, args=(source, is_camera), daemon=True
        )
        self._stream_thread.start()

    def _stop_stream_if_running(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stop_event.set()
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None
        self.reaction.reset()
        self.view.after(0, lambda: self.view.set_running_state(running=False, is_camera=True))

    def _stream_loop(self, source, is_camera: bool) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            self.view.after(
                0,
                lambda: self.view.show_error(
                    "Помилка джерела відео",
                    "Не вдалося відкрити камеру." if is_camera else "Не вдалося відкрити відеофайл.",
                ),
            )
            self.view.after(0, lambda: self.view.set_running_state(running=False, is_camera=True))
            return

        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        frame_delay = 1.0 / fps if fps > 0 else 0.04

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                try:
                    result = self._predict_safely(frame)
                except Exception as exc:
                    self.view.after(
                        0, lambda e=exc: self.view.show_error("Помилка розпізнавання", str(e))
                    )
                    break

                self.view.after(0, lambda r=result: self._render_result(r, stream=True))

                if not is_camera:
                    time.sleep(frame_delay)
        finally:
            capture.release()
            self.view.after(0, lambda: self.view.set_running_state(running=False, is_camera=True))
            if not self._stop_event.is_set():
                self.view.after(
                    0, lambda: self.view.set_status("Відтворення завершено." if not is_camera else "Камеру вимкнено.")
                )


    def _predict_safely(self, frame_bgr):
        return self.model.predict(frame_bgr)

    def _render_result(self, result, stream: bool = False) -> None:
        rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.view.show_image(pil_image)

        best = self.model.best_detection(result.detections)
        if best is not None:
            self.view.show_results(result.detections, best.label_ua, best.confidence)
            reaction = self.reaction.react(best.label, best.confidence, stream=stream)
        else:
            self.view.show_results([], "Обличчя не знайдено", None)
            reaction = self.reaction.react(None, None, stream=stream)

        if reaction is not None:
            self.view.show_reaction(reaction.message or reaction.scenario)
