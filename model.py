from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


EMOTION_TRANSLATIONS = {
    "happy": "Радість",
    "sad": "Смуток",
    "angry": "Злість",
    "surprise": "Подив",
    "fear": "Страх",
    "disgust": "Відраза",
    "neutral": "Нейтральний",
}


@dataclass
class Detection:
    label: str
    label_ua: str
    confidence: float
    box: tuple


@dataclass
class PredictionResult:
    annotated_image: np.ndarray
    detections: List[Detection]
    face_found: bool


class EmotionModel:

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.4,
        face_detector=None,
    ):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.face_detector = face_detector
        self._model = None  # завантажується лениво, через load()

    def load(self) -> None:
        
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"Файл моделі не знайдено: {self.weights_path}\n"
                f"Вкажіть правильний шлях до файлу .pt з навченими вагами."
            )

        from ultralytics import YOLO

        self._model = YOLO(self.weights_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict:
        if not self.is_loaded:
            return {}
        return self._model.names

    def predict(self, image_bgr: np.ndarray) -> PredictionResult:
        if not self.is_loaded:
            raise RuntimeError("Модель не завантажена. Спочатку викличте load().")

        if self.face_detector is None:
            detections = self._run_yolo(image_bgr)
        else:
            detections = self._predict_with_face_detector(image_bgr)

        annotated = self._draw_detections(image_bgr.copy(), detections)

        return PredictionResult(
            annotated_image=annotated,
            detections=detections,
            face_found=len(detections) > 0,
        )

    def _run_yolo(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=image_bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        result = results[0]

        detections: List[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            label = self._model.names.get(cls_id, str(cls_id))
            label_ua = EMOTION_TRANSLATIONS.get(label.lower(), label)

            detections.append(
                Detection(
                    label=label,
                    label_ua=label_ua,
                    confidence=conf,
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    def _predict_with_face_detector(self, image_bgr: np.ndarray) -> List[Detection]:

        detections: List[Detection] = []

        for (x1, y1, x2, y2) in self.face_detector.detect(image_bgr):
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            best = self.best_detection(self._run_yolo(crop))
            if best is None:
                continue

            detections.append(
                Detection(
                    label=best.label,
                    label_ua=best.label_ua,
                    confidence=best.confidence,
                    box=(x1, y1, x2, y2),
                )
            )
        return detections

    @staticmethod
    def _draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = (60, 180, 75)  # зелений (BGR)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            caption = f"{det.label_ua} {det.confidence * 100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                image,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                image,
                caption,
                (x1 + 3, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    @staticmethod
    def best_detection(detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
