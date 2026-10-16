from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (48, 48),
        padding: float = 0.2,
    ):

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Не вдалося завантажити каскад Хаара: {path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.padding = padding

    def detect(self, image_bgr: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        h, w = image_bgr.shape[:2]
        boxes: List[Box] = []
        for (x, y, fw, fh) in faces:
            pad_x, pad_y = int(fw * self.padding), int(fh * self.padding)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
            boxes.append((x1, y1, x2, y2))
        return boxes
