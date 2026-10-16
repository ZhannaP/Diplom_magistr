import threading

from model import EmotionModel
from view import EmotionApp
from controller import EmotionController
from face_detector import FaceDetector
from robot_reaction import RobotReactionManager

USE_FACE_DETECTOR = False

WEIGHTS_PATH = r"C:\Study\emotion_mvc_app\emotion_mvc_app\model_weights\best.pt"

detector = FaceDetector() if USE_FACE_DETECTOR else None

CONFIDENCE_THRESHOLD = 0.4


def main() -> None:
    model = EmotionModel(weights_path=WEIGHTS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    view = EmotionApp()
    controller = EmotionController(model=model, view=view)

    model = EmotionModel(WEIGHTS_PATH, CONFIDENCE_THRESHOLD, face_detector=detector)
    controller = EmotionController(model=model, view=view, reaction=RobotReactionManager())

    threading.Thread(target=controller.load_model, daemon=True).start()

    view.mainloop()


if __name__ == "__main__":
    main()
