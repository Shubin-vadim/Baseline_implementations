import cv2
import mediapipe as mp
import time
import numpy as np


class MpObjectron:
    def __init__(self, max_num_objects=1,
                 min_threshold=0.5,
                 thickness=1,
                 color=[255, 0, 255],
                 model_name="Cup") -> None:
        self.max_hands = max_num_objects
        self.min_threshold = min_threshold
        self.thickness = thickness
        self.color = [color[2], color[1], color[0]]
        self.max_num_objects = max_num_objects
        self.model_name = model_name
        self.styles = mp.solutions.drawing_styles
        self.draw = mp.solutions.drawing_utils
        self.draw_spec = self.draw.DrawingSpec(thickness=self.thickness,
                                               color=self.color)
        self.mp_hand_traking = mp.solutions.objectron
        self.hand_traking = self.mp_hand_traking.Objectron(
            max_num_objects=self.max_num_objects,
            min_detection_confidence=self.min_threshold
        )
        self.rezults = None

    def find_object(self, frame, draw=True) -> np.ndarray:
        h, w, c = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rezults = self.hand_traking.process(frameRGB)
        if self.rezults.detected_objects is not None:
            for detected_objects in self.rezults.detected_objects:
                if draw:
                    self.draw.draw_landmarks(
                        image=frame,
                        landmark_list=detected_objects.landmarks_2d,
                        connections=self.mp_hand_traking.BOX_CONNECTIONS,
                        connection_drawing_spec=self.draw_spec
                    )

                    self.draw.draw_axis(image=frame,
                                        rotation=detected_objects.rotation,
                                        translation=detected_objects.translation
                                        )
        return frame


def objectron_detection(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    if not webcam:
        detector = MpObjectron(max_num_objects=1)
        img = cv2.imread("../cup.jpg")
        img = cv2.resize(img, (512, 512))
        img = detector.find_object(frame=img)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpObjectron(max_num_objects=1)
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break
        frame = tracking.find_object(frame=frame, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS {int(fps)}", (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2,
                    )
        cv2.imshow("output", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objectron_detection(webcam=False)
