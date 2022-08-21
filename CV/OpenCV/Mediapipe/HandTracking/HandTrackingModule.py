import cv2
import mediapipe as mp
import time

import numpy as np


class MpHandTracking:
    def __init__(self, max_hands=2,
                 min_threshold=0.5,
                 thickness=1,
                 color=[255, 0, 255]) -> None:
        self.max_hands = max_hands
        self.min_threshold = min_threshold
        self.thickness = thickness
        self.color = [color[2], color[1], color[0]]
        self.styles = mp.solutions.drawing_styles
        self.draw = mp.solutions.drawing_utils
        self.draw_spec = self.draw.DrawingSpec(thickness=self.thickness,
                                               color=self.color)
        self.mp_hand_traking = mp.solutions.hands
        self.hand_traking = self.mp_hand_traking.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_threshold
        )
        self.rezults = None

    def find_hands(self, frame, draw=True) -> np.ndarray:
        h, w, c = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rezults = self.hand_traking.process(frameRGB)
        if self.rezults.multi_hand_landmarks is not None:
            for hand_landmarks in self.rezults.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                            cv2.circle(frame, (cx, cy), self.thickness,  self.color, -1)
                    self.draw.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hand_traking.HAND_CONNECTIONS,
                        connection_drawing_spec=self.draw_spec
                    )
        return frame

    def find_position(self, frame, hand=0, draw=True) -> list:
        h, w, c = frame.shape
        lmList = []
        if self.rezults.multi_hand_landmarks is not None:
            myHand = self.rezults.multi_hand_landmarks[hand]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), self.thickness, self.color, -1)
        return lmList


def hand_tracking(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    if not webcam:
        detector = MpHandTracking(max_hands=2)
        img = cv2.imread("../test_hands.jpg")
        img = cv2.resize(img, (512, 512))
        img = detector.find_hands(frame=img)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpHandTracking(max_hands=2)
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break
        frame = tracking.find_hands(frame=frame, draw=True)
        position_l = tracking.find_position(frame)
        if len(position_l) > 0:
            print(position_l)
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
    hand_tracking(webcam=False)
