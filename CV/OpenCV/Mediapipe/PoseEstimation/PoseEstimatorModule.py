import cv2
import mediapipe as mp
import time
import math
import numpy as np


class MpSelfieSegmentation:
    def __init__(self,
                 color=[255, 0, 255],
                 model_selection=0,
                 ) -> None:
        self.model_selection = model_selection
        self.color = [color[2], color[1], color[0]]
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=self.model_selection
        )
        self.rezults = None
        self.lmList = None

    def find_pose(self, frame, draw=True) -> np.ndarray:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rezults = self.pose_traking.process(frameRGB)
        if self.rezults.pose_landmarks is not None:
            if draw:
                self.draw.draw_landmarks(
                    image=frame,
                    landmark_list=self.rezults.pose_landmarks,
                    connections=self.mp_pose_traking.POSE_CONNECTIONS,
                    connection_drawing_spec=self.draw_spec
                    )
        return frame

    def find_position(self, frame, draw=True) -> list:
        h, w, c = frame.shape
        self.lmList = []
        if self.rezults.pose_landmarks is not None:
            for id, lm in enumerate(self.rezults.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, self.color, -1)
        return self.lmList


def selfie_segmentation(img=True, color=[0, 0, 255], imgs=[]) -> None:
    cap = cv2.VideoCapture(0)
    if not img:
        tracking = MpSelfieSegmentation()
        img = cv2.imread("../pose.jpg")
        img = cv2.resize(img, (512, 512))
        img = tracking.find_pose(frame=img)
        coords = tracking.find_position(img)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpSelfieSegmentation()
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break
        frame = tracking.find_pose(frame=frame, draw=True)
        coords = tracking.find_position(frame)
        print(coords)
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
    imgs = ["../background-bar.jpg", "../background-sls.jpg", "../background-tropic.jpg"]
    selfie_segmentation(img=False, imgs=imgs)
