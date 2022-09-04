import cv2
import mediapipe as mp
import time
import math
import numpy as np


class MpPoseEstimator:
    def __init__(self,
                 min_threshold_tracking=0.5,
                 min_threshold_detection=0.5,
                 thickness=1,
                 color=[255, 0, 255]) -> None:
        self.min_threshold_tracking = min_threshold_tracking
        self.min_threshold_detection = min_threshold_detection
        self.thickness = thickness
        self.color = [color[2], color[1], color[0]]
        self.styles = mp.solutions.drawing_styles
        self.draw = mp.solutions.drawing_utils
        self.draw_spec = self.draw.DrawingSpec(
            thickness=self.thickness,
            color=self.color
                                               )
        self.mp_pose_traking = mp.solutions.pose
        self.pose_traking = self.mp_pose_traking.Pose(
            min_tracking_confidence=self.min_threshold_tracking,
            min_detection_confidence=self.min_threshold_detection
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


def pose_estimator(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    if not webcam:
        tracking = MpPoseEstimator()
        img = cv2.imread("../pose.jpg")
        img = cv2.resize(img, (512, 512))
        img = tracking.find_pose(frame=img)
        coords = tracking.find_position(img)
        print(coords)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpPoseEstimator()
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
    pose_estimator(webcam=False)
