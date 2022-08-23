import cv2
import mediapipe as mp
import time
import math
import numpy as np


class MpHolistic:
    def __init__(self,
                 min_threshold_tracking=0.5,
                 min_threshold_detection=0.5,
                 thickness=1,
                 color_pose=[255, 0, 255],
                 color_mesh=[255, 0, 0]) -> None:
        self.min_threshold_tracking = min_threshold_tracking
        self.min_threshold_detection = min_threshold_detection
        self.thickness = thickness
        self.color_pose = [color_pose[2], color_pose[1], color_pose[0]]
        self.color_mesh = [color_pose[2], color_pose[1], color_pose[0]]
        self.styles = mp.solutions.drawing_styles
        self.draw = mp.solutions.drawing_utils
        self.draw_spec_mesh = self.draw.DrawingSpec(
            thickness=self.thickness,
            color=self.color_mesh
                                               )
        self.draw_spec_pose = self.draw.DrawingSpec(
            thickness=self.thickness,
            color=self.color_pose
        )
        self.mp_holistic = mp.solutions.holistic
        self.holistic_tracking = self.mp_holistic.Holistic(
            min_tracking_confidence=self.min_threshold_tracking,
            min_detection_confidence=self.min_threshold_detection
        )
        self.rezults = None
        self.list_pose = None
        self.list_mesh = None

    def find_holistic(self, frame, draw_pose=True, draw_mesh=True) -> np.ndarray:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rezults = self.holistic_tracking.process(frameRGB)
        if self.rezults.pose_landmarks is not None:
            if draw_pose:
                self.draw.draw_landmarks(
                    image=frame,
                    landmark_list=self.rezults.pose_landmarks,
                    connections=self.mp_holistic.POSE_CONNECTIONS,
                    connection_drawing_spec=self.draw_spec_pose
                    )
        if self.rezults.face_landmarks is not None:
            if draw_mesh:
                self.draw.draw_landmarks(
                    image=frame,
                    landmark_list=self.rezults.face_landmarks,
                    connections=self.mp_holistic.FACEMESH_CONTOURS,
                    connection_drawing_spec=self.draw_spec_mesh
                    )
        return frame

    def find_pose_position(self, frame, draw=True) -> list:
        h, w, c = frame.shape
        self.list_pose = []
        if self.rezults.pose_landmarks is not None:
            for id, lm in enumerate(self.rezults.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.list_pose.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), self.thickness, self.color_pose, -1)
        return self.list_pose

    def find_mesh_position(self, frame, draw=True) -> list:
        h, w, c = frame.shape
        self.list_mesh = []
        if self.rezults.face_landmarks is not None:
            for id, lm in enumerate(self.rezults.face_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.list_mesh.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), self.thickness, self.color_mesh, -1)
        return self.list_mesh


def holistic_tracking(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    if not webcam:
        tracking = MpHolistic()
        img = cv2.imread("../pose.jpg")
        img = cv2.resize(img, (512, 512))
        img = tracking.find_holistic(frame=img)
        # coords = tracking.find_position(img)
        # print(coords)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpHolistic(color_pose=[0,0,255],color_mesh=[0,255,0])
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break
        frame = tracking.find_holistic(frame=frame, draw_pose=True, draw_mesh=False)
        coords_pose = tracking.find_pose_position(frame)
        coords_mesh = tracking.find_mesh_position(frame)
        print(f"Coords pose: {coords_pose}")
        print(f"Coords mesh: {coords_mesh}")
        # coords = tracking.find_position(frame)
        # print(coords)
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
    holistic_tracking(webcam=True)
