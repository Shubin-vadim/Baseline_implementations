import cv2
import mediapipe as mp
import time


class MpFaceMesh:
    def __init__(self, max_faces=1, min_threshold=0.5, thickness=1, circle_radius=1, color=[255, 0, 0]) -> None:
        self.max_faces = max_faces
        self.min_threshold = min_threshold
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.color = [color[2], color[1], color[0]]
        self.styles = mp.solutions.drawing_styles
        self.draw = mp.solutions.drawing_utils
        self.draw_spec = self.draw.DrawingSpec(thickness=self.thickness,
                                               circle_radius=self.circle_radius,
                                               color=self.color)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.faceMesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_faces,
            min_detection_confidence=self.min_threshold
        )

    def find_mesh(self, frame, points_draw=True):
        h, w, c = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rezults = self.faceMesh.process(frameRGB)
        if rezults.multi_face_landmarks is not None:
            for face_landmarks in rezults.multi_face_landmarks:
                if points_draw:
                    for i in range(0, 468):
                        x = int(face_landmarks.landmark[i].x * w)
                        y = int(face_landmarks.landmark[i].y * h)
                        cv2.circle(frame, (x, y), self.circle_radius, self.color, self.thickness)
                else:
                    self.draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.draw_spec,
                    connection_drawing_spec=self.draw_spec)
        return frame


def face_mesh(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    detector = MpFaceMesh()
    if not webcam:
        img = cv2.imread("../test.jpg")
        img = cv2.resize(img, (512, 512))
        img = detector.find_faces(frame=img)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break
        frame = detector.find_mesh(frame=frame, points_draw=False)
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
    face_mesh()
