import cv2
import mediapipe as mp
import time
import glob

import numpy as np

cap = cv2.VideoCapture(0)
webcam = True

if not cap.isOpened():
    print("Video capture not found")
    exit()

pTime = 0
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=3, color=[255, 0, 0])
mpSelfieSegmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
SelfieSegmentation = mpSelfieSegmentation.SelfieSegmentation(model_selection=0)
path_dir = "../"
images = glob.glob(path_dir + "background-*.jpg")
print(images)

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    rezults = SelfieSegmentation.process(frameRGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    condition = np.stack((rezults.segmentation_mask, ) * 3, axis=-1) > 0.1

    print(rezults)
    if rezults.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=rezults.pose_landmarks,
                connections=mpHandTrack.POSE_CONNECTIONS,
               connection_drawing_spec=draw_spec
                )
            for idx, landmark in enumerate(rezults.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255,0,255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
    )
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()