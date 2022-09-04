import os
import numpy as np
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
webcam = True

if not cap.isOpened():
    print("Video capture not found")
    exit()

pTime = 0
mp_drawing = mp.solutions.drawing_utils
mpSelfieSegmentTrack = mp.solutions.selfie_segmentation
SelfieSegmentTrack = mpSelfieSegmentTrack.SelfieSegmentation(model_selection=1)
color_frame = False
imgs = ["background-bar.jpg", "background-sls.jpg", "background-tropic.jpg"]
BG_COLOR = [0, 0, 128]
bg_image = None
while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape

    rezults = SelfieSegmentTrack.process(frame)
    condition = np.stack((rezults.segmentation_mask,) * 3, axis=-1) > 0.1
    if color_frame:
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    else:
        bg_image = cv2.imread(fr"../{imgs[1]}")
        bg_image = cv2.resize(bg_image, (640, 480))
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    frame = np.where(condition, frame, bg_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
