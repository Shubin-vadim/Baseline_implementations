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
draw_spec = mp_drawing.DrawingSpec(thickness=3, color=[255, 0, 0])
mpHandTrack = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
HandTrack = mpHandTrack.Pose(
                             min_tracking_confidence=0.6,
                             min_detection_confidence=0.6
                            )

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    rezults = HandTrack.process(frameRGB)
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
