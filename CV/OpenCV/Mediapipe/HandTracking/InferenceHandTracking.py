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
mpHandTrack = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
HandTrack = mpHandTrack.Hands(max_num_hands=2,
                             min_tracking_confidence=0.6,
                             min_detection_confidence=0.6)

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    rezults = HandTrack.process(frameRGB)
    print(rezults)
    if rezults.multi_hand_landmarks is not None:
        for hand_landmarks in rezults.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 255), -1)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mpHandTrack.HAND_CONNECTIONS,
                connection_drawing_spec=draw_spec
            )
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=hand_landmarks,
            #     connections=mpHandTrack.HAND_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            #    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            # )
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
