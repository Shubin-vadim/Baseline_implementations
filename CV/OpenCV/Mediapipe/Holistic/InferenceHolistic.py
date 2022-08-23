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
mpHolistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles
HolisticDetector = mpHolistic.Holistic(
                             min_tracking_confidence=0.6,
                             min_detection_confidence=0.6,
                            )

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    rezults = HolisticDetector.process(frameRGB)
    print(rezults)
    if rezults.face_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=rezults.face_landmarks,
                connections=mpHolistic.FACEMESH_CONTOURS,
               connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

    if rezults.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=rezults.pose_landmarks,
            connections=mpHolistic.POSE_CONNECTIONS,
            landmark_drawing_spec=draw_spec
        )
    # mp_drawing.plot_landmarks(
    #     rezults.pose_world_landmarks, mpHolistic.POSE_CONNECTIONS
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
