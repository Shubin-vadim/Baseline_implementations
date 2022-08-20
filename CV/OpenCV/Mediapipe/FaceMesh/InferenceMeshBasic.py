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
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=[255, 0,0 ])
mpFaceMesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10,
                               min_detection_confidence=0.6
                               )

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h,w,c = frame.shape
    rezults = faceMesh.process(frameRGB)
    print(rezults)
    if rezults.multi_face_landmarks is not None:
        for face_landmarks in rezults.multi_face_landmarks:
            for i in range(0, 468):
                x = int(face_landmarks.landmark[i].x * w)
                y = int(face_landmarks.landmark[i].y * h)
                cv2.circle(frame, (x,y), 2, (255,0,255),-1)
                print(x,y)
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mpFaceMesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=draw_spec,
            #     connection_drawing_spec=draw_spec)

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
