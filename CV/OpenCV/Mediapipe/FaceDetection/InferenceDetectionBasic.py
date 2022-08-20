import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
webcam = True

if not cap.isOpened():
    print("Video capture not found")
    exit()

pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetect = mpFaceDetection.FaceDetection(0.6)

while True:
    succes, frame = cap.read()
    if not succes:
        print("Unfortunaly you can't found your video camera")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rezults = faceDetect.process(frameRGB)
    if rezults.detections is not None:
        for detect_box in rezults.detections:
            score = detect_box.score
            mpDraw.draw_detection(frame, detect_box)
            bbox_relative = detect_box.location_data.relative_bounding_box
            h, w, c = frame.shape
            bbox = (
                int(bbox_relative.xmin * w),
                int(bbox_relative.ymin * h),
                int(bbox_relative.width * w),
                int(bbox_relative.height * h),
            )
            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, f"Score : {int(score[0] * 100)}%",
                        (int(bbox_relative.xmin * w), int(bbox_relative.ymin * h) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2,
                        )
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
    )
    cv2.imshow("output", frame)
    cv2.waitKey(1)

cap.release()
