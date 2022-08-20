import cv2
import mediapipe as mp
import time


class MpFaceDetector:
    def __init__(self, min_threshold=0.5) -> None:
        self.min_threshold = min_threshold
        self.mp_detector = mp.solutions.face_detection
        self.detector = self.mp_detector.FaceDetection(self.min_threshold)

    def find_faces(self, frame, draw=True) -> tuple:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rezults = self.detector.process(frameRGB)
        h, w, c = frameRGB.shape
        bbox = []
        score = None
        if rezults.detections is not None:
            for detect_box in rezults.detections:
                score = detect_box.score
                bbox_relative = detect_box.location_data.relative_bounding_box
                bbox = (
                    int(bbox_relative.xmin * w),
                    int(bbox_relative.ymin * h),
                    int(bbox_relative.width * w),
                    int(bbox_relative.height * h),
                )
                if draw:
                    cv2.rectangle(frame, bbox, (255, 0, 255), 3)
                    cv2.putText(frame, f"Score : {int(score[0] * 100)}%",
                                (int(bbox_relative.xmin * w), int(bbox_relative.ymin * h) - 10),
                                cv2.FONT_HERSHEY_PLAIN,2, (255, 255, 0), 2,
                                )
        return frame, score, bbox


def face_detection(webcam=False) -> None:
    cap = cv2.VideoCapture(0)
    detector = MpFaceDetector()
    if not webcam:
        img = cv2.imread("../test.jpg")
        img = cv2.resize(img, (512, 512))
        img, score, bboxes = detector.find_faces(frame=img)
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
        frame, score, bboxes = detector.find_faces(frame=frame)
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
    face_detection()
