import cv2
import mediapipe as mp
import time
import numpy as np


class MpSelfieSSegmentation:
    def __init__(self, model=1) -> None:
        self.model = model
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
          model_selection=self.model
        )
        self.rezults = None

    def find_pose(self, frame, background=(255, 0, 255), threshold=0.1) -> np.ndarray:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if background is None:
            return frame

        self.rezults = self.selfie_segmentation.process(frameRGB)
        h, w, c = frame.shape
        bg_image = None
        condition = np.stack((self.rezults.segmentation_mask,) * 3, axis=-1) > threshold
        if isinstance(background, tuple):
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = background
        else:
            bg_image = cv2.resize(background, (w, h))

        frame = np.where(condition, frame, bg_image)
        return frame


def selfie_segmentation(webcam=True) -> None:
    cap = cv2.VideoCapture(0)
    if not webcam:
        tracking = MpSelfieSSegmentation()
        img = cv2.imread("../test_hands.jpg")
        img = cv2.resize(img, (512, 512))
        img = tracking.find_pose(frame=img)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        exit()
    if not cap.isOpened():
        print("Video camera not found")
        exit()
    tracking = MpSelfieSSegmentation()
    # bg_img = cv2.imread("../background-sls.jpg")
    bg_img = (255, 0, 0)
    pTime = 0
    while True:
        succes, frame = cap.read()
        if not succes:
            print("Cap not reading")
            break

        frame = tracking.find_pose(frame=frame, background=bg_img)
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
    selfie_segmentation(webcam=True)
