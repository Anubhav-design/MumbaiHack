import cv2
import numpy as np
import os
from pathlib import Path
# ROI settings
ROI_top = 100
ROI_bottom = 300
ROI_left = 150
ROI_right = 350

ACCUMULATED_WEIGHT = 0.5
BACKGROUND_FRAMES = 60
THRESHOLD = 25
MAX_IMAGES = 100

# Save directory
gesture_number = 1 # Change this for different gestures
save_dir = Path(fr"D:\gesture\test\{gesture_number}")
save_dir.mkdir(parents=True, exist_ok=True)

background = None
num_frames = 0
num_saved = 0


def accumulate_background(frame):
    global background
    if background is None:
        background = frame.astype("float")
        return
    cv2.accumulateWeighted(frame, background, ACCUMULATED_WEIGHT)


def segment(frame):
    global background
    if background is None:
        return None

    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, th = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    return th, max(contours, key=cv2.contourArea)


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 40, 60])
    upper = np.array([20, 255, 255])

    skin_mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # ---- Finger contour detection ----
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finger_contour = None
    best_ratio = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        ratio = max(w, h) / (min(w, h) + 1)

        # Finger is long + thin → large ratio
        if ratio > best_ratio:
            best_ratio = ratio
            finger_contour = cnt

    # ---- Extract only the finger in color ----
    finger_mask = np.zeros_like(skin_mask)
    if finger_contour is not None:
        cv2.drawContours(finger_mask, [finger_contour], -1, 255, -1)

    finger_color = cv2.bitwise_and(roi, roi, mask=finger_mask)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    if num_frames < BACKGROUND_FRAMES:
        accumulate_background(gray)
        cv2.putText(frame, "KEEP HAND OUT OF ROI – Capturing background...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        result = segment(gray)
        if result is not None:
            thresh, hand_contour = result

            # Draw contour
            offset = np.array([ROI_left, ROI_top])
            cv2.drawContours(frame, [hand_contour + offset], -1, (255,0,0), 2)

            cv2.imshow("Threshold", thresh)

            if num_saved < MAX_IMAGES:
                color_filename = save_dir / f"color_{num_saved:04d}.jpg"
                cv2.imwrite(str(color_filename), finger_color)

                #mask_filename = save_dir / f"mask_{num_saved:04d}.jpg"
                #cv2.imwrite(str(mask_filename), thresh)

                num_saved += 1
                cv2.putText(frame, f"Saving: {num_saved}/{MAX_IMAGES}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                print("Done saving images.")
                break
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 2)
    cv2.imshow("Hand Capture", frame)

    num_frames += 1

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
