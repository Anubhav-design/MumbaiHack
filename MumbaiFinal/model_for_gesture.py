import numpy as np
import cv2
import keras
import tensorflow as tf

# Load model
model = keras.models.load_model(r"best_model_gesture.h5")

# Label mapping
word_dict = {0:'One', 1:'Two', 2:'Three', 3:'Four', 4:'Five'}

# Background averaging setup
background = None
accumulated_weight = 0.5

# ROI coordinates
ROI_top, ROI_bottom, ROI_left, ROI_right = 100, 300, 150, 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
    thresholded = cv2.dilate(thresholded, kernel, iterations=1)

    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    hand_segment = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand_segment) < 5000:  
        return None

    return thresholded, hand_segment


cam = cv2.VideoCapture(0)
num_frames = 0

# OPTIONAL: prediction smoothing
pred_history = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_left:ROI_right]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 70:
        cal_accum_avg(gray_frame, accumulated_weight)
        progress = int((num_frames / 70) * 100)
        cv2.putText(frame_copy, f"Calibrating background: {progress}%", 
                    (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    else:
        hand = segment_hand(gray_frame)

        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, 
                             [hand_segment + (ROI_left, ROI_top)], 
                             -1, (255, 0, 0), 2)

            cv2.imshow("Thresholded Hand", thresholded)

            # Resize and convert to RGB
            img = cv2.resize(thresholded, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype("float32")

            # Preprocess same as training
            img = tf.keras.applications.vgg16.preprocess_input(img)

            img = np.expand_dims(img, axis=0)

            # Predict ONCE
            pred = model.predict(img, verbose=0)
            pred_label = int(np.argmax(pred))
            confidence = float(np.max(pred))

            # Smooth predictions (optional)
            pred_history.append(pred_label)
            if len(pred_history) > 12:
                pred_history.pop(0)

            smoothed_label = max(set(pred_history), key=pred_history.count)

            cv2.putText(frame_copy, 
                        f"{word_dict[smoothed_label]} ({confidence*100:.1f}%)",
                        (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        else:
            cv2.putText(frame_copy, "No hand detected", 
                        (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ROI Box
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 2)
    cv2.putText(frame_copy, "Sign Language Recognition", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51,255,51), 1)

    cv2.imshow("Gesture Recognition", frame_copy)

    num_frames += 1
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
