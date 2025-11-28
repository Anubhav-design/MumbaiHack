"""
Gesture Detector using MediaPipe Hands
Detects 21 hand landmarks in real-time
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class HandDetector:
    """
    MediaPipe-based hand detector for real-time hand landmark detection.
    Detects 21 landmarks per hand with high accuracy.
    """
    
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7, 
                 tracking_confidence: float = 0.7):
        """
        Initialize the hand detector.
        
        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence threshold
            tracking_confidence: Minimum tracking confidence threshold
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Landmark indices for fingers
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_pips = [3, 6, 10, 14, 18]  # PIP joints (for bend detection)
        self.finger_mcps = [2, 5, 9, 13, 17]   # MCP joints (knuckles)
        
    def find_hands(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, Optional[List]]:
        """
        Detect hands in the frame and optionally draw landmarks.
        
        Args:
            frame: BGR image from webcam
            draw: Whether to draw landmarks on the frame
            
        Returns:
            Tuple of (processed frame, list of hand landmarks)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        all_hands = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (left/right)
                handedness = results.multi_handedness[hand_idx].classification[0].label
                
                # Extract landmark positions
                landmarks = []
                h, w, c = frame.shape
                
                for lm in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'px': cx,
                        'py': cy
                    })
                
                hand_data = {
                    'landmarks': landmarks,
                    'handedness': handedness,
                    'raw_landmarks': hand_landmarks
                }
                all_hands.append(hand_data)
                
                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )
        
        return frame, all_hands if all_hands else None
    
    def get_finger_states(self, landmarks: List) -> List[bool]:
        """
        Determine which fingers are extended (up) or folded (down).
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky] - True if extended
        """
        fingers = []
        
        # Thumb - compare x position (special case due to thumb orientation)
        # For right hand: tip.x < pip.x means extended
        # For left hand: tip.x > pip.x means extended
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Use x-coordinate comparison for thumb
        if landmarks[0]['x'] < 0.5:  # Right hand (mirrored)
            fingers.append(thumb_tip['x'] < thumb_ip['x'])
        else:  # Left hand
            fingers.append(thumb_tip['x'] > thumb_ip['x'])
        
        # Other fingers - compare y position (tip above pip = extended)
        for tip_idx, pip_idx in zip(self.finger_tips[1:], self.finger_pips[1:]):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            fingers.append(tip['y'] < pip['y'])
        
        return fingers
    
    def count_fingers(self, landmarks: List) -> int:
        """
        Count the number of extended fingers.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            Number of extended fingers (0-5)
        """
        finger_states = self.get_finger_states(landmarks)
        return sum(finger_states)
    
    def get_hand_center(self, landmarks: List) -> Tuple[int, int]:
        """
        Get the center point of the palm.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            Tuple of (x, y) pixel coordinates for palm center
        """
        # Palm center is approximately at landmark 9 (middle finger MCP)
        palm = landmarks[9]
        return palm['px'], palm['py']
    
    def get_bounding_box(self, landmarks: List) -> Tuple[int, int, int, int]:
        """
        Get bounding box around the hand.
        
        Args:
            landmarks: List of 21 hand landmarks
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = [lm['px'] for lm in landmarks]
        y_coords = [lm['py'] for lm in landmarks]
        
        padding = 20
        return (
            max(0, min(x_coords) - padding),
            max(0, min(y_coords) - padding),
            max(x_coords) + padding,
            max(y_coords) + padding
        )
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


# Test the detector
if __name__ == "__main__":
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame, hands = detector.find_hands(frame)
        
        if hands:
            hand = hands[0]
            fingers = detector.get_finger_states(hand['landmarks'])
            count = detector.count_fingers(hand['landmarks'])
            
            cv2.putText(frame, f"Fingers: {count}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"States: {fingers}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Hand Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

