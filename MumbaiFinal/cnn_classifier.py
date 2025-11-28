"""
CNN-based Gesture Classifier for Hospital Sign Language System
Uses TensorFlow/Keras model for ML-based gesture recognition
Hybrid approach: CNN + Rule-based for comprehensive coverage
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import os

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[CNN] TensorFlow not available. Run: pip install tensorflow")


@dataclass
class CNNPrediction:
    """Represents a CNN model prediction"""
    class_id: int
    class_name: str
    confidence: float
    all_probabilities: List[float]


class GestureCNN:
    """
    CNN-based gesture classifier using pre-trained model.
    
    Features:
    - Uses VGG16 preprocessing for consistent results
    - Background subtraction for better hand segmentation
    - Prediction smoothing to reduce flickering
    - Confidence thresholding
    """
    
    # Default class mapping (matches training data)
    DEFAULT_CLASSES = {
        0: "One",      # 1 finger
        1: "Two",      # 2 fingers  
        2: "Three",    # 3 fingers
        3: "Four",     # 4 fingers
        4: "Five",     # 5 fingers/open palm
    }
    
    # Map CNN classes to hospital gesture IDs
    CNN_TO_HOSPITAL_GESTURE = {
        0: 3,   # One -> YES (index finger)
        1: 4,   # Two -> NO (peace sign)
        2: 20,  # Three -> INJECTION
        3: 21,  # Four -> BLOOD TEST
        4: 5,   # Five -> STOP/WAIT (open palm)
    }
    
    def __init__(self, model_path: str = "best_model_gesture.h5",
                 confidence_threshold: float = 0.6):
        """
        Initialize the CNN classifier.
        
        Args:
            model_path: Path to the trained Keras model
            confidence_threshold: Minimum confidence to accept prediction
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_available = False
        
        # Class names
        self.class_names = self.DEFAULT_CLASSES.copy()
        
        # Background subtraction for hand segmentation
        self.background = None
        self.accumulated_weight = 0.5
        self.calibration_frames = 0
        self.calibration_target = 60
        self.is_calibrated = False
        
        # ROI configuration
        self.roi_top = 100
        self.roi_bottom = 300
        self.roi_left = 400
        self.roi_right = 600
        
        # Prediction smoothing
        self.prediction_history: List[int] = []
        self.history_size = 10
        
        # Input size expected by model
        self.input_size = (64, 64)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained CNN model."""
        if not TF_AVAILABLE:
            print("[CNN] TensorFlow not available, CNN disabled")
            return
        
        if not os.path.exists(self.model_path):
            print(f"[CNN] Model file not found: {self.model_path}")
            return
        
        try:
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            self.model = keras.models.load_model(self.model_path, compile=False)
            self.is_available = True
            print(f"[CNN] ✅ Model loaded successfully from {self.model_path}")
            print(f"[CNN] Input shape: {self.model.input_shape}")
            print(f"[CNN] Output classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"[CNN] ❌ Failed to load model: {e}")
            self.is_available = False
    
    def set_roi(self, top: int, bottom: int, left: int, right: int):
        """Set the Region of Interest for hand detection."""
        self.roi_top = top
        self.roi_bottom = bottom
        self.roi_left = left
        self.roi_right = right
    
    def calibrate_background(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Calibrate background for hand segmentation.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            Tuple of (is_complete, progress_percentage)
        """
        # Extract ROI
        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Accumulate background
        if self.background is None:
            self.background = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, self.background, self.accumulated_weight)
        
        self.calibration_frames += 1
        progress = int((self.calibration_frames / self.calibration_target) * 100)
        
        if self.calibration_frames >= self.calibration_target:
            self.is_calibrated = True
            return True, 100
        
        return False, progress
    
    def reset_calibration(self):
        """Reset background calibration."""
        self.background = None
        self.calibration_frames = 0
        self.is_calibrated = False
        self.prediction_history = []
    
    def segment_hand(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Segment hand from background.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            Tuple of (thresholded_image, hand_contour) or None
        """
        if not self.is_calibrated or self.background is None:
            return None
        
        # Extract ROI
        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Background subtraction
        diff = cv2.absdiff(self.background.astype("uint8"), gray)
        
        # Threshold
        _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        thresholded = cv2.dilate(thresholded, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresholded.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Get largest contour (assumed to be hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Filter by area
        if cv2.contourArea(hand_contour) < 5000:
            return None
        
        return thresholded, hand_contour
    
    def preprocess_for_model(self, thresholded: np.ndarray) -> np.ndarray:
        """
        Preprocess segmented hand image for model input.
        
        Args:
            thresholded: Binary thresholded hand image
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        img = cv2.resize(thresholded, self.input_size)
        
        # Convert to RGB (3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Convert to float
        img = img.astype("float32")
        
        # Apply VGG16 preprocessing (same as training)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame: np.ndarray) -> Optional[CNNPrediction]:
        """
        Make prediction on a frame.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            CNNPrediction object or None
        """
        if not self.is_available or not self.is_calibrated:
            return None
        
        # Segment hand
        result = self.segment_hand(frame)
        if result is None:
            return None
        
        thresholded, _ = result
        
        # Preprocess
        preprocessed = self.preprocess_for_model(thresholded)
        
        # Predict
        predictions = self.model.predict(preprocessed, verbose=0)
        
        # Get results
        class_id = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None
        
        return CNNPrediction(
            class_id=class_id,
            class_name=self.class_names.get(class_id, f"Unknown_{class_id}"),
            confidence=confidence,
            all_probabilities=predictions[0].tolist()
        )
    
    def predict_with_smoothing(self, frame: np.ndarray) -> Optional[CNNPrediction]:
        """
        Make prediction with temporal smoothing.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            Smoothed CNNPrediction or None
        """
        prediction = self.predict(frame)
        
        if prediction is None:
            self.prediction_history = []
            return None
        
        # Add to history
        self.prediction_history.append(prediction.class_id)
        
        # Trim history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Need minimum history for smoothing
        if len(self.prediction_history) < 3:
            return prediction
        
        # Find most common prediction
        from collections import Counter
        counter = Counter(self.prediction_history)
        most_common_id, count = counter.most_common(1)[0]
        
        # Require majority
        if count < len(self.prediction_history) * 0.5:
            return None
        
        # Return smoothed prediction
        return CNNPrediction(
            class_id=most_common_id,
            class_name=self.class_names.get(most_common_id, f"Unknown_{most_common_id}"),
            confidence=prediction.confidence,
            all_probabilities=prediction.all_probabilities
        )
    
    def get_hospital_gesture_id(self, cnn_prediction: CNNPrediction) -> Optional[int]:
        """
        Map CNN prediction to hospital gesture ID.
        
        Args:
            cnn_prediction: CNN prediction result
            
        Returns:
            Hospital gesture ID or None
        """
        return self.CNN_TO_HOSPITAL_GESTURE.get(cnn_prediction.class_id)
    
    def draw_roi(self, frame: np.ndarray, color: Tuple[int, int, int] = (255, 128, 0)) -> np.ndarray:
        """Draw ROI rectangle on frame."""
        cv2.rectangle(
            frame,
            (self.roi_left, self.roi_top),
            (self.roi_right, self.roi_bottom),
            color, 2
        )
        return frame
    
    def draw_prediction(self, frame: np.ndarray, prediction: CNNPrediction) -> np.ndarray:
        """Draw prediction info on frame."""
        text = f"{prediction.class_name} ({prediction.confidence*100:.1f}%)"
        cv2.putText(
            frame, text,
            (self.roi_left, self.roi_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )
        return frame


class HybridGestureClassifier:
    """
    Hybrid classifier combining CNN and rule-based approaches.
    
    Strategy:
    1. Use MediaPipe for robust hand detection
    2. Use CNN for finger counting validation
    3. Use rule-based for complex gestures
    4. Combine confidences for final decision
    """
    
    def __init__(self, cnn_model_path: str = "best_model_gesture.h5"):
        """
        Initialize hybrid classifier.
        
        Args:
            cnn_model_path: Path to CNN model
        """
        # CNN classifier
        self.cnn = GestureCNN(cnn_model_path)
        
        # Import rule-based classifier
        try:
            from gesture_classifier import GestureClassifier, Gesture
            self.rule_classifier = GestureClassifier()
            self.rule_available = True
        except ImportError:
            self.rule_classifier = None
            self.rule_available = False
            print("[Hybrid] Rule-based classifier not available")
        
        # Fusion weights
        self.cnn_weight = 0.4
        self.rule_weight = 0.6
        
        # Stats
        self.cnn_predictions = 0
        self.rule_predictions = 0
        self.hybrid_predictions = 0
    
    def classify(self, frame: np.ndarray, landmarks: Optional[List] = None,
                prev_landmarks: Optional[List] = None) -> Optional[Dict]:
        """
        Classify gesture using hybrid approach.
        
        Args:
            frame: BGR frame from camera
            landmarks: MediaPipe hand landmarks (optional)
            prev_landmarks: Previous frame landmarks (optional)
            
        Returns:
            Dict with classification results or None
        """
        result = {
            "gesture_id": None,
            "gesture_name": None,
            "meaning": None,
            "voice_text": None,
            "confidence": 0.0,
            "source": None,  # 'cnn', 'rule', 'hybrid'
            "cnn_prediction": None,
            "rule_prediction": None,
        }
        
        # Get CNN prediction
        cnn_pred = None
        if self.cnn.is_available and self.cnn.is_calibrated:
            cnn_pred = self.cnn.predict_with_smoothing(frame)
            if cnn_pred:
                result["cnn_prediction"] = {
                    "class": cnn_pred.class_name,
                    "confidence": cnn_pred.confidence,
                    "gesture_id": self.cnn.get_hospital_gesture_id(cnn_pred)
                }
        
        # Get rule-based prediction
        rule_pred = None
        if self.rule_available and landmarks:
            rule_pred = self.rule_classifier.classify_with_smoothing(landmarks, prev_landmarks)
            if rule_pred:
                result["rule_prediction"] = {
                    "gesture_id": rule_pred.id,
                    "name": rule_pred.name,
                    "meaning": rule_pred.meaning,
                    "confidence": 0.9  # Rule-based has high confidence when matched
                }
        
        # Fusion logic
        if cnn_pred and rule_pred:
            # Both predictions available - check agreement
            cnn_gesture_id = self.cnn.get_hospital_gesture_id(cnn_pred)
            
            if cnn_gesture_id == rule_pred.id:
                # Agreement - high confidence
                result["gesture_id"] = rule_pred.id
                result["gesture_name"] = rule_pred.name
                result["meaning"] = rule_pred.meaning
                result["voice_text"] = rule_pred.voice_text
                result["confidence"] = min(1.0, cnn_pred.confidence * 0.5 + 0.5)
                result["source"] = "hybrid"
                self.hybrid_predictions += 1
            else:
                # Disagreement - prefer rule-based for complex gestures
                result["gesture_id"] = rule_pred.id
                result["gesture_name"] = rule_pred.name
                result["meaning"] = rule_pred.meaning
                result["voice_text"] = rule_pred.voice_text
                result["confidence"] = 0.8
                result["source"] = "rule"
                self.rule_predictions += 1
                
        elif rule_pred:
            # Only rule-based available
            result["gesture_id"] = rule_pred.id
            result["gesture_name"] = rule_pred.name
            result["meaning"] = rule_pred.meaning
            result["voice_text"] = rule_pred.voice_text
            result["confidence"] = 0.85
            result["source"] = "rule"
            self.rule_predictions += 1
            
        elif cnn_pred:
            # Only CNN available - map to hospital gesture
            gesture_id = self.cnn.get_hospital_gesture_id(cnn_pred)
            if gesture_id and self.rule_available:
                gesture = self.rule_classifier.gestures.get(gesture_id)
                if gesture:
                    result["gesture_id"] = gesture_id
                    result["gesture_name"] = gesture.name
                    result["meaning"] = gesture.meaning
                    result["voice_text"] = gesture.voice_text
                    result["confidence"] = cnn_pred.confidence
                    result["source"] = "cnn"
                    self.cnn_predictions += 1
        
        # Return None if no valid prediction
        if result["gesture_id"] is None:
            return None
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        total = self.cnn_predictions + self.rule_predictions + self.hybrid_predictions
        return {
            "total_predictions": total,
            "cnn_only": self.cnn_predictions,
            "rule_only": self.rule_predictions,
            "hybrid": self.hybrid_predictions,
            "cnn_percentage": f"{(self.cnn_predictions/total*100):.1f}%" if total > 0 else "0%",
            "rule_percentage": f"{(self.rule_predictions/total*100):.1f}%" if total > 0 else "0%",
            "hybrid_percentage": f"{(self.hybrid_predictions/total*100):.1f}%" if total > 0 else "0%",
        }
    
    def calibrate_cnn(self, frame: np.ndarray) -> Tuple[bool, int]:
        """Calibrate CNN background."""
        return self.cnn.calibrate_background(frame)
    
    def reset_cnn_calibration(self):
        """Reset CNN calibration."""
        self.cnn.reset_calibration()


# Test the CNN classifier
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Testing CNN Gesture Classifier")
    print("=" * 60)
    
    # Initialize classifier
    cnn = GestureCNN()
    
    if not cnn.is_available:
        print("\n⚠️ CNN model not available. Exiting test.")
        exit()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()
    
    print("\n📹 Camera opened. Starting calibration...")
    print("   Keep your hand OUT of the ROI box during calibration.")
    print("   Press ESC to exit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Calibration phase
        if not cnn.is_calibrated:
            is_done, progress = cnn.calibrate_background(frame)
            cv2.putText(frame, f"Calibrating: {progress}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if is_done:
                print("✅ Calibration complete! Show gestures in the ROI box.")
        else:
            # Make prediction
            prediction = cnn.predict_with_smoothing(frame)
            
            if prediction:
                frame = cnn.draw_prediction(frame, prediction)
                hospital_id = cnn.get_hospital_gesture_id(prediction)
                print(f"Detected: {prediction.class_name} ({prediction.confidence:.2%}) -> Hospital ID: {hospital_id}")
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw ROI
        frame = cnn.draw_roi(frame)
        
        cv2.imshow("CNN Gesture Test", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ CNN test complete!")

