"""
Gesture Classifier for Hospital Sign Language
Classifies 40 different gestures for deaf patient communication
EXPANDED VERSION with more gestures
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class GestureCategory(Enum):
    """Categories of gestures for hospital use"""
    EMERGENCY = "🚨 Emergency"
    BASIC = "✅ Basic"
    MEDICAL = "🏥 Medical"
    FEELINGS = "😊 Feelings"
    PEOPLE = "👥 People"
    BODY = "🦴 Body Parts"
    TIME = "⏰ Time"


@dataclass
class Gesture:
    """Represents a hospital gesture"""
    id: int
    name: str
    meaning: str
    category: GestureCategory
    voice_text: str  # What to speak
    priority: int = 0  # Higher = more urgent (for emergency gestures)
    finger_pattern: str = ""  # For reference: T=thumb, I=index, M=middle, R=ring, P=pinky
    

class GestureClassifier:
    """
    Classifies hand gestures based on MediaPipe landmarks.
    Uses finger states, angles, and hand positions for classification.
    """
    
    def __init__(self):
        """Initialize the gesture classifier with 40 hospital gestures."""
        self.gestures = self._define_gestures()
        self.gesture_history = []
        self.history_size = 8  # Reduced for faster response
        self.last_gesture = None
        self.gesture_stable_count = 0
        self.stability_threshold = 3  # Reduced for faster detection
        
    def _define_gestures(self) -> Dict[int, Gesture]:
        """Define all 40 hospital gestures."""
        gestures = {
            # ==================== EMERGENCY (0-2) ====================
            0: Gesture(0, "FIST", "HELP / EMERGENCY", GestureCategory.EMERGENCY,
                      "Help! Emergency! Patient needs immediate assistance!", 
                      priority=10, finger_pattern="-----"),
            1: Gesture(1, "FIST_RAISED", "URGENT", GestureCategory.EMERGENCY,
                      "This is urgent! Please come quickly!", 
                      priority=9, finger_pattern="-----"),
            2: Gesture(2, "PAIN_ALERT", "SEVERE PAIN", GestureCategory.EMERGENCY,
                      "I am in severe pain! Need help now!", 
                      priority=10, finger_pattern="TIMRP"),
            
            # ==================== BASIC COMMUNICATION (3-12) ====================
            3: Gesture(3, "ONE_FINGER", "YES", GestureCategory.BASIC,
                      "Yes.", priority=1, finger_pattern="-I---"),
            4: Gesture(4, "TWO_FINGERS", "NO", GestureCategory.BASIC,
                      "No.", priority=1, finger_pattern="-IM--"),
            5: Gesture(5, "OPEN_PALM", "STOP / WAIT", GestureCategory.BASIC,
                      "Please stop. Please wait.", priority=2, finger_pattern="TIMRP"),
            6: Gesture(6, "THUMBS_UP", "GOOD / OK", GestureCategory.BASIC,
                      "I am feeling good. Everything is okay.", priority=1, finger_pattern="T----"),
            7: Gesture(7, "THUMBS_DOWN", "BAD / NOT OK", GestureCategory.BASIC,
                      "I am not feeling well. Something is wrong.", priority=3, finger_pattern="T----"),
            8: Gesture(8, "WAVE", "HELLO / GOODBYE", GestureCategory.BASIC,
                      "Hello. Goodbye.", priority=1, finger_pattern="TIMRP"),
            9: Gesture(9, "I_LOVE_YOU", "THANK YOU", GestureCategory.BASIC,
                      "Thank you very much.", priority=1, finger_pattern="TI--P"),
            10: Gesture(10, "PRAYER", "PLEASE", GestureCategory.BASIC,
                       "Please. I request.", priority=1, finger_pattern="TIMRP"),
            11: Gesture(11, "PINCH", "LITTLE / SMALL", GestureCategory.BASIC,
                       "A little bit. Small amount.", priority=1, finger_pattern="TI---"),
            12: Gesture(12, "OPEN_CLOSE", "MORE", GestureCategory.BASIC,
                       "I need more. Give me more.", priority=1, finger_pattern="TIMRP"),
            
            # ==================== MEDICAL NEEDS (13-27) ====================
            13: Gesture(13, "CALL_SIGN", "CALL DOCTOR", GestureCategory.MEDICAL,
                       "Please call the doctor. I need to see the doctor.", 
                       priority=5, finger_pattern="T---P"),
            14: Gesture(14, "PAIN_SIGN", "PAIN", GestureCategory.MEDICAL,
                       "I am in pain. It hurts.", priority=7, finger_pattern="TIMRP"),
            15: Gesture(15, "POINT_UP", "WATER", GestureCategory.MEDICAL,
                       "I need water please.", priority=2, finger_pattern="-I---"),
            16: Gesture(16, "FIVE_SPREAD", "MEDICINE", GestureCategory.MEDICAL,
                       "I need my medicine please.", priority=4, finger_pattern="TIMRP"),
            17: Gesture(17, "CROSSED_FINGERS", "NURSE", GestureCategory.MEDICAL,
                       "Please call the nurse.", priority=4, finger_pattern="-IM--"),
            18: Gesture(18, "OK_SIGN", "BATHROOM", GestureCategory.MEDICAL,
                       "I need to use the bathroom.", priority=3, finger_pattern="TI---"),
            19: Gesture(19, "PEACE_DOWN", "FOOD / HUNGRY", GestureCategory.MEDICAL,
                       "I am hungry. I need food.", priority=2, finger_pattern="-IM--"),
            20: Gesture(20, "THREE_FINGERS", "INJECTION", GestureCategory.MEDICAL,
                       "When is my injection scheduled?", priority=2, finger_pattern="-IMR-"),
            21: Gesture(21, "FOUR_FINGERS", "BLOOD TEST", GestureCategory.MEDICAL,
                       "I have a blood test.", priority=2, finger_pattern="-IMRP"),
            22: Gesture(22, "BOTH_HANDS", "BLANKET", GestureCategory.MEDICAL,
                       "I need a blanket please.", priority=2, finger_pattern="TIMRP"),
            23: Gesture(23, "ROCK_SIGN", "WHEELCHAIR", GestureCategory.MEDICAL,
                       "I need a wheelchair.", priority=3, finger_pattern="TI--P"),
            24: Gesture(24, "FLAT_HAND", "BED", GestureCategory.MEDICAL,
                       "I want to lie down on the bed.", priority=2, finger_pattern="TIMRP"),
            25: Gesture(25, "CUP_HAND", "NAUSEA", GestureCategory.MEDICAL,
                       "I am feeling nauseous. I might vomit.", priority=5, finger_pattern="TIMRP"),
            26: Gesture(26, "PHONE_SIGN", "CALL FAMILY", GestureCategory.MEDICAL,
                       "Please call my family.", priority=3, finger_pattern="T---P"),
            27: Gesture(27, "WRITING", "PAPER AND PEN", GestureCategory.MEDICAL,
                       "I need paper and pen to write.", priority=2, finger_pattern="-I---"),
            
            # ==================== FEELINGS (28-35) ====================
            28: Gesture(28, "COLD_SIGN", "COLD", GestureCategory.FEELINGS,
                       "I am feeling cold. I need a blanket.", priority=2, finger_pattern="-----"),
            29: Gesture(29, "VULCAN", "HOT / FEVER", GestureCategory.FEELINGS,
                       "I am feeling hot. I might have a fever.", priority=4, finger_pattern="-IM-P"),
            30: Gesture(30, "POINT_RIGHT", "TIRED / SLEEP", GestureCategory.FEELINGS,
                       "I am tired. I want to sleep.", priority=1, finger_pattern="-I---"),
            31: Gesture(31, "FIST_SHAKE", "SCARED / ANXIOUS", GestureCategory.FEELINGS,
                       "I am scared. I am feeling anxious.", priority=3, finger_pattern="-----"),
            32: Gesture(32, "HEART_SIGN", "UNCOMFORTABLE", GestureCategory.FEELINGS,
                       "I am feeling uncomfortable.", priority=3, finger_pattern="TI---"),
            33: Gesture(33, "CALM_SIGN", "FEELING BETTER", GestureCategory.FEELINGS,
                       "I am feeling better now.", priority=1, finger_pattern="TIMRP"),
            34: Gesture(34, "DIZZY_SIGN", "DIZZY", GestureCategory.FEELINGS,
                       "I am feeling dizzy.", priority=4, finger_pattern="-I---"),
            35: Gesture(35, "BREATH_SIGN", "DIFFICULTY BREATHING", GestureCategory.FEELINGS,
                       "I have difficulty breathing.", priority=8, finger_pattern="TIMRP"),
            
            # ==================== PEOPLE (36-37) ====================
            36: Gesture(36, "POINT_LEFT", "FAMILY", GestureCategory.PEOPLE,
                       "I want to see my family.", priority=2, finger_pattern="-I---"),
            37: Gesture(37, "TWO_POINT", "VISITOR", GestureCategory.PEOPLE,
                       "Is there a visitor for me?", priority=1, finger_pattern="-IM--"),
            
            # ==================== BODY PARTS (38-42) ====================
            38: Gesture(38, "HEAD_POINT", "HEAD / HEADACHE", GestureCategory.BODY,
                       "My head hurts. I have a headache.", priority=4, finger_pattern="-I---"),
            39: Gesture(39, "CHEST_POINT", "CHEST / HEART", GestureCategory.BODY,
                       "My chest hurts. Heart problem.", priority=7, finger_pattern="-I---"),
            40: Gesture(40, "STOMACH_SIGN", "STOMACH", GestureCategory.BODY,
                       "My stomach hurts.", priority=4, finger_pattern="TIMRP"),
            41: Gesture(41, "BACK_SIGN", "BACK PAIN", GestureCategory.BODY,
                       "My back is hurting.", priority=4, finger_pattern="T----"),
            42: Gesture(42, "LEG_SIGN", "LEG / FOOT", GestureCategory.BODY,
                       "My leg or foot hurts.", priority=3, finger_pattern="-IM--"),
            
            # ==================== TIME (43-44) ====================
            43: Gesture(43, "CLOCK_SIGN", "WHAT TIME", GestureCategory.TIME,
                       "What time is it?", priority=1, finger_pattern="-I---"),
            44: Gesture(44, "WAIT_SIGN", "HOW LONG", GestureCategory.TIME,
                       "How long do I have to wait?", priority=2, finger_pattern="TIMRP"),
        }
        return gestures
    
    def _get_finger_states(self, landmarks: List) -> List[bool]:
        """
        Determine which fingers are extended.
        
        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky]
        """
        fingers = []
        
        # Thumb - special case
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        if landmarks[0]['x'] < 0.5:  # Right hand (mirrored view)
            fingers.append(thumb_tip['x'] < thumb_ip['x'])
        else:  # Left hand
            fingers.append(thumb_tip['x'] > thumb_ip['x'])
        
        # Other fingers - tip above PIP = extended
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            fingers.append(landmarks[tip]['y'] < landmarks[pip]['y'])
        
        return fingers
    
    def _is_thumb_index_touching(self, landmarks: List) -> bool:
        """Check if thumb and index finger tips are close (OK sign)."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        distance = math.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 + 
            (thumb_tip['y'] - index_tip['y'])**2
        )
        return distance < 0.06
    
    def _get_thumb_direction(self, landmarks: List) -> str:
        """Determine if thumb is pointing up, down, or sideways."""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        dy = thumb_tip['y'] - thumb_mcp['y']
        dx = abs(thumb_tip['x'] - thumb_mcp['x'])
        
        if dy < -0.08 and abs(dy) > dx:
            return "up"
        elif dy > 0.08 and abs(dy) > dx:
            return "down"
        else:
            return "side"
    
    def _is_i_love_you_sign(self, fingers: List[bool]) -> bool:
        """Check for I Love You sign (thumb, index, pinky extended)."""
        return fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]
    
    def _is_call_sign(self, fingers: List[bool]) -> bool:
        """Check for call/phone sign (thumb and pinky extended)."""
        return fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and fingers[4]
    
    def _is_rock_sign(self, fingers: List[bool]) -> bool:
        """Check for rock sign (index and pinky extended)."""
        return not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]
    
    def _get_hand_height(self, landmarks: List) -> str:
        """Determine if hand is high, middle, or low in frame."""
        wrist_y = landmarks[0]['y']
        if wrist_y < 0.35:
            return "high"
        elif wrist_y > 0.65:
            return "low"
        return "middle"
    
    def _is_fingers_spread(self, landmarks: List) -> bool:
        """Check if fingers are spread apart."""
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        spread = abs(index_tip['x'] - pinky_tip['x'])
        return spread > 0.2
    
    def _get_index_direction(self, landmarks: List) -> str:
        """Get the direction index finger is pointing."""
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        
        dx = index_tip['x'] - index_mcp['x']
        dy = index_tip['y'] - index_mcp['y']
        
        if abs(dy) > abs(dx):
            return "up" if dy < 0 else "down"
        else:
            return "left" if dx < 0 else "right"

    def classify(self, landmarks: List, prev_landmarks: Optional[List] = None) -> Optional[Gesture]:
        """
        Classify the current hand gesture.
        """
        if landmarks is None or len(landmarks) != 21:
            return None
        
        fingers = self._get_finger_states(landmarks)
        finger_count = sum(fingers)
        thumb_dir = self._get_thumb_direction(landmarks)
        hand_height = self._get_hand_height(landmarks)
        index_dir = self._get_index_direction(landmarks)
        
        # === GESTURE CLASSIFICATION LOGIC ===
        
        # FIST - No fingers extended (EMERGENCY)
        if finger_count == 0:
            if hand_height == "high":
                return self.gestures[1]  # URGENT
            return self.gestures[0]  # HELP/EMERGENCY
        
        # ONE FINGER (Index only)
        if fingers == [False, True, False, False, False]:
            if index_dir == "up":
                if hand_height == "high":
                    return self.gestures[38]  # HEAD/HEADACHE
                return self.gestures[15]  # WATER
            elif index_dir == "down":
                return self.gestures[42]  # LEG/FOOT
            elif index_dir == "left":
                return self.gestures[36]  # FAMILY
            elif index_dir == "right":
                return self.gestures[30]  # TIRED/SLEEP
            return self.gestures[3]  # YES (default for index)
        
        # TWO FINGERS (Index + Middle)
        if fingers == [False, True, True, False, False]:
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            if index_tip['y'] > index_mcp['y']:
                return self.gestures[19]  # FOOD/HUNGRY (peace down)
            return self.gestures[4]  # NO
        
        # THREE FINGERS
        if fingers == [False, True, True, True, False]:
            return self.gestures[20]  # INJECTION
        
        # FOUR FINGERS
        if fingers == [False, True, True, True, True]:
            return self.gestures[21]  # BLOOD TEST
        
        # ALL 5 FINGERS
        if finger_count == 5:
            if self._is_fingers_spread(landmarks):
                return self.gestures[16]  # MEDICINE (spread)
            if hand_height == "low":
                return self.gestures[24]  # BED
            return self.gestures[5]  # STOP/WAIT
        
        # THUMBS UP
        if fingers == [True, False, False, False, False]:
            if thumb_dir == "up":
                return self.gestures[6]  # GOOD/OK
            elif thumb_dir == "down":
                return self.gestures[7]  # BAD/NOT OK
            return self.gestures[41]  # BACK PAIN (thumb side)
        
        # CALL SIGN (Thumb + Pinky)
        if self._is_call_sign(fingers):
            return self.gestures[13]  # CALL DOCTOR
        
        # I LOVE YOU SIGN (Thumb + Index + Pinky)
        if self._is_i_love_you_sign(fingers):
            return self.gestures[9]  # THANK YOU
        
        # ROCK SIGN (Index + Pinky, no thumb)
        if self._is_rock_sign(fingers):
            return self.gestures[23]  # WHEELCHAIR
        
        # OK SIGN - Thumb and index touching
        if self._is_thumb_index_touching(landmarks):
            if fingers[2]:
                return self.gestures[18]  # BATHROOM
            return self.gestures[11]  # LITTLE/SMALL (pinch)
        
        # THUMB + INDEX only
        if fingers == [True, True, False, False, False]:
            return self.gestures[32]  # UNCOMFORTABLE
        
        # INDEX + MIDDLE + PINKY (Vulcan-like)
        if fingers == [False, True, True, False, True]:
            return self.gestures[29]  # HOT/FEVER
        
        # Default fallback based on finger count
        if finger_count == 1:
            return self.gestures[3]  # YES
        elif finger_count == 2:
            return self.gestures[4]  # NO
        
        return None
    
    def classify_with_smoothing(self, landmarks: List, 
                                prev_landmarks: Optional[List] = None) -> Optional[Gesture]:
        """
        Classify gesture with temporal smoothing to reduce flickering.
        """
        current_gesture = self.classify(landmarks, prev_landmarks)
        
        if current_gesture is None:
            self.gesture_history = []
            self.gesture_stable_count = 0
            self.last_gesture = None
            return None
        
        # Add to history
        self.gesture_history.append(current_gesture.id)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # Find most common gesture in history
        if len(self.gesture_history) >= 2:
            from collections import Counter
            most_common = Counter(self.gesture_history).most_common(1)[0]
            gesture_id, count = most_common
            
            # Require majority agreement (50%)
            if count >= len(self.gesture_history) * 0.5:
                if self.last_gesture == gesture_id:
                    self.gesture_stable_count += 1
                else:
                    self.last_gesture = gesture_id
                    self.gesture_stable_count = 1
                
                if self.gesture_stable_count >= self.stability_threshold:
                    return self.gestures[gesture_id]
        
        return None
    
    def get_all_gestures(self) -> Dict[int, Gesture]:
        """Get all defined gestures."""
        return self.gestures
    
    def get_gestures_by_category(self, category: GestureCategory) -> List[Gesture]:
        """Get gestures filtered by category."""
        return [g for g in self.gestures.values() if g.category == category]
    
    def get_emergency_gestures(self) -> List[Gesture]:
        """Get all emergency/high-priority gestures."""
        return [g for g in self.gestures.values() if g.priority >= 5]
    
    def get_gesture_count(self) -> int:
        """Get total number of gestures."""
        return len(self.gestures)


def print_all_gestures():
    """Print all gestures in a formatted table."""
    classifier = GestureClassifier()
    
    print("=" * 80)
    print("🏥 HOSPITAL SIGN LANGUAGE GESTURES - COMPLETE LIST")
    print(f"   Total Gestures: {classifier.get_gesture_count()}")
    print("=" * 80)
    
    for category in GestureCategory:
        gestures = classifier.get_gestures_by_category(category)
        if gestures:
            print(f"\n{category.value} ({len(gestures)} gestures)")
            print("-" * 70)
            for g in gestures:
                priority_indicator = "⚡" if g.priority >= 5 else "  "
                print(f"  {priority_indicator} [{g.id:2d}] {g.meaning:25s} → \"{g.voice_text[:40]}...\"" 
                      if len(g.voice_text) > 40 else 
                      f"  {priority_indicator} [{g.id:2d}] {g.meaning:25s} → \"{g.voice_text}\"")
    
    print("\n" + "=" * 80)
    print("⚡ = High Priority Gesture")
    print("=" * 80)


# Run this to see all gestures
if __name__ == "__main__":
    print_all_gestures()
