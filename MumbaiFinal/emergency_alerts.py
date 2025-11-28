"""
Emergency Alert System for Hospital Sign Language
Handles critical gestures with visual and audio alerts
"""

import threading
import time
from typing import Callable, Optional, List
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class Alert:
    """Represents an emergency alert."""
    timestamp: datetime
    gesture_name: str
    message: str
    priority: int
    acknowledged: bool = False


class EmergencyAlertSystem:
    """
    Manages emergency alerts for critical patient gestures.
    Provides visual indicators, sound alerts, and logging.
    """
    
    def __init__(self, log_file: str = "emergency_log.txt"):
        """
        Initialize the emergency alert system.
        
        Args:
            log_file: Path to the emergency log file
        """
        self.log_file = log_file
        self.alerts: List[Alert] = []
        self.is_alerting = False
        self.alert_callbacks: List[Callable] = []
        
        # Alert configuration
        self.alert_duration = 5.0  # seconds
        self.flash_interval = 0.3  # seconds for visual flash
        
        # Emergency gesture IDs (from gesture_classifier)
        self.emergency_gesture_ids = {0, 1, 2, 35}  # HELP, URGENT, SEVERE PAIN, DIFFICULTY BREATHING
        self.high_priority_ids = {7, 13, 14, 25, 29, 31, 39}  # BAD, CALL DOCTOR, PAIN, NAUSEA, FEVER, SCARED, CHEST
        
        # Alert state
        self.current_alert: Optional[Alert] = None
        self.alert_thread: Optional[threading.Thread] = None
        self._stop_alert = threading.Event()
        
    def register_callback(self, callback: Callable[[Alert], None]):
        """
        Register a callback function to be called when an alert is triggered.
        
        Args:
            callback: Function that takes an Alert object
        """
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, gesture_id: int, gesture_name: str, message: str, priority: int):
        """
        Trigger an emergency alert.
        
        Args:
            gesture_id: ID of the triggering gesture
            gesture_name: Name of the gesture
            message: Alert message
            priority: Priority level (higher = more urgent)
        """
        alert = Alert(
            timestamp=datetime.now(),
            gesture_name=gesture_name,
            message=message,
            priority=priority
        )
        
        self.alerts.append(alert)
        self.current_alert = alert
        self.is_alerting = True
        
        # Log the alert
        self._log_alert(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        # Start alert thread
        self._stop_alert.clear()
        self.alert_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.alert_thread.start()
    
    def _alert_worker(self):
        """Background thread for managing alert duration."""
        start_time = time.time()
        
        while not self._stop_alert.is_set():
            elapsed = time.time() - start_time
            
            if elapsed >= self.alert_duration:
                self.is_alerting = False
                break
            
            time.sleep(0.1)
    
    def acknowledge_alert(self):
        """Acknowledge and clear the current alert."""
        if self.current_alert:
            self.current_alert.acknowledged = True
            self._stop_alert.set()
            self.is_alerting = False
            self.current_alert = None
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_entry = (
                    f"[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"PRIORITY:{alert.priority} | {alert.gesture_name} | {alert.message}\n"
                )
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to log alert: {e}")
    
    def is_emergency_gesture(self, gesture_id: int) -> bool:
        """Check if gesture is an emergency gesture."""
        return gesture_id in self.emergency_gesture_ids
    
    def is_high_priority(self, gesture_id: int) -> bool:
        """Check if gesture is high priority."""
        return gesture_id in self.high_priority_ids or gesture_id in self.emergency_gesture_ids
    
    def get_alert_color(self) -> tuple:
        """
        Get the current alert color for UI (flashing effect).
        
        Returns:
            RGB tuple for alert color
        """
        if not self.is_alerting:
            return (0, 0, 0)  # No alert - black/transparent
        
        # Flash between red and orange
        flash_state = int(time.time() / self.flash_interval) % 2
        
        if self.current_alert and self.current_alert.priority >= 7:
            # Critical - flash red/yellow
            return (255, 0, 0) if flash_state else (255, 255, 0)
        else:
            # High priority - flash orange/yellow
            return (255, 165, 0) if flash_state else (255, 255, 0)
    
    def get_alert_history(self, limit: int = 10) -> List[Alert]:
        """Get recent alert history."""
        return self.alerts[-limit:]
    
    def clear_history(self):
        """Clear alert history."""
        self.alerts = []
    
    def get_statistics(self) -> dict:
        """Get alert statistics."""
        if not self.alerts:
            return {"total": 0, "acknowledged": 0, "pending": 0}
        
        acknowledged = sum(1 for a in self.alerts if a.acknowledged)
        return {
            "total": len(self.alerts),
            "acknowledged": acknowledged,
            "pending": len(self.alerts) - acknowledged,
            "last_alert": self.alerts[-1].timestamp.strftime('%H:%M:%S') if self.alerts else None
        }


class VisualAlertOverlay:
    """
    Creates visual overlay effects for alerts.
    Can be used with OpenCV or GUI frameworks.
    """
    
    def __init__(self):
        self.alert_system: Optional[EmergencyAlertSystem] = None
        self.overlay_alpha = 0.3
    
    def attach(self, alert_system: EmergencyAlertSystem):
        """Attach to an alert system."""
        self.alert_system = alert_system
    
    def get_overlay_color_bgr(self) -> tuple:
        """Get BGR color for OpenCV overlay."""
        if not self.alert_system or not self.alert_system.is_alerting:
            return None
        
        rgb = self.alert_system.get_alert_color()
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    
    def apply_to_frame(self, frame, border_width: int = 10):
        """
        Apply alert overlay to an OpenCV frame.
        
        Args:
            frame: OpenCV frame (BGR)
            border_width: Width of the alert border
            
        Returns:
            Modified frame with alert overlay
        """
        import cv2
        import numpy as np
        
        color = self.get_overlay_color_bgr()
        if color is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw alert border
        cv2.rectangle(frame, (0, 0), (w, border_width), color, -1)  # Top
        cv2.rectangle(frame, (0, h - border_width), (w, h), color, -1)  # Bottom
        cv2.rectangle(frame, (0, 0), (border_width, h), color, -1)  # Left
        cv2.rectangle(frame, (w - border_width, 0), (w, h), color, -1)  # Right
        
        # Add alert text if there's a current alert
        if self.alert_system.current_alert:
            alert = self.alert_system.current_alert
            text = f"⚠ {alert.gesture_name}: {alert.message}"
            
            # Draw text background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, h - 80), (text_w + 20, h - 45), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (15, h - 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


# Test the alert system
if __name__ == "__main__":
    print("Testing Emergency Alert System...")
    
    def alert_callback(alert: Alert):
        print(f"\n🚨 ALERT RECEIVED!")
        print(f"   Time: {alert.timestamp}")
        print(f"   Gesture: {alert.gesture_name}")
        print(f"   Message: {alert.message}")
        print(f"   Priority: {alert.priority}")
    
    alert_system = EmergencyAlertSystem()
    alert_system.register_callback(alert_callback)
    
    # Simulate emergency gesture
    print("\nTriggering test emergency...")
    alert_system.trigger_alert(
        gesture_id=0,
        gesture_name="FIST",
        message="Help! Emergency! Patient needs immediate assistance!",
        priority=10
    )
    
    # Show flashing colors
    print("\nAlert colors (flashing):")
    for i in range(10):
        color = alert_system.get_alert_color()
        print(f"  Flash {i+1}: RGB{color}")
        time.sleep(0.5)
    
    # Acknowledge
    print("\nAcknowledging alert...")
    alert_system.acknowledge_alert()
    
    # Show stats
    print("\nAlert Statistics:")
    stats = alert_system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")

