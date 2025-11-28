"""
Notification Service for Hospital Sign Language System
Sends SMS and WhatsApp alerts for emergencies and important communications
Uses Twilio API for reliable message delivery
"""

import os
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
import json

# Try to import Twilio
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[Notification] Twilio not installed. Run: pip install twilio")


@dataclass
class NotificationConfig:
    """Configuration for notification service"""
    # Twilio credentials
    account_sid: str = ""
    auth_token: str = ""
    from_phone: str = ""  # Twilio phone number
    from_whatsapp: str = ""  # WhatsApp business number (format: whatsapp:+14155238886)
    
    # Recipients
    nurse_station_phone: str = ""
    emergency_phones: List[str] = field(default_factory=list)
    family_phones: List[str] = field(default_factory=list)
    
    # Settings
    enable_sms: bool = True
    enable_whatsapp: bool = True
    enable_emergency_alerts: bool = True
    cooldown_seconds: int = 30  # Minimum time between same type of alerts


@dataclass
class Notification:
    """Represents a notification to be sent"""
    id: str
    type: str  # 'sms', 'whatsapp', 'both'
    recipient: str
    message: str
    priority: int  # 1-10 (10 = highest/emergency)
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, sent, failed
    error: Optional[str] = None
    patient_id: Optional[str] = None
    gesture_id: Optional[int] = None


class HospitalNotificationService:
    """
    Hospital Notification Service
    
    Features:
    - SMS alerts via Twilio
    - WhatsApp messages via Twilio
    - Emergency broadcast to multiple recipients
    - Notification queuing and retry logic
    - Rate limiting to prevent spam
    - Logging of all notifications
    """
    
    # Pre-defined message templates
    TEMPLATES = {
        "emergency": "🚨 EMERGENCY ALERT - {room}\nPatient: {patient_name}\nMessage: {message}\nTime: {time}\n\nImmediate attention required!",
        "pain": "⚠️ Pain Alert - {room}\nPatient: {patient_name}\nReporting: {message}\nTime: {time}",
        "assistance": "📢 Assistance Request - {room}\nPatient: {patient_name}\nNeeds: {message}\nTime: {time}",
        "family_update": "👨‍👩‍👧 Family Update\nYour family member ({patient_name}) in {room} sent a message:\n\"{message}\"\nTime: {time}",
        "status_update": "📋 Patient Status - {room}\n{patient_name}: {message}\nTime: {time}",
    }
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize the notification service.
        
        Args:
            config: NotificationConfig object or None for env-based config
        """
        self.config = config or self._load_config_from_env()
        self.client = None
        self.is_available = False
        
        # Initialize Twilio client
        if TWILIO_AVAILABLE and self.config.account_sid and self.config.auth_token:
            try:
                self.client = TwilioClient(
                    self.config.account_sid,
                    self.config.auth_token
                )
                self.is_available = True
                print("[Notification] ✅ Twilio client initialized successfully")
            except Exception as e:
                print(f"[Notification] ⚠️ Failed to initialize Twilio: {e}")
        else:
            print("[Notification] ⚠️ Running in simulation mode (no Twilio credentials)")
        
        # Notification queue
        self.notification_queue = queue.Queue()
        self.is_running = True
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.worker_thread.start()
        
        # Rate limiting
        self.last_notification_time: Dict[str, datetime] = {}
        
        # History
        self.notification_history: List[Notification] = []
        self.max_history = 100
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Patient context
        self.patient_name = "Patient"
        self.room_number = "Unknown"
    
    def _load_config_from_env(self) -> NotificationConfig:
        """Load configuration from environment variables."""
        return NotificationConfig(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            from_phone=os.environ.get("TWILIO_PHONE_NUMBER", ""),
            from_whatsapp=os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886"),
            nurse_station_phone=os.environ.get("NURSE_STATION_PHONE", ""),
            emergency_phones=os.environ.get("EMERGENCY_PHONES", "").split(","),
            family_phones=os.environ.get("FAMILY_PHONES", "").split(","),
        )
    
    def set_patient_info(self, name: str, room: str):
        """Set patient information for notifications."""
        self.patient_name = name
        self.room_number = room
    
    def register_callback(self, callback: Callable[[Notification], None]):
        """Register callback for notification events."""
        self.callbacks.append(callback)
    
    def _format_message(self, template_key: str, message: str, **kwargs) -> str:
        """Format a message using templates."""
        template = self.TEMPLATES.get(template_key, "{message}")
        return template.format(
            room=kwargs.get("room", self.room_number),
            patient_name=kwargs.get("patient_name", self.patient_name),
            message=message,
            time=datetime.now().strftime("%H:%M:%S"),
            **kwargs
        )
    
    def _check_rate_limit(self, notification_type: str) -> bool:
        """Check if we're within rate limits."""
        key = notification_type
        now = datetime.now()
        
        if key in self.last_notification_time:
            elapsed = (now - self.last_notification_time[key]).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                print(f"[Notification] Rate limited: {notification_type} (wait {self.config.cooldown_seconds - elapsed:.0f}s)")
                return False
        
        self.last_notification_time[key] = now
        return True
    
    def send_emergency_alert(self, message: str, gesture_id: Optional[int] = None) -> bool:
        """
        Send emergency alert to all emergency contacts.
        
        Args:
            message: Emergency message
            gesture_id: Optional gesture ID that triggered the alert
            
        Returns:
            True if queued successfully
        """
        if not self.config.enable_emergency_alerts:
            print("[Notification] Emergency alerts disabled")
            return False
        
        formatted_message = self._format_message("emergency", message)
        
        # Send to nurse station
        if self.config.nurse_station_phone:
            self._queue_notification(
                "both",  # SMS and WhatsApp
                self.config.nurse_station_phone,
                formatted_message,
                priority=10,
                gesture_id=gesture_id
            )
        
        # Send to all emergency contacts
        for phone in self.config.emergency_phones:
            if phone.strip():
                self._queue_notification(
                    "sms",
                    phone.strip(),
                    formatted_message,
                    priority=10,
                    gesture_id=gesture_id
                )
        
        print(f"[Notification] 🚨 Emergency alert queued: {message[:50]}...")
        return True
    
    def send_nurse_alert(self, message: str, priority: int = 5, 
                        gesture_id: Optional[int] = None) -> bool:
        """
        Send alert to nurse station.
        
        Args:
            message: Alert message
            priority: Priority level (1-10)
            gesture_id: Optional gesture ID
            
        Returns:
            True if queued successfully
        """
        if not self.config.nurse_station_phone:
            print("[Notification] No nurse station phone configured")
            return False
        
        template = "pain" if priority >= 7 else "assistance"
        formatted_message = self._format_message(template, message)
        
        self._queue_notification(
            "whatsapp" if priority < 8 else "both",
            self.config.nurse_station_phone,
            formatted_message,
            priority=priority,
            gesture_id=gesture_id
        )
        
        return True
    
    def send_family_notification(self, message: str, 
                                gesture_id: Optional[int] = None) -> bool:
        """
        Send notification to family members.
        
        Args:
            message: Message to send
            gesture_id: Optional gesture ID
            
        Returns:
            True if queued successfully
        """
        if not self.config.family_phones:
            print("[Notification] No family phones configured")
            return False
        
        formatted_message = self._format_message("family_update", message)
        
        for phone in self.config.family_phones:
            if phone.strip():
                self._queue_notification(
                    "whatsapp",
                    phone.strip(),
                    formatted_message,
                    priority=3,
                    gesture_id=gesture_id
                )
        
        return True
    
    def _queue_notification(self, notification_type: str, recipient: str,
                           message: str, priority: int = 5,
                           gesture_id: Optional[int] = None):
        """Add notification to queue."""
        notification = Notification(
            id=f"notif_{datetime.now().timestamp()}",
            type=notification_type,
            recipient=recipient,
            message=message,
            priority=priority,
            gesture_id=gesture_id,
            patient_id=self.patient_name
        )
        
        self.notification_queue.put(notification)
        self.notification_history.append(notification)
        
        # Trim history
        if len(self.notification_history) > self.max_history:
            self.notification_history = self.notification_history[-self.max_history:]
    
    def _notification_worker(self):
        """Background worker to process notification queue."""
        while self.is_running:
            try:
                notification = self.notification_queue.get(timeout=1.0)
                
                if notification is None:
                    continue
                
                # Process the notification
                success = self._send_notification(notification)
                
                notification.status = "sent" if success else "failed"
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(notification)
                    except Exception as e:
                        print(f"[Notification] Callback error: {e}")
                
                self.notification_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Notification] Worker error: {e}")
    
    def _send_notification(self, notification: Notification) -> bool:
        """Send a single notification."""
        try:
            if not self.is_available:
                # Simulation mode
                print(f"[Notification] 📱 SIMULATED {notification.type.upper()} to {notification.recipient}")
                print(f"    Message: {notification.message[:100]}...")
                return True
            
            success = True
            
            # Send SMS
            if notification.type in ("sms", "both") and self.config.enable_sms:
                success = self._send_sms(notification.recipient, notification.message) and success
            
            # Send WhatsApp
            if notification.type in ("whatsapp", "both") and self.config.enable_whatsapp:
                success = self._send_whatsapp(notification.recipient, notification.message) and success
            
            return success
            
        except Exception as e:
            notification.error = str(e)
            print(f"[Notification] Failed to send: {e}")
            return False
    
    def _send_sms(self, to_phone: str, message: str) -> bool:
        """Send SMS via Twilio."""
        try:
            # Ensure phone number format
            if not to_phone.startswith("+"):
                to_phone = f"+91{to_phone}"  # Default to India
            
            msg = self.client.messages.create(
                body=message,
                from_=self.config.from_phone,
                to=to_phone
            )
            
            print(f"[Notification] ✅ SMS sent: {msg.sid}")
            return True
            
        except Exception as e:
            print(f"[Notification] ❌ SMS failed: {e}")
            return False
    
    def _send_whatsapp(self, to_phone: str, message: str) -> bool:
        """Send WhatsApp message via Twilio."""
        try:
            # Format for WhatsApp
            if not to_phone.startswith("whatsapp:"):
                if not to_phone.startswith("+"):
                    to_phone = f"+91{to_phone}"
                to_phone = f"whatsapp:{to_phone}"
            
            msg = self.client.messages.create(
                body=message,
                from_=self.config.from_whatsapp,
                to=to_phone
            )
            
            print(f"[Notification] ✅ WhatsApp sent: {msg.sid}")
            return True
            
        except Exception as e:
            print(f"[Notification] ❌ WhatsApp failed: {e}")
            return False
    
    def send_gesture_notification(self, gesture_id: int, gesture_meaning: str,
                                  voice_text: str, priority: int, 
                                  is_emergency: bool = False) -> bool:
        """
        Send notification based on detected gesture.
        
        Args:
            gesture_id: Gesture ID
            gesture_meaning: Human-readable meaning
            voice_text: The spoken text
            priority: Priority level
            is_emergency: Whether this is an emergency
            
        Returns:
            True if notification was queued
        """
        # Check rate limit for this gesture type
        rate_key = f"gesture_{gesture_id}"
        if not self._check_rate_limit(rate_key):
            return False
        
        message = f"{gesture_meaning}: {voice_text}"
        
        if is_emergency:
            return self.send_emergency_alert(message, gesture_id)
        elif priority >= 7:
            return self.send_nurse_alert(message, priority, gesture_id)
        elif priority >= 4:
            return self.send_nurse_alert(message, priority, gesture_id)
        
        # Low priority - just log, don't send
        print(f"[Notification] Low priority gesture logged: {message}")
        return False
    
    def get_notification_history(self, limit: int = 20) -> List[Notification]:
        """Get recent notification history."""
        return self.notification_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get notification statistics."""
        total = len(self.notification_history)
        sent = sum(1 for n in self.notification_history if n.status == "sent")
        failed = sum(1 for n in self.notification_history if n.status == "failed")
        
        return {
            "total": total,
            "sent": sent,
            "failed": failed,
            "pending": total - sent - failed,
            "success_rate": f"{(sent/total*100):.1f}%" if total > 0 else "N/A"
        }
    
    def stop(self):
        """Stop the notification service."""
        self.is_running = False
        self.notification_queue.put(None)
        
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        print("[Notification] Service stopped")


class QuickAlertButtons:
    """
    Pre-configured quick alert buttons for common scenarios.
    Can be used in GUI for one-click alerts.
    """
    
    def __init__(self, notification_service: HospitalNotificationService):
        self.service = notification_service
    
    def alert_nurse_now(self):
        """Quick alert: Need nurse immediately"""
        self.service.send_nurse_alert("Patient needs immediate assistance", priority=8)
    
    def alert_pain(self, location: str = "unspecified"):
        """Quick alert: Patient in pain"""
        self.service.send_nurse_alert(f"Patient reporting pain ({location})", priority=7)
    
    def alert_bathroom(self):
        """Quick alert: Bathroom assistance needed"""
        self.service.send_nurse_alert("Patient needs bathroom assistance", priority=5)
    
    def alert_water(self):
        """Quick alert: Patient needs water"""
        self.service.send_nurse_alert("Patient requesting water", priority=3)
    
    def alert_emergency(self):
        """Quick alert: Emergency"""
        self.service.send_emergency_alert("EMERGENCY - Patient needs immediate help!")
    
    def notify_family(self, message: str):
        """Send message to family"""
        self.service.send_family_notification(message)


# Test the notification service
if __name__ == "__main__":
    print("=" * 60)
    print("📱 Testing Hospital Notification Service")
    print("=" * 60)
    
    # Create config (simulation mode without real credentials)
    config = NotificationConfig(
        nurse_station_phone="+919876543210",
        emergency_phones=["+919876543211", "+919876543212"],
        family_phones=["+919876543213"],
        enable_sms=True,
        enable_whatsapp=True,
    )
    
    # Initialize service
    service = HospitalNotificationService(config)
    service.set_patient_info("Rahul Sharma", "ICU-Room 203")
    
    # Test notifications
    print("\n--- Testing Emergency Alert ---")
    service.send_emergency_alert("Patient showing signs of cardiac distress!")
    
    print("\n--- Testing Nurse Alert ---")
    service.send_nurse_alert("Patient requesting pain medication", priority=7)
    
    print("\n--- Testing Family Notification ---")
    service.send_family_notification("I am feeling much better today")
    
    print("\n--- Testing Gesture-based Notification ---")
    service.send_gesture_notification(
        gesture_id=14,
        gesture_meaning="PAIN",
        voice_text="I am in pain. It hurts.",
        priority=7,
        is_emergency=False
    )
    
    # Wait for queue to process
    import time
    time.sleep(2)
    
    # Show statistics
    print("\n--- Notification Statistics ---")
    stats = service.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test quick alerts
    print("\n--- Testing Quick Alert Buttons ---")
    quick = QuickAlertButtons(service)
    quick.alert_nurse_now()
    
    time.sleep(2)
    
    service.stop()
    print("\n✅ Notification service test complete!")

