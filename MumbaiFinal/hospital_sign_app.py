"""
Hospital Sign Language Communication System
Main GUI Application for Deaf Patient Communication

A beautiful, modern interface for real-time sign language recognition
to help deaf patients communicate in hospitals.

ENHANCED VERSION with:
- AI Agent (Groq) for intelligent responses
- CNN Model integration for ML-based recognition
- SMS/WhatsApp notifications via Twilio
- Multi-language support
"""

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from typing import Optional

# Import our modules
from gesture_detector import HandDetector
from gesture_classifier import GestureClassifier, Gesture, GestureCategory
from text_to_speech import HospitalTTS
from emergency_alerts import EmergencyAlertSystem, VisualAlertOverlay, Alert

# Import new enhanced modules
try:
    from ai_agent import HospitalAIAgent, PatientContext
    AI_AGENT_AVAILABLE = True
except ImportError:
    AI_AGENT_AVAILABLE = False
    print("[App] AI Agent module not available")

try:
    from notification_service import HospitalNotificationService, NotificationConfig
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    print("[App] Notification service not available")

try:
    from cnn_classifier import HybridGestureClassifier, GestureCNN
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("[App] CNN classifier not available")


# ============== THEME CONFIGURATION ==============
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Custom Colors - Enhanced color palette
COLORS = {
    'bg_dark': '#0a0a0f',
    'bg_card': '#14141f',
    'bg_secondary': '#1a1a2e',
    'accent_blue': '#4361ee',
    'accent_cyan': '#00d4ff',
    'accent_purple': '#7209b7',
    'accent_pink': '#f72585',
    'accent_green': '#00f5a0',
    'success': '#00f5a0',
    'warning': '#ffc107',
    'danger': '#ff3366',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0b0',
    'border': '#2a2a3e',
    'ai_response': '#1e3a5f',
    'notification': '#2d1f3d',
}


class AIResponsePanel(ctk.CTkFrame):
    """Panel showing AI agent responses and suggestions."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(15, 5))
        
        header = ctk.CTkLabel(
            header_frame, text="🤖 AI Assistant",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS['accent_cyan']
        )
        header.pack(side="left")
        
        # AI status indicator
        self.ai_status = ctk.CTkLabel(
            header_frame,
            text="● Online",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['success']
        )
        self.ai_status.pack(side="right")
        
        # AI Response display
        self.response_frame = ctk.CTkFrame(
            self, fg_color=COLORS['ai_response'], corner_radius=10
        )
        self.response_frame.pack(fill="x", padx=15, pady=10)
        
        self.response_label = ctk.CTkLabel(
            self.response_frame,
            text="AI assistant ready. Waiting for patient gestures...",
            font=ctk.CTkFont(size=13),
            text_color=COLORS['text_primary'],
            wraplength=300,
            justify="left"
        )
        self.response_label.pack(padx=15, pady=15, anchor="w")
        
        # Nurse suggestions section
        suggestions_header = ctk.CTkLabel(
            self, text="📋 Nurse Follow-up Questions:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_secondary']
        )
        suggestions_header.pack(padx=15, pady=(10, 5), anchor="w")
        
        self.suggestions_frame = ctk.CTkFrame(
            self, fg_color=COLORS['bg_secondary'], corner_radius=8
        )
        self.suggestions_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.suggestion_labels = []
        for i in range(3):
            lbl = ctk.CTkLabel(
                self.suggestions_frame,
                text="",
                font=ctk.CTkFont(size=11),
                text_color=COLORS['text_secondary'],
                wraplength=280,
                justify="left"
            )
            lbl.pack(padx=10, pady=3, anchor="w")
            self.suggestion_labels.append(lbl)
    
    def update_response(self, response: str, is_emergency: bool = False):
        """Update AI response display."""
        color = COLORS['danger'] if is_emergency else COLORS['text_primary']
        self.response_label.configure(text=response, text_color=color)
        
        if is_emergency:
            self.response_frame.configure(fg_color=COLORS['danger'])
        else:
            self.response_frame.configure(fg_color=COLORS['ai_response'])
    
    def update_suggestions(self, suggestions: list):
        """Update nurse suggestions."""
        for i, lbl in enumerate(self.suggestion_labels):
            if i < len(suggestions):
                lbl.configure(text=f"• {suggestions[i]}")
            else:
                lbl.configure(text="")
    
    def set_status(self, status: str, online: bool = True):
        """Update AI status indicator."""
        color = COLORS['success'] if online else COLORS['warning']
        self.ai_status.configure(text=f"● {status}", text_color=color)


class NotificationPanel(ctk.CTkFrame):
    """Panel showing notification status and controls."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Header
        header = ctk.CTkLabel(
            self, text="📱 Notifications",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS['text_primary']
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Status indicators
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=5)
        
        self.sms_status = ctk.CTkLabel(
            status_frame, text="📱 SMS: Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['success']
        )
        self.sms_status.pack(side="left", padx=(0, 15))
        
        self.whatsapp_status = ctk.CTkLabel(
            status_frame, text="💬 WhatsApp: Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['success']
        )
        self.whatsapp_status.pack(side="left")
        
        # Quick action buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=10)
        
        self.call_nurse_btn = ctk.CTkButton(
            btn_frame,
            text="🔔 Alert Nurse",
            width=100,
            height=32,
            fg_color=COLORS['accent_blue'],
            hover_color=COLORS['accent_purple'],
            font=ctk.CTkFont(size=11)
        )
        self.call_nurse_btn.pack(side="left", padx=(0, 5))
        
        self.emergency_btn = ctk.CTkButton(
            btn_frame,
            text="🚨 Emergency",
            width=100,
            height=32,
            fg_color=COLORS['danger'],
            hover_color="#cc2952",
            font=ctk.CTkFont(size=11)
        )
        self.emergency_btn.pack(side="left", padx=5)
        
        self.family_btn = ctk.CTkButton(
            btn_frame,
            text="👨‍👩‍👧 Family",
            width=80,
            height=32,
            fg_color=COLORS['bg_secondary'],
            hover_color=COLORS['accent_purple'],
            font=ctk.CTkFont(size=11)
        )
        self.family_btn.pack(side="left", padx=5)
        
        # Last notification display
        self.last_notif_label = ctk.CTkLabel(
            self,
            text="No notifications sent yet",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['text_secondary']
        )
        self.last_notif_label.pack(padx=15, pady=(5, 15), anchor="w")
    
    def update_last_notification(self, message: str):
        """Update last notification display."""
        time_str = datetime.now().strftime("%H:%M")
        self.last_notif_label.configure(text=f"[{time_str}] {message[:40]}...")
    
    def set_button_commands(self, nurse_cmd, emergency_cmd, family_cmd):
        """Set button command callbacks."""
        self.call_nurse_btn.configure(command=nurse_cmd)
        self.emergency_btn.configure(command=emergency_cmd)
        self.family_btn.configure(command=family_cmd)


class PatientInfoPanel(ctk.CTkFrame):
    """Panel for patient information and settings."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Header
        header = ctk.CTkLabel(
            self, text="👤 Patient Information",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS['text_primary']
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Patient name
        name_frame = ctk.CTkFrame(self, fg_color="transparent")
        name_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            name_frame, text="Name:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        ).pack(side="left")
        
        self.name_entry = ctk.CTkEntry(
            name_frame, width=150, height=28,
            placeholder_text="Patient Name",
            fg_color=COLORS['bg_secondary']
        )
        self.name_entry.pack(side="right")
        self.name_entry.insert(0, "Patient")
        
        # Room number
        room_frame = ctk.CTkFrame(self, fg_color="transparent")
        room_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            room_frame, text="Room:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        ).pack(side="left")
        
        self.room_entry = ctk.CTkEntry(
            room_frame, width=150, height=28,
            placeholder_text="Room Number",
            fg_color=COLORS['bg_secondary']
        )
        self.room_entry.pack(side="right")
        self.room_entry.insert(0, "ICU-101")
        
        # Emergency contact
        contact_frame = ctk.CTkFrame(self, fg_color="transparent")
        contact_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(
            contact_frame, text="Emergency:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary']
        ).pack(side="left")
        
        self.contact_entry = ctk.CTkEntry(
            contact_frame, width=150, height=28,
            placeholder_text="+91-XXXXXXXXXX",
            fg_color=COLORS['bg_secondary']
        )
        self.contact_entry.pack(side="right")
        
        # Save button
        self.save_btn = ctk.CTkButton(
            self,
            text="💾 Save Info",
            width=100,
            height=28,
            fg_color=COLORS['success'],
            hover_color="#00c580",
            font=ctk.CTkFont(size=11)
        )
        self.save_btn.pack(pady=(10, 15))
    
    def get_patient_info(self) -> dict:
        """Get current patient info."""
        return {
            "name": self.name_entry.get(),
            "room": self.room_entry.get(),
            "emergency_contact": self.contact_entry.get()
        }


class CommunicationHistoryPanel(ctk.CTkFrame):
    """Panel showing communication history."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Header
        header = ctk.CTkLabel(
            self, text="📋 Communication History",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS['text_primary']
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Scrollable history
        self.history_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=COLORS['accent_blue']
        )
        self.history_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.messages = []
    
    def add_message(self, gesture: Gesture, timestamp: datetime, ai_response: str = None):
        """Add a message to history."""
        # Create message card
        msg_frame = ctk.CTkFrame(
            self.history_frame,
            fg_color=COLORS['bg_secondary'],
            corner_radius=10
        )
        msg_frame.pack(fill="x", pady=5, padx=5)
        
        # Time
        time_label = ctk.CTkLabel(
            msg_frame,
            text=timestamp.strftime("%H:%M:%S"),
            font=ctk.CTkFont(size=10),
            text_color=COLORS['text_secondary']
        )
        time_label.pack(anchor="w", padx=10, pady=(8, 0))
        
        # Category badge color
        category_colors = {
            GestureCategory.EMERGENCY: COLORS['danger'],
            GestureCategory.MEDICAL: COLORS['accent_blue'],
            GestureCategory.FEELINGS: COLORS['accent_purple'],
            GestureCategory.BASIC: COLORS['success'],
            GestureCategory.PEOPLE: COLORS['accent_cyan'],
            GestureCategory.BODY: COLORS['warning'],
            GestureCategory.TIME: COLORS['accent_pink']
        }
        
        badge_color = category_colors.get(gesture.category, COLORS['accent_blue'])
        
        # Gesture name with category color
        name_label = ctk.CTkLabel(
            msg_frame,
            text=f"● {gesture.meaning}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=badge_color
        )
        name_label.pack(anchor="w", padx=10)
        
        # Voice text
        voice_label = ctk.CTkLabel(
            msg_frame,
            text=f'"{gesture.voice_text}"',
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_secondary'],
            wraplength=250
        )
        voice_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # AI response if available
        if ai_response:
            ai_label = ctk.CTkLabel(
                msg_frame,
                text=f"🤖 {ai_response[:80]}...",
                font=ctk.CTkFont(size=11),
                text_color=COLORS['accent_cyan'],
                wraplength=250
            )
            ai_label.pack(anchor="w", padx=10, pady=(0, 8))
        
        self.messages.append(msg_frame)
        
        # Keep only last 50 messages
        if len(self.messages) > 50:
            old_msg = self.messages.pop(0)
            old_msg.destroy()
    
    def clear_history(self):
        """Clear all history."""
        for msg in self.messages:
            msg.destroy()
        self.messages = []


class GestureGuidePanel(ctk.CTkFrame):
    """Panel showing gesture guide."""
    
    def __init__(self, parent, classifier: GestureClassifier, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        self.classifier = classifier
        
        # Header
        header = ctk.CTkLabel(
            self, text="📖 Gesture Guide",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS['text_primary']
        )
        header.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Category selector
        self.category_var = ctk.StringVar(value="All")
        categories = ["All", "Emergency", "Basic", "Medical", "Feelings", "People", "Body", "Time"]
        
        category_menu = ctk.CTkSegmentedButton(
            self,
            values=categories,
            variable=self.category_var,
            command=self._filter_gestures,
            font=ctk.CTkFont(size=11),
            fg_color=COLORS['bg_secondary'],
            selected_color=COLORS['accent_blue'],
            selected_hover_color=COLORS['accent_purple']
        )
        category_menu.pack(fill="x", padx=15, pady=(0, 10))
        
        # Scrollable gesture list
        self.gesture_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=COLORS['accent_blue']
        )
        self.gesture_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self._populate_gestures()
    
    def _populate_gestures(self, category_filter: str = "All"):
        """Populate gesture list."""
        # Clear existing
        for widget in self.gesture_frame.winfo_children():
            widget.destroy()
        
        # Category mapping
        category_map = {
            "Emergency": GestureCategory.EMERGENCY,
            "Basic": GestureCategory.BASIC,
            "Medical": GestureCategory.MEDICAL,
            "Feelings": GestureCategory.FEELINGS,
            "People": GestureCategory.PEOPLE,
            "Body": GestureCategory.BODY,
            "Time": GestureCategory.TIME
        }
        
        gestures = self.classifier.get_all_gestures().values()
        
        if category_filter != "All":
            target_cat = category_map.get(category_filter)
            gestures = [g for g in gestures if g.category == target_cat]
        
        for gesture in gestures:
            self._create_gesture_card(gesture)
    
    def _create_gesture_card(self, gesture: Gesture):
        """Create a card for a gesture."""
        card = ctk.CTkFrame(
            self.gesture_frame,
            fg_color=COLORS['bg_secondary'],
            corner_radius=8,
            height=50
        )
        card.pack(fill="x", pady=3, padx=5)
        card.pack_propagate(False)
        
        # ID badge
        id_label = ctk.CTkLabel(
            card,
            text=f"{gesture.id:02d}",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLORS['accent_cyan'],
            width=25
        )
        id_label.pack(side="left", padx=(10, 5), pady=5)
        
        # Meaning
        meaning_label = ctk.CTkLabel(
            card,
            text=gesture.meaning,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS['text_primary'],
            anchor="w"
        )
        meaning_label.pack(side="left", fill="x", expand=True, padx=5)
        
        # Category indicator
        cat_colors = {
            GestureCategory.EMERGENCY: ("🚨", COLORS['danger']),
            GestureCategory.MEDICAL: ("🏥", COLORS['accent_blue']),
            GestureCategory.FEELINGS: ("😊", COLORS['accent_purple']),
            GestureCategory.BASIC: ("✅", COLORS['success']),
            GestureCategory.PEOPLE: ("👥", COLORS['accent_cyan']),
            GestureCategory.BODY: ("🦴", COLORS['warning']),
            GestureCategory.TIME: ("⏰", COLORS['accent_pink'])
        }
        
        emoji, color = cat_colors.get(gesture.category, ("●", COLORS['text_secondary']))
        cat_label = ctk.CTkLabel(
            card,
            text=emoji,
            font=ctk.CTkFont(size=14)
        )
        cat_label.pack(side="right", padx=10)
    
    def _filter_gestures(self, category: str):
        """Filter gestures by category."""
        self._populate_gestures(category)


class VideoPanel(ctk.CTkFrame):
    """Main video display panel."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Header with status
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="🎥 Live Camera Feed",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=COLORS['text_primary']
        )
        self.title_label.pack(side="left")
        
        # ML mode indicator
        self.ml_indicator = ctk.CTkLabel(
            self.header_frame,
            text="🧠 ML",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['accent_purple']
        )
        self.ml_indicator.pack(side="left", padx=(10, 0))
        
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="● Initializing...",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['warning']
        )
        self.status_label.pack(side="right")
        
        # Video canvas
        self.video_label = ctk.CTkLabel(self, text="", fg_color=COLORS['bg_dark'])
        self.video_label.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.current_image = None
    
    def update_frame(self, frame):
        """Update the video frame."""
        if frame is None:
            return
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit panel (maintain aspect ratio)
        panel_width = self.video_label.winfo_width()
        panel_height = self.video_label.winfo_height()
        
        if panel_width > 1 and panel_height > 1:
            # Calculate scaling
            img_ratio = img.width / img.height
            panel_ratio = panel_width / panel_height
            
            if img_ratio > panel_ratio:
                new_width = panel_width
                new_height = int(panel_width / img_ratio)
            else:
                new_height = panel_height
                new_width = int(panel_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to CTkImage
        self.current_image = ctk.CTkImage(light_image=img, dark_image=img, 
                                          size=(img.width, img.height))
        self.video_label.configure(image=self.current_image)
    
    def set_status(self, status: str, color: str = None):
        """Update status indicator."""
        self.status_label.configure(text=f"● {status}", 
                                   text_color=color or COLORS['text_secondary'])
    
    def set_ml_status(self, active: bool):
        """Update ML indicator."""
        if active:
            self.ml_indicator.configure(text="🧠 ML Active", text_color=COLORS['success'])
        else:
            self.ml_indicator.configure(text="🧠 ML", text_color=COLORS['text_secondary'])


class DetectionPanel(ctk.CTkFrame):
    """Panel showing current detection result."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=COLORS['bg_card'], corner_radius=15, **kwargs)
        
        # Main gesture display
        self.gesture_label = ctk.CTkLabel(
            self,
            text="Show a gesture...",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=COLORS['accent_cyan']
        )
        self.gesture_label.pack(pady=(20, 5))
        
        # Voice text display
        self.voice_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=COLORS['text_secondary'],
            wraplength=400
        )
        self.voice_label.pack(pady=(0, 5))
        
        # Category badge
        self.category_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_secondary'],
            fg_color=COLORS['bg_secondary'],
            corner_radius=8,
            padx=12,
            pady=4
        )
        self.category_label.pack(pady=(0, 10))
        
        # Confidence bar
        self.confidence_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.confidence_frame.pack(fill="x", padx=40, pady=(0, 15))
        
        self.confidence_bar = ctk.CTkProgressBar(
            self.confidence_frame,
            height=6,
            corner_radius=3,
            fg_color=COLORS['bg_secondary'],
            progress_color=COLORS['accent_blue']
        )
        self.confidence_bar.pack(fill="x")
        self.confidence_bar.set(0)
    
    def update_detection(self, gesture: Optional[Gesture], confidence: float = 1.0):
        """Update the detection display."""
        if gesture is None:
            self.gesture_label.configure(text="Show a gesture...", 
                                        text_color=COLORS['text_secondary'])
            self.voice_label.configure(text="")
            self.category_label.configure(text="")
            self.confidence_bar.set(0)
            return
        
        # Color based on category
        cat_colors = {
            GestureCategory.EMERGENCY: COLORS['danger'],
            GestureCategory.MEDICAL: COLORS['accent_blue'],
            GestureCategory.FEELINGS: COLORS['accent_purple'],
            GestureCategory.BASIC: COLORS['success'],
            GestureCategory.PEOPLE: COLORS['accent_cyan'],
            GestureCategory.BODY: COLORS['warning'],
            GestureCategory.TIME: COLORS['accent_pink']
        }
        
        color = cat_colors.get(gesture.category, COLORS['accent_cyan'])
        
        self.gesture_label.configure(text=gesture.meaning, text_color=color)
        self.voice_label.configure(text=f'🔊 "{gesture.voice_text}"')
        self.category_label.configure(text=f"{gesture.category.value}")
        self.confidence_bar.set(confidence)
        self.confidence_bar.configure(progress_color=color)


class HospitalSignApp(ctk.CTk):
    """Main Application Window - Enhanced Version."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("🏥 Hospital Sign Language Communication System - AI Enhanced")
        self.geometry("1500x950")
        self.minsize(1300, 850)
        self.configure(fg_color=COLORS['bg_dark'])
        
        # Initialize core components
        self.detector = HandDetector(max_hands=1, detection_confidence=0.7)
        self.classifier = GestureClassifier()
        self.tts = HospitalTTS()
        self.alert_system = EmergencyAlertSystem()
        self.alert_overlay = VisualAlertOverlay()
        self.alert_overlay.attach(self.alert_system)
        
        # Initialize enhanced components
        self._init_ai_agent()
        self._init_notification_service()
        self._init_cnn_classifier()
        
        # Register alert callback
        self.alert_system.register_callback(self._on_alert)
        
        # Camera
        self.cap = None
        self.is_running = False
        self.calibrating = True
        self.calibration_frames = 0
        self.calibration_target = 30
        
        # Gesture state
        self.last_spoken_gesture = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0
        
        # Build UI
        self._build_ui()
        
        # Start camera
        self._start_camera()
        
        # Protocol for window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _init_ai_agent(self):
        """Initialize AI Agent."""
        self.ai_agent = None
        if AI_AGENT_AVAILABLE:
            try:
                self.ai_agent = HospitalAIAgent()
                print("[App] ✅ AI Agent initialized")
            except Exception as e:
                print(f"[App] ⚠️ AI Agent init failed: {e}")
    
    def _init_notification_service(self):
        """Initialize Notification Service."""
        self.notification_service = None
        if NOTIFICATION_AVAILABLE:
            try:
                self.notification_service = HospitalNotificationService()
                print("[App] ✅ Notification service initialized")
            except Exception as e:
                print(f"[App] ⚠️ Notification service init failed: {e}")
    
    def _init_cnn_classifier(self):
        """Initialize CNN Classifier."""
        self.hybrid_classifier = None
        if CNN_AVAILABLE:
            try:
                self.hybrid_classifier = HybridGestureClassifier()
                print("[App] ✅ Hybrid CNN classifier initialized")
            except Exception as e:
                print(f"[App] ⚠️ CNN classifier init failed: {e}")
    
    def _build_ui(self):
        """Build the user interface."""
        # Main container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header
        self._build_header()
        
        # Content area - 3 columns
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # Left column - Video + Detection
        self.left_column = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.left_column.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        self.video_panel = VideoPanel(self.left_column)
        self.video_panel.pack(fill="both", expand=True, pady=(0, 8))
        
        self.detection_panel = DetectionPanel(self.left_column, height=160)
        self.detection_panel.pack(fill="x")
        
        # Middle column - AI + Notifications + Patient Info
        self.middle_column = ctk.CTkFrame(self.content_frame, fg_color="transparent", width=340)
        self.middle_column.pack(side="left", fill="y", padx=8)
        self.middle_column.pack_propagate(False)
        
        # AI Response Panel
        self.ai_panel = AIResponsePanel(self.middle_column, height=280)
        self.ai_panel.pack(fill="x", pady=(0, 8))
        
        # Update AI status
        if self.ai_agent and self.ai_agent.is_available:
            self.ai_panel.set_status("Online (Groq)", True)
        else:
            self.ai_panel.set_status("Offline Mode", False)
        
        # Notification Panel
        self.notification_panel = NotificationPanel(self.middle_column, height=180)
        self.notification_panel.pack(fill="x", pady=(0, 8))
        self.notification_panel.set_button_commands(
            self._quick_nurse_alert,
            self._quick_emergency_alert,
            self._quick_family_notify
        )
        
        # Patient Info Panel
        self.patient_panel = PatientInfoPanel(self.middle_column)
        self.patient_panel.pack(fill="x")
        self.patient_panel.save_btn.configure(command=self._save_patient_info)
        
        # Right column - History + Guide
        self.right_column = ctk.CTkFrame(self.content_frame, fg_color="transparent", width=320)
        self.right_column.pack(side="right", fill="y", padx=(8, 0))
        self.right_column.pack_propagate(False)
        
        self.history_panel = CommunicationHistoryPanel(self.right_column)
        self.history_panel.pack(fill="both", expand=True, pady=(0, 8))
        
        self.guide_panel = GestureGuidePanel(self.right_column, self.classifier, height=280)
        self.guide_panel.pack(fill="x")
    
    def _build_header(self):
        """Build the header section."""
        header_frame = ctk.CTkFrame(self.main_container, fg_color=COLORS['bg_card'], 
                                    corner_radius=15, height=65)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=12)
        
        logo_label = ctk.CTkLabel(
            title_frame,
            text="🏥",
            font=ctk.CTkFont(size=28)
        )
        logo_label.pack(side="left", padx=(0, 10))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="Hospital Sign Language System",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLORS['text_primary']
        )
        title_label.pack(side="left")
        
        # Feature badges
        badge_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        badge_frame.pack(side="left", padx=(15, 0))
        
        ai_badge = ctk.CTkLabel(
            badge_frame,
            text="🤖 AI" if self.ai_agent else "⚪ AI",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['accent_cyan'] if self.ai_agent else COLORS['text_secondary'],
            fg_color=COLORS['bg_secondary'],
            corner_radius=5,
            padx=8,
            pady=2
        )
        ai_badge.pack(side="left", padx=2)
        
        ml_badge = ctk.CTkLabel(
            badge_frame,
            text="🧠 ML" if self.hybrid_classifier else "⚪ ML",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['accent_purple'] if self.hybrid_classifier else COLORS['text_secondary'],
            fg_color=COLORS['bg_secondary'],
            corner_radius=5,
            padx=8,
            pady=2
        )
        ml_badge.pack(side="left", padx=2)
        
        notif_badge = ctk.CTkLabel(
            badge_frame,
            text="📱 SMS" if self.notification_service else "⚪ SMS",
            font=ctk.CTkFont(size=10),
            text_color=COLORS['success'] if self.notification_service else COLORS['text_secondary'],
            fg_color=COLORS['bg_secondary'],
            corner_radius=5,
            padx=8,
            pady=2
        )
        notif_badge.pack(side="left", padx=2)
        
        # Controls
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.pack(side="right", padx=20)
        
        # Mute button
        self.mute_var = ctk.BooleanVar(value=False)
        self.mute_btn = ctk.CTkSwitch(
            controls_frame,
            text="🔊 Voice",
            variable=self.mute_var,
            command=self._toggle_mute,
            onvalue=False,
            offvalue=True,
            fg_color=COLORS['bg_secondary'],
            progress_color=COLORS['success']
        )
        self.mute_btn.pack(side="left", padx=10)
        
        # Clear history button
        clear_btn = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear",
            width=70,
            height=28,
            fg_color=COLORS['bg_secondary'],
            hover_color=COLORS['danger'],
            command=self._clear_history
        )
        clear_btn.pack(side="left", padx=10)
        
        # Emergency alert indicator
        self.alert_indicator = ctk.CTkLabel(
            controls_frame,
            text="",
            font=ctk.CTkFont(size=18),
            text_color=COLORS['danger']
        )
        self.alert_indicator.pack(side="left", padx=10)
    
    def _start_camera(self):
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.video_panel.set_status("Camera not found!", COLORS['danger'])
            return
        
        self.is_running = True
        self.video_panel.set_status("Calibrating...", COLORS['warning'])
        
        # Start video thread
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        
        # Speak greeting
        self.after(1000, self.tts.speak_greeting)
    
    def _video_loop(self):
        """Main video processing loop."""
        prev_landmarks = None
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame, hands = self.detector.find_hands(frame, draw=True)
            
            # Calibration phase
            if self.calibrating:
                self.calibration_frames += 1
                progress = int((self.calibration_frames / self.calibration_target) * 100)
                
                cv2.putText(frame, f"Calibrating: {progress}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if self.calibration_frames >= self.calibration_target:
                    self.calibrating = False
                    self.after(0, lambda: self.video_panel.set_status("Ready", COLORS['success']))
                    self.tts.speak("System ready. You can now show gestures.", force=True)
            
            else:
                # Process gestures
                if hands:
                    hand = hands[0]
                    landmarks = hand['landmarks']
                    
                    # Use hybrid classifier if available
                    gesture = None
                    confidence = 1.0
                    
                    if self.hybrid_classifier:
                        result = self.hybrid_classifier.classify(frame, landmarks, prev_landmarks)
                        if result:
                            gesture = self.classifier.gestures.get(result['gesture_id'])
                            confidence = result.get('confidence', 1.0)
                            # Update ML indicator
                            self.after(0, lambda: self.video_panel.set_ml_status(True))
                    else:
                        gesture = self.classifier.classify_with_smoothing(landmarks, prev_landmarks)
                    
                    if gesture:
                        # Check for emergency
                        is_emergency = self.alert_system.is_emergency_gesture(gesture.id)
                        
                        # Get AI response
                        ai_response = None
                        if self.ai_agent:
                            try:
                                ai_response = self.ai_agent.process_gesture(
                                    gesture.id, gesture.meaning, 
                                    gesture.voice_text, is_emergency
                                )
                                # Update AI panel
                                suggestions = self.ai_agent.get_nurse_suggestions(gesture.id)
                                self.after(0, lambda r=ai_response, e=is_emergency, s=suggestions: 
                                          self._update_ai_panel(r, e, s))
                            except Exception as e:
                                print(f"[App] AI error: {e}")
                        
                        # Send notification for important gestures
                        if self.notification_service and gesture.priority >= 5:
                            try:
                                self.notification_service.send_gesture_notification(
                                    gesture.id, gesture.meaning,
                                    gesture.voice_text, gesture.priority, is_emergency
                                )
                            except Exception as e:
                                print(f"[App] Notification error: {e}")
                        
                        # Speak the gesture
                        if not self.mute_var.get():
                            spoken = self.tts.speak_gesture(
                                gesture.voice_text, 
                                gesture_id=gesture.id, 
                                is_emergency=is_emergency
                            )
                            
                            if spoken:
                                # Trigger alert if emergency
                                if is_emergency:
                                    self.alert_system.trigger_alert(
                                        gesture.id, gesture.name, gesture.voice_text, gesture.priority
                                    )
                                
                                # Update UI
                                self.after(0, lambda g=gesture, c=confidence, r=ai_response: 
                                          self._update_detection(g, c, r))
                        
                        # Draw gesture info on frame
                        cv2.putText(frame, gesture.meaning, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    prev_landmarks = landmarks
                else:
                    self.after(0, lambda: self.detection_panel.update_detection(None))
                    self.after(0, lambda: self.video_panel.set_ml_status(False))
                    prev_landmarks = None
            
            # Apply alert overlay if active
            if self.alert_system.is_alerting:
                frame = self.alert_overlay.apply_to_frame(frame)
            
            # Update video panel
            self.after(0, lambda f=frame.copy(): self.video_panel.update_frame(f))
            
            time.sleep(0.03)  # ~30 FPS
    
    def _update_detection(self, gesture: Gesture, confidence: float = 1.0, ai_response: str = None):
        """Update detection panel and history."""
        self.detection_panel.update_detection(gesture, confidence)
        self.history_panel.add_message(gesture, datetime.now(), ai_response)
    
    def _update_ai_panel(self, response: str, is_emergency: bool, suggestions: list):
        """Update AI response panel."""
        self.ai_panel.update_response(response, is_emergency)
        self.ai_panel.update_suggestions(suggestions)
    
    def _on_alert(self, alert: Alert):
        """Handle emergency alert."""
        def flash():
            for _ in range(5):
                self.alert_indicator.configure(text="🚨 EMERGENCY")
                time.sleep(0.3)
                self.alert_indicator.configure(text="")
                time.sleep(0.3)
        
        threading.Thread(target=flash, daemon=True).start()
    
    def _toggle_mute(self):
        """Toggle voice output."""
        if self.mute_var.get():
            self.mute_btn.configure(text="🔇 Muted")
        else:
            self.mute_btn.configure(text="🔊 Voice")
    
    def _clear_history(self):
        """Clear communication history."""
        self.history_panel.clear_history()
        if self.ai_agent:
            self.ai_agent.clear_history()
    
    def _save_patient_info(self):
        """Save patient information."""
        info = self.patient_panel.get_patient_info()
        
        # Update AI agent context
        if self.ai_agent:
            context = PatientContext(
                name=info['name'],
                room_number=info['room'],
                emergency_contact=info['emergency_contact']
            )
            self.ai_agent.set_patient_context(context)
        
        # Update notification service
        if self.notification_service:
            self.notification_service.set_patient_info(info['name'], info['room'])
            if info['emergency_contact']:
                self.notification_service.config.family_phones = [info['emergency_contact']]
        
        print(f"[App] Patient info saved: {info['name']} in {info['room']}")
    
    def _quick_nurse_alert(self):
        """Quick alert to nurse station."""
        if self.notification_service:
            self.notification_service.send_nurse_alert("Patient requesting assistance", priority=7)
            self.notification_panel.update_last_notification("Nurse alert sent")
        self.tts.speak("Alerting the nurse station now.", force=True)
    
    def _quick_emergency_alert(self):
        """Quick emergency alert."""
        if self.notification_service:
            self.notification_service.send_emergency_alert("Patient emergency - immediate assistance required!")
            self.notification_panel.update_last_notification("EMERGENCY alert sent!")
        self.tts.speak("Emergency! Alerting all staff immediately!", force=True)
        self.alert_system.trigger_alert(0, "MANUAL_EMERGENCY", "Manual emergency triggered", 10)
    
    def _quick_family_notify(self):
        """Quick notification to family."""
        if self.notification_service:
            self.notification_service.send_family_notification("Patient is doing well and sends their regards.")
            self.notification_panel.update_last_notification("Family notified")
        self.tts.speak("Notifying your family now.", force=True)
    
    def _on_close(self):
        """Clean up and close application."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.detector.release()
        self.tts.stop()
        
        if self.notification_service:
            self.notification_service.stop()
        
        self.destroy()


def main():
    """Main entry point."""
    print("=" * 60)
    print("🏥 Hospital Sign Language Communication System")
    print("   Enhanced with AI Agent, CNN, and Notifications")
    print("=" * 60)
    print("\nFeatures:")
    print(f"  🤖 AI Agent: {'Available' if AI_AGENT_AVAILABLE else 'Not installed'}")
    print(f"  🧠 CNN Model: {'Available' if CNN_AVAILABLE else 'Not installed'}")
    print(f"  📱 Notifications: {'Available' if NOTIFICATION_AVAILABLE else 'Not installed'}")
    print("\nStarting application...")
    print("Press ESC in camera window or close GUI to exit.\n")
    
    app = HospitalSignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
