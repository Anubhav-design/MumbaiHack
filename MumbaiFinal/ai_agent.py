"""
AI Agent for Hospital Sign Language System
Uses Groq API for fast, intelligent responses
Provides context-aware communication assistance
"""

import os
import json
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field

# Try to import Groq, fallback to basic responses if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[AI Agent] Groq not installed. Run: pip install groq")


@dataclass
class ConversationMessage:
    """Single message in conversation"""
    role: str  # 'patient', 'system', 'assistant', 'nurse'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    gesture_id: Optional[int] = None
    emotion: Optional[str] = None


@dataclass
class PatientContext:
    """Patient context for personalized responses"""
    patient_id: str = "PATIENT_001"
    name: str = "Patient"
    age: Optional[int] = None
    condition: Optional[str] = None
    room_number: Optional[str] = None
    language: str = "English"
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    emergency_contact: Optional[str] = None
    admission_reason: Optional[str] = None


class HospitalAIAgent:
    """
    Intelligent AI Agent for Hospital Communication
    
    Features:
    - Context-aware responses based on patient history
    - Medical terminology understanding
    - Suggests follow-up questions for nurses
    - Multi-turn conversation memory
    - Emergency escalation detection
    """
    
    SYSTEM_PROMPT = """You are a compassionate AI assistant in a hospital, helping deaf patients communicate with medical staff.

Your role:
1. Understand patient's gestures/messages and provide helpful responses
2. Suggest follow-up questions nurses should ask
3. Detect emergencies and escalate appropriately
4. Be warm, reassuring, and professional
5. Keep responses concise (2-3 sentences max for speech)

Patient Context:
- Name: {patient_name}
- Room: {room_number}
- Condition: {condition}
- Allergies: {allergies}

Recent conversation history is provided. Respond to the latest patient gesture/message.

IMPORTANT: 
- If this is an EMERGENCY, start with "EMERGENCY ALERT:" 
- Always be empathetic and reassuring
- Suggest specific actions for hospital staff when relevant
- Keep responses under 50 words for text-to-speech clarity"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI Agent.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env variable)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.client = None
        self.is_available = False
        
        # Initialize Groq client
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.is_available = True
                print("[AI Agent] ✅ Groq AI Agent initialized successfully")
            except Exception as e:
                print(f"[AI Agent] ⚠️ Failed to initialize Groq: {e}")
        else:
            print("[AI Agent] ⚠️ Running in offline mode (no API key)")
        
        # Conversation history
        self.conversation_history: List[ConversationMessage] = []
        self.max_history = 20
        
        # Patient context
        self.patient_context = PatientContext()
        
        # Response queue for async processing
        self.response_queue = queue.Queue()
        self.is_processing = False
        
        # Callbacks
        self.response_callbacks: List[Callable] = []
        
        # Pre-defined responses for offline mode
        self.offline_responses = self._load_offline_responses()
        
        # Model configuration
        self.model = "llama-3.1-8b-instant"  # Fast and efficient
        self.temperature = 0.7
        self.max_tokens = 150
    
    def _load_offline_responses(self) -> Dict[int, str]:
        """Load pre-defined responses for offline mode."""
        return {
            # Emergency responses
            0: "I understand you need immediate help. I'm alerting the medical team right now. Stay calm, help is on the way.",
            1: "This is urgent. I'm notifying the nurse station immediately. Someone will be with you very soon.",
            2: "I can see you're in severe pain. I'm calling for pain management assistance right away.",
            35: "I'm detecting breathing difficulty. Emergency team is being notified. Try to stay calm and breathe slowly.",
            
            # Basic responses
            3: "I understand - yes. Is there anything specific you need?",
            4: "I understand - no. Please let me know if you need anything else.",
            5: "I understand you want me to wait. Take your time.",
            6: "That's great to hear you're feeling okay! Let me know if anything changes.",
            7: "I'm sorry you're not feeling well. I'll notify the nurse to check on you.",
            9: "You're welcome! I'm here to help you communicate.",
            
            # Medical needs
            13: "I'm paging the doctor for you now. They should arrive shortly.",
            14: "I understand you're in pain. Can you show me where it hurts? I'll get pain medication organized.",
            15: "I'll arrange for water to be brought to you right away.",
            16: "I'm notifying the nurse about your medication. They'll bring it shortly.",
            17: "The nurse has been called and will be with you soon.",
            18: "I understand you need the bathroom. Let me call for assistance.",
            19: "I'll arrange for food to be brought to you. Any dietary restrictions I should note?",
            25: "I'm sorry you're feeling nauseous. I'll get anti-nausea medication and a basin for you.",
            26: "I'm contacting your family now. They'll be notified of your message.",
            
            # Feelings
            28: "I'll get you an extra blanket right away. You should feel warmer soon.",
            29: "I'm detecting you may have a fever. I'll have the nurse check your temperature.",
            30: "I understand you're tired. I'll dim the lights and let the staff know you need rest.",
            31: "It's okay to feel scared. You're in good hands here. Would you like me to call someone?",
            33: "That's wonderful to hear you're feeling better! I'll note this improvement.",
            34: "Dizziness can be concerning. I'm alerting the nurse to check on you.",
            
            # Body parts
            38: "I understand your head hurts. I'll request pain relief for your headache.",
            39: "Chest pain is serious. I'm alerting the cardiac team immediately. Try to stay calm.",
            40: "I understand your stomach is hurting. I'll notify the nurse for assessment.",
            41: "Back pain noted. I'll arrange for proper positioning and pain management.",
            42: "I understand your leg/foot is bothering you. I'll have someone check on it.",
        }
    
    def set_patient_context(self, context: PatientContext):
        """Update patient context for personalized responses."""
        self.patient_context = context
        print(f"[AI Agent] Patient context updated: {context.name}")
    
    def register_callback(self, callback: Callable[[str, bool], None]):
        """
        Register callback for AI responses.
        
        Args:
            callback: Function(response_text, is_emergency)
        """
        self.response_callbacks.append(callback)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with patient context."""
        return self.SYSTEM_PROMPT.format(
            patient_name=self.patient_context.name,
            room_number=self.patient_context.room_number or "Not assigned",
            condition=self.patient_context.condition or "Under observation",
            allergies=", ".join(self.patient_context.allergies) if self.patient_context.allergies else "None known"
        )
    
    def _build_messages(self, current_gesture: str) -> List[Dict]:
        """Build message history for API call."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        
        # Add recent conversation history
        for msg in self.conversation_history[-10:]:
            role = "user" if msg.role == "patient" else "assistant"
            messages.append({"role": role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": f"[Patient gesture/message]: {current_gesture}"})
        
        return messages
    
    def process_gesture(self, gesture_id: int, gesture_meaning: str, 
                       voice_text: str, is_emergency: bool = False) -> str:
        """
        Process a detected gesture and generate AI response.
        
        Args:
            gesture_id: ID of the detected gesture
            gesture_meaning: Human-readable meaning
            voice_text: Text that will be spoken
            is_emergency: Whether this is an emergency gesture
            
        Returns:
            AI-generated response string
        """
        # Add to conversation history
        patient_msg = ConversationMessage(
            role="patient",
            content=f"{gesture_meaning}: {voice_text}",
            gesture_id=gesture_id
        )
        self.conversation_history.append(patient_msg)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Generate response
        if self.is_available:
            response = self._get_ai_response(voice_text, is_emergency)
        else:
            response = self._get_offline_response(gesture_id, is_emergency)
        
        # Add AI response to history
        ai_msg = ConversationMessage(
            role="assistant",
            content=response
        )
        self.conversation_history.append(ai_msg)
        
        # Notify callbacks
        for callback in self.response_callbacks:
            try:
                callback(response, is_emergency)
            except Exception as e:
                print(f"[AI Agent] Callback error: {e}")
        
        return response
    
    def _get_ai_response(self, gesture_text: str, is_emergency: bool) -> str:
        """Get response from Groq AI."""
        try:
            messages = self._build_messages(gesture_text)
            
            # Use faster model for emergencies
            model = "llama-3.1-8b-instant" if is_emergency else self.model
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5 if is_emergency else self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"[AI Agent] Response: {ai_response[:100]}...")
            return ai_response
            
        except Exception as e:
            print(f"[AI Agent] API error: {e}")
            return self._get_offline_response_generic(gesture_text, is_emergency)
    
    def _get_offline_response(self, gesture_id: int, is_emergency: bool) -> str:
        """Get pre-defined offline response."""
        if gesture_id in self.offline_responses:
            return self.offline_responses[gesture_id]
        return self._get_offline_response_generic("", is_emergency)
    
    def _get_offline_response_generic(self, gesture_text: str, is_emergency: bool) -> str:
        """Generate generic offline response."""
        if is_emergency:
            return "I understand this is urgent. I'm alerting the medical team immediately. Help is on the way."
        return "I understand your message. I'm notifying the appropriate staff to assist you."
    
    def get_nurse_suggestions(self, gesture_id: int) -> List[str]:
        """
        Get suggested follow-up questions for nurses.
        
        Args:
            gesture_id: The gesture that was detected
            
        Returns:
            List of suggested questions
        """
        suggestions = {
            # Pain-related
            14: [
                "On a scale of 1-10, how severe is your pain?",
                "Is the pain constant or does it come and go?",
                "When did the pain start?",
            ],
            38: [  # Headache
                "Is this a new headache or recurring?",
                "Do you have any vision changes?",
                "Have you taken any medication for it?",
            ],
            39: [  # Chest
                "Is the pain sharp or dull?",
                "Does it radiate to your arm or jaw?",
                "Are you having trouble breathing?",
            ],
            40: [  # Stomach
                "When did you last eat?",
                "Any nausea or vomiting?",
                "Is it cramping or constant pain?",
            ],
            
            # Emergency
            0: [
                "What specifically is wrong?",
                "Are you having trouble breathing?",
                "Do you have chest pain?",
            ],
            35: [  # Breathing
                "When did the breathing difficulty start?",
                "Do you have asthma or other lung conditions?",
                "Are you experiencing any chest tightness?",
            ],
            
            # Needs
            15: [  # Water
                "Would you prefer cold or room temperature water?",
                "Are there any fluid restrictions I should know about?",
            ],
            19: [  # Food
                "Do you have any food allergies?",
                "Any dietary restrictions?",
                "Would you prefer hot or cold food?",
            ],
            18: [  # Bathroom
                "Do you need assistance getting up?",
                "Would you prefer a bedpan or to go to the bathroom?",
            ],
            
            # Feelings
            31: [  # Scared
                "What specifically are you worried about?",
                "Would you like me to explain what's happening?",
                "Would you like to speak with someone?",
            ],
            29: [  # Fever
                "Do you feel chills as well?",
                "Any body aches?",
                "When did you start feeling warm?",
            ],
        }
        
        return suggestions.get(gesture_id, [
            "Is there anything else you need?",
            "Are you comfortable?",
            "Would you like me to call someone?",
        ])
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for handoff."""
        if not self.conversation_history:
            return "No conversation recorded yet."
        
        summary_parts = []
        for msg in self.conversation_history[-10:]:
            time_str = msg.timestamp.strftime("%H:%M")
            summary_parts.append(f"[{time_str}] {msg.role.upper()}: {msg.content[:100]}")
        
        return "\n".join(summary_parts)
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("[AI Agent] Conversation history cleared")
    
    def analyze_patient_state(self) -> Dict:
        """
        Analyze patient's current state based on recent gestures.
        
        Returns:
            Dict with analysis results
        """
        if not self.conversation_history:
            return {"status": "unknown", "mood": "neutral", "needs_attention": False}
        
        recent_gestures = [
            msg.gesture_id for msg in self.conversation_history[-5:]
            if msg.gesture_id is not None
        ]
        
        # Emergency detection
        emergency_ids = {0, 1, 2, 35, 39}
        pain_ids = {14, 38, 40, 41, 42}
        negative_ids = {7, 25, 28, 29, 31, 34}
        positive_ids = {6, 9, 33}
        
        has_emergency = any(g in emergency_ids for g in recent_gestures)
        has_pain = any(g in pain_ids for g in recent_gestures)
        negative_count = sum(1 for g in recent_gestures if g in negative_ids)
        positive_count = sum(1 for g in recent_gestures if g in positive_ids)
        
        # Determine status
        if has_emergency:
            status = "critical"
            mood = "distressed"
        elif has_pain:
            status = "needs_attention"
            mood = "uncomfortable"
        elif negative_count > positive_count:
            status = "monitor"
            mood = "negative"
        elif positive_count > 0:
            status = "stable"
            mood = "positive"
        else:
            status = "stable"
            mood = "neutral"
        
        return {
            "status": status,
            "mood": mood,
            "needs_attention": has_emergency or has_pain,
            "recent_gestures": recent_gestures,
            "emergency_detected": has_emergency,
            "pain_indicators": has_pain
        }


class AIResponsePanel:
    """Helper class to display AI responses in GUI."""
    
    def __init__(self, agent: HospitalAIAgent):
        self.agent = agent
        self.current_response = ""
        self.nurse_suggestions = []
    
    def process_and_display(self, gesture_id: int, gesture_meaning: str, 
                           voice_text: str, is_emergency: bool = False):
        """Process gesture and prepare display data."""
        self.current_response = self.agent.process_gesture(
            gesture_id, gesture_meaning, voice_text, is_emergency
        )
        self.nurse_suggestions = self.agent.get_nurse_suggestions(gesture_id)
        return self.current_response, self.nurse_suggestions


# Test the AI Agent
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Testing Hospital AI Agent")
    print("=" * 60)
    
    # Initialize agent (will work in offline mode without API key)
    agent = HospitalAIAgent()
    
    # Set patient context
    context = PatientContext(
        name="John Doe",
        age=45,
        room_number="ICU-203",
        condition="Post-surgery recovery",
        allergies=["Penicillin"],
        emergency_contact="+91-9876543210"
    )
    agent.set_patient_context(context)
    
    # Test gestures
    test_gestures = [
        (14, "PAIN", "I am in pain. It hurts.", False),
        (15, "WATER", "I need water please.", False),
        (0, "HELP", "Help! Emergency! Patient needs immediate assistance!", True),
        (6, "GOOD", "I am feeling good. Everything is okay.", False),
    ]
    
    print("\n--- Testing Gesture Processing ---\n")
    
    for gesture_id, meaning, voice_text, is_emergency in test_gestures:
        print(f"Gesture: {meaning} (ID: {gesture_id})")
        print(f"Patient says: \"{voice_text}\"")
        
        response = agent.process_gesture(gesture_id, meaning, voice_text, is_emergency)
        print(f"AI Response: {response}")
        
        suggestions = agent.get_nurse_suggestions(gesture_id)
        print(f"Nurse suggestions: {suggestions[:2]}")
        print("-" * 40)
    
    # Test patient state analysis
    print("\n--- Patient State Analysis ---")
    state = agent.analyze_patient_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Test conversation summary
    print("\n--- Conversation Summary ---")
    print(agent.get_conversation_summary())
    
    print("\n✅ AI Agent test complete!")

