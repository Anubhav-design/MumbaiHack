"""
Configuration loader for Hospital Sign Language System
Loads settings from environment variables or .env file
"""

import os
from typing import Optional, List
from dataclasses import dataclass, field

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[Config] Loaded settings from .env file")
except ImportError:
    print("[Config] python-dotenv not installed, using environment variables only")


@dataclass
class AIConfig:
    """AI Agent configuration"""
    api_key: str = ""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    max_tokens: int = 150
    
    @classmethod
    def from_env(cls) -> "AIConfig":
        return cls(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get("AI_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.environ.get("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("AI_MAX_TOKENS", "150"))
        )


@dataclass
class TwilioConfig:
    """Twilio notification configuration"""
    account_sid: str = ""
    auth_token: str = ""
    phone_number: str = ""
    whatsapp_number: str = "whatsapp:+14155238886"
    
    @classmethod
    def from_env(cls) -> "TwilioConfig":
        return cls(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            phone_number=os.environ.get("TWILIO_PHONE_NUMBER", ""),
            whatsapp_number=os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        )
    
    @property
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.phone_number)


@dataclass
class HospitalConfig:
    """Hospital contacts configuration"""
    nurse_station_phone: str = ""
    emergency_phones: List[str] = field(default_factory=list)
    family_phones: List[str] = field(default_factory=list)
    notification_cooldown: int = 30
    
    @classmethod
    def from_env(cls) -> "HospitalConfig":
        emergency = os.environ.get("EMERGENCY_PHONES", "")
        family = os.environ.get("FAMILY_PHONES", "")
        
        return cls(
            nurse_station_phone=os.environ.get("NURSE_STATION_PHONE", ""),
            emergency_phones=[p.strip() for p in emergency.split(",") if p.strip()],
            family_phones=[p.strip() for p in family.split(",") if p.strip()],
            notification_cooldown=int(os.environ.get("NOTIFICATION_COOLDOWN", "30"))
        )


@dataclass
class PatientConfig:
    """Default patient configuration"""
    name: str = "Patient"
    room_number: str = "ICU-101"
    
    @classmethod
    def from_env(cls) -> "PatientConfig":
        return cls(
            name=os.environ.get("DEFAULT_PATIENT_NAME", "Patient"),
            room_number=os.environ.get("DEFAULT_ROOM_NUMBER", "ICU-101")
        )


@dataclass
class AppConfig:
    """Complete application configuration"""
    ai: AIConfig = field(default_factory=AIConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    hospital: HospitalConfig = field(default_factory=HospitalConfig)
    patient: PatientConfig = field(default_factory=PatientConfig)
    
    # App settings
    camera_index: int = 0
    detection_confidence: float = 0.7
    gesture_cooldown: float = 2.0
    calibration_frames: int = 30
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment"""
        return cls(
            ai=AIConfig.from_env(),
            twilio=TwilioConfig.from_env(),
            hospital=HospitalConfig.from_env(),
            patient=PatientConfig.from_env(),
            camera_index=int(os.environ.get("CAMERA_INDEX", "0")),
            detection_confidence=float(os.environ.get("DETECTION_CONFIDENCE", "0.7")),
            gesture_cooldown=float(os.environ.get("GESTURE_COOLDOWN", "2.0")),
            calibration_frames=int(os.environ.get("CALIBRATION_FRAMES", "30"))
        )
    
    def print_status(self):
        """Print configuration status"""
        print("\n" + "=" * 50)
        print("📋 Configuration Status")
        print("=" * 50)
        print(f"  🤖 AI Agent: {'✅ Configured' if self.ai.api_key else '⚠️ No API key'}")
        print(f"  📱 Twilio: {'✅ Configured' if self.twilio.is_configured else '⚠️ Not configured'}")
        print(f"  🏥 Nurse Station: {'✅ ' + self.hospital.nurse_station_phone if self.hospital.nurse_station_phone else '⚠️ Not set'}")
        print(f"  🚨 Emergency Contacts: {len(self.hospital.emergency_phones)} configured")
        print(f"  👨‍👩‍👧 Family Contacts: {len(self.hospital.family_phones)} configured")
        print(f"  📷 Camera: Index {self.camera_index}")
        print("=" * 50 + "\n")


# Global configuration instance
config = AppConfig.load()


def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment"""
    global config
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    config = AppConfig.load()
    return config


# Print status on import
if __name__ == "__main__":
    config.print_status()

