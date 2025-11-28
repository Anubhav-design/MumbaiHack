"""
Quick launcher for Hospital Sign Language System - Enhanced Version
Run this file to start the application

Features:
- Auto-installs missing dependencies
- Loads configuration from .env
- Displays feature status before launch
"""

import subprocess
import sys
import os

def check_core_dependencies():
    """Check if core packages are installed."""
    required = ['cv2', 'mediapipe', 'customtkinter', 'pyttsx3', 'PIL', 'numpy']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            pip_names = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow'
            }
            missing.append(pip_names.get(package, package))
    
    if missing:
        print("❌ Missing core packages detected!")
        print(f"   Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Core packages installed!\n")
    else:
        print("✅ Core packages: OK")

def check_enhanced_features():
    """Check enhanced feature dependencies."""
    features = {
        'AI Agent (Groq)': ('groq', 'groq'),
        'Notifications (Twilio)': ('twilio', 'twilio'),
        'Deep Learning (TensorFlow)': ('tensorflow', 'tensorflow'),
        'Environment Config': ('dotenv', 'python-dotenv'),
    }
    
    status = {}
    
    print("\n📋 Enhanced Features Status:")
    print("-" * 40)
    
    for feature, (module, pip_name) in features.items():
        try:
            __import__(module)
            status[feature] = True
            print(f"  ✅ {feature}: Installed")
        except ImportError:
            status[feature] = False
            print(f"  ⚪ {feature}: Not installed")
    
    return status

def offer_install_enhanced():
    """Offer to install enhanced features."""
    print("\n" + "-" * 40)
    print("Would you like to install enhanced features?")
    print("This will enable AI responses, SMS/WhatsApp alerts, and ML recognition.")
    print("\nEnhanced packages:")
    print("  - groq (AI Agent - Free API)")
    print("  - twilio (SMS/WhatsApp notifications)")
    print("  - tensorflow (CNN gesture recognition)")
    print("  - python-dotenv (Configuration management)")
    
    try:
        choice = input("\nInstall enhanced features? [y/N]: ").strip().lower()
        if choice == 'y':
            packages = ['groq', 'twilio', 'tensorflow', 'python-dotenv']
            print(f"\n📦 Installing: {', '.join(packages)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
            print("✅ Enhanced features installed!")
            return True
    except Exception as e:
        print(f"⚠️ Installation skipped: {e}")
    
    return False

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 API Key Status:")
    print("-" * 40)
    
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    groq_key = os.environ.get("GROQ_API_KEY", "")
    twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    
    if groq_key and groq_key != "gsk_your_groq_api_key_here":
        print("  ✅ Groq API Key: Configured")
    else:
        print("  ⚪ Groq API Key: Not set (AI will use offline mode)")
        print("     Get free key at: https://console.groq.com/keys")
    
    if twilio_sid and twilio_sid != "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
        print("  ✅ Twilio: Configured")
    else:
        print("  ⚪ Twilio: Not set (Notifications will be simulated)")
        print("     Sign up at: https://www.twilio.com/try-twilio")
    
    print("\n  📝 To configure: Copy env_example.txt to .env and add your keys")

def display_banner():
    """Display application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🏥  HOSPITAL SIGN LANGUAGE COMMUNICATION SYSTEM  🏥        ║
║                                                              ║
║   AI-Enhanced Deaf Patient Communication Assistant           ║
║                                                              ║
║   Features:                                                  ║
║   • Real-time sign language recognition                      ║
║   • 45+ hospital-specific gestures                           ║
║   • AI-powered intelligent responses                         ║
║   • SMS/WhatsApp emergency alerts                            ║
║   • CNN-based gesture classification                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main launcher function."""
    display_banner()
    
    print("=" * 60)
    print("🔧 System Check")
    print("=" * 60)
    
    # Check core dependencies
    print("\n📦 Checking core dependencies...")
    check_core_dependencies()
    
    # Check enhanced features
    feature_status = check_enhanced_features()
    
    # Check if any enhanced features are missing
    all_installed = all(feature_status.values())
    
    if not all_installed:
        # Offer to install missing packages (non-interactive for automation)
        print("\n💡 Tip: Run 'pip install -r requirements.txt' for all features")
    
    # Check API keys
    check_api_keys()
    
    # Launch application
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING APPLICATION")
    print("=" * 60)
    print("\n⏳ Starting... Please wait for the camera to initialize.\n")
    
    try:
        from hospital_sign_app import main as app_main
        app_main()
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your webcam is connected")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify camera is not being used by another application")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
