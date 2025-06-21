# Voice Configuration for AI-Enhanced PDF Reader
# Optional premium voice services configuration

# Azure Cognitive Services (Optional)
# Sign up at: https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/
# Free tier: 5 hours of speech per month
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")  # Your Azure Speech Service key
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")  # e.g., "eastus"

# Preferred Azure voices (high quality neural voices)
AZURE_VOICES = {
    "female_en": "en-US-JennyNeural",     # Natural female voice
    "male_en": "en-US-GuyNeural",         # Natural male voice
    "female_uk": "en-GB-SoniaNeural",     # British female
    "male_uk": "en-GB-RyanNeural",        # British male
    "female_au": "en-AU-NatashaNeural",   # Australian female
    "male_au": "en-AU-WilliamNeural",     # Australian male
}

# Google Cloud Text-to-Speech (Optional)
# Setup at: https://cloud.google.com/text-to-speech
GOOGLE_CREDENTIALS_PATH = ""  # Path to your Google credentials JSON file

# Voice quality preferences
VOICE_PREFERENCES = {
    "use_azure": True,  # Set to True if you have Azure credentials
    "use_google": False,  # Set to True if you have Google credentials
    "prefer_neural": False,  # Prefer neural voices when available
    "default_gender": "female",  # "female" or "male"
    "default_accent": "us",  # "us", "uk", "au", etc.
}

# Instructions for setting up premium voices:
"""
SETUP INSTRUCTIONS:

1. AZURE COGNITIVE SERVICES (Recommended):
   - Go to: https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/
   - Create a free account (5 hours/month free)
   - Create a Speech Service resource
   - Copy your key and region to AZURE_SPEECH_KEY and AZURE_SPEECH_REGION above
   - Set use_azure = True
   - Install: pip install azure-cognitiveservices-speech

2. GOOGLE CLOUD TEXT-TO-SPEECH:
   - Go to: https://cloud.google.com/text-to-speech
   - Create a project and enable the API
   - Create credentials and download JSON file
   - Set GOOGLE_CREDENTIALS_PATH to the JSON file path
   - Set use_google = True
   - Install: pip install google-cloud-texttospeech

3. WINDOWS VOICES (Default):
   - Uses built-in Windows SAPI voices
   - Free but limited quality
   - Works offline
   - No setup required
""" 