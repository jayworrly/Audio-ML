"""
Azure Text-to-Speech Integration
High-quality neural voices for the AI PDF Reader
"""

import os
import tempfile
import wave
from typing import Optional, Dict, List
import threading

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

class AzureTTS:
    def __init__(self, speech_key: str = "", speech_region: str = ""):
        """Initialize Azure TTS with credentials"""
        self.speech_key = speech_key or os.getenv('AZURE_SPEECH_KEY', '')
        self.speech_region = speech_region or os.getenv('AZURE_SPEECH_REGION', '')
        self.speech_config = None
        self.available_voices = []
        self.current_voice = "en-US-JennyNeural"  # Default high-quality voice
        
        if AZURE_AVAILABLE and self.speech_key and self.speech_region:
            self.setup_azure_speech()
    
    def setup_azure_speech(self):
        """Setup Azure Speech Service"""
        try:
            if not self.speech_key or not self.speech_region:
                print("Azure TTS setup failed: Missing credentials")
                return False
            
            print(f"Setting up Azure TTS with region: {self.speech_region}")
            
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, 
                region=self.speech_region
            )
            self.speech_config.speech_synthesis_voice_name = self.current_voice
            
            # Test the connection with a simple synthesis
            test_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            test_result = test_synthesizer.speak_text_async("Test").get()
            
            if test_result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(test_result)
                print(f"Azure TTS test failed: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {cancellation.error_details}")
                return False
            
            # Get available voices
            self.get_available_voices()
            print("Azure TTS setup completed successfully")
            return True
            
        except Exception as e:
            print(f"Azure TTS setup failed: {e}")
            return False
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available Azure neural voices"""
        if not self.speech_config:
            return []
        
        try:
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            voices_result = synthesizer.get_voices_async().get()
            
            self.available_voices = []
            for voice in voices_result.voices:
                if "Neural" in voice.short_name:  # Only neural voices
                    voice_info = {
                        'id': voice.short_name,
                        'name': voice.local_name,
                        'gender': voice.gender.name,
                        'locale': voice.locale,
                        'style_list': getattr(voice, 'style_list', [])
                    }
                    self.available_voices.append(voice_info)
            
            return self.available_voices
        except Exception as e:
            print(f"Failed to get Azure voices: {e}")
            return []
    
    def get_premium_voices(self) -> Dict[str, str]:
        """Get recommended premium voices"""
        return {
            # English voices
            "Jenny (US Female)": "en-US-JennyNeural",
            "Guy (US Male)": "en-US-GuyNeural", 
            "Aria (US Female)": "en-US-AriaNeural",
            "Davis (US Male)": "en-US-DavisNeural",
            "Jane (US Female)": "en-US-JaneNeural",
            "Jason (US Male)": "en-US-JasonNeural",
            "Sara (US Female)": "en-US-SaraNeural",
            "Tony (US Male)": "en-US-TonyNeural",
            
            # British voices
            "Sonia (UK Female)": "en-GB-SoniaNeural",
            "Ryan (UK Male)": "en-GB-RyanNeural",
            "Libby (UK Female)": "en-GB-LibbyNeural",
            "Maisie (UK Female)": "en-GB-MaisieNeural",
            
            # Australian voices
            "Natasha (AU Female)": "en-AU-NatashaNeural",
            "William (AU Male)": "en-AU-WilliamNeural",
        }
    
    def set_voice(self, voice_name: str):
        """Set the current voice"""
        if self.speech_config:
            self.current_voice = voice_name
            self.speech_config.speech_synthesis_voice_name = voice_name
    
    def set_speech_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.speech_config:
            # Convert WPM to Azure rate format
            # Azure uses relative rate: -50% to +200%
            # 180 WPM is normal, so we calculate relative to that
            relative_rate = ((rate - 180) / 180) * 100
            relative_rate = max(-50, min(200, relative_rate))  # Clamp to Azure limits
            
            rate_string = f"{relative_rate:+.0f}%"
            self.speech_config.speech_synthesis_voice_name = f"{self.current_voice}"
    
    def speak_text(self, text: str, callback=None) -> bool:
        """Speak text using Azure TTS"""
        if not self.speech_config:
            print("Azure TTS: No speech config available")
            return False
        
        try:
            # Limit text length for better reliability
            if len(text) > 5000:
                text = text[:5000] + "..."
            
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            
            def synthesis_completed(evt):
                if callback:
                    callback()
            
            def synthesis_canceled(evt):
                print(f"Azure TTS canceled: {evt.reason}")
                if evt.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {evt.error_details}")
            
            synthesizer.synthesis_completed.connect(synthesis_completed)
            synthesizer.synthesis_canceled.connect(synthesis_canceled)
            
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                print(f"Azure TTS canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {cancellation.error_details}")
                    print("Check your Azure credentials and quota")
                return False
            else:
                print(f"Azure TTS failed: {result.reason}")
                return False
                
        except Exception as e:
            print(f"Azure TTS error: {e}")
            return False
    
    def save_to_file(self, text: str, file_path: str, rate: int = 180, volume: float = 1.0) -> bool:
        """Save text as audio file using Azure TTS"""
        if not self.speech_config:
            print("Azure TTS: No speech config for file save")
            return False
        
        try:
            # Limit text length for better reliability
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(filename=file_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Create SSML for better control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{self.current_voice}">
                    <prosody rate="{self._convert_rate_to_ssml(rate)}" volume="{volume:.1f}">
                        {self._escape_ssml(text)}
                    </prosody>
                </voice>
            </speak>
            """
            
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                print(f"Azure TTS file save canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {cancellation.error_details}")
                return False
            else:
                print(f"Azure TTS file save failed: {result.reason}")
                return False
                
        except Exception as e:
            print(f"Azure TTS file save error: {e}")
            return False
    
    def _convert_rate_to_ssml(self, wpm: int) -> str:
        """Convert WPM to SSML rate format"""
        # 180 WPM is considered normal
        if wpm <= 120:
            return "x-slow"
        elif wpm <= 150:
            return "slow"
        elif wpm <= 210:
            return "medium"
        elif wpm <= 250:
            return "fast"
        else:
            return "x-fast"
    
    def _escape_ssml(self, text: str) -> str:
        """Escape special characters for SSML"""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        return text
    
    def is_available(self) -> bool:
        """Check if Azure TTS is available and configured"""
        return AZURE_AVAILABLE and bool(self.speech_config)
    
    def preview_voice(self, voice_name: str, sample_text: str = None) -> bool:
        """Preview a specific voice"""
        if not sample_text:
            sample_text = "Hello! This is a preview of how I sound when reading your documents. I'm a high-quality neural voice from Azure."
        
        old_voice = self.current_voice
        self.set_voice(voice_name)
        result = self.speak_text(sample_text)
        self.set_voice(old_voice)  # Restore previous voice
        return result

# Global instance
azure_tts = AzureTTS() 