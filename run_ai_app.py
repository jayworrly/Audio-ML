#!/usr/bin/env python3
"""
Launcher for AI-Enhanced PDF Text-to-Speech Reader
This script handles dependency installation and provides setup guidance.
"""

import sys
import subprocess
import importlib.util
import os

def print_banner():
    """Print application banner"""
    print("🤖 AI-Enhanced PDF Text-to-Speech Reader")
    print("=" * 50)
    print("Features:")
    print("• 📄 PDF text extraction and reading")
    print("• 🧠 AI document analysis")
    print("• 📝 Automatic summarization")
    print("• 🎯 Key point extraction")
    print("• 😊 Sentiment analysis")
    print("• ⚙️ Smart voice optimization")
    print("• 🎵 Advanced playback controls")
    print("=" * 50)
    print()

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--quiet"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    # Core dependencies (required for basic functionality)
    core_deps = [
        ('pypdf', 'pypdf==4.0.1'),
        ('pyttsx3', 'pyttsx3==2.90'),
        ('tkinter', None)  # Built-in
    ]
    
    # ML dependencies (for AI features)
    ml_deps = [
        ('transformers', 'transformers==4.35.2'),
        ('torch', 'torch==2.1.0'),
        ('sklearn', 'scikit-learn==1.3.2'),
        ('textstat', 'textstat==0.7.3'),
        ('nltk', 'nltk==3.8.1'),
    ]
    
    missing_core = []
    missing_ml = []
    
    # Check core dependencies
    for module, package in core_deps:
        if module == 'tkinter':
            try:
                import tkinter
                print("✅ tkinter - Available")
            except ImportError:
                print("❌ tkinter - Missing (required for GUI)")
                print("Install: sudo apt-get install python3-tk (Linux)")
                return False
        else:
            spec = importlib.util.find_spec(module)
            if spec is None:
                missing_core.append(package)
            else:
                print(f"✅ {module} - Available")
    
    # Check ML dependencies
    for module, package in ml_deps:
        spec = importlib.util.find_spec(module)
        if spec is None:
            missing_ml.append(package)
        else:
            print(f"✅ {module} - Available")
    
    # Install missing core dependencies
    if missing_core:
        print(f"\n📦 Installing core dependencies...")
        for package in missing_core:
            if not install_package(package):
                print(f"❌ Failed to install {package}")
                return False
    
    # Install missing ML dependencies
    if missing_ml:
        print(f"\n🧠 Installing AI dependencies (this may take a while)...")
        print("Note: ML models will be downloaded on first use (~1-2GB)")
        
        for package in missing_ml:
            print(f"Installing {package.split('==')[0]}...")
            if not install_package(package):
                print(f"⚠️ Failed to install {package}")
                print("AI features may not work properly")
    
    if not missing_core and not missing_ml:
        print("✅ All dependencies satisfied!")
    
    return True

def launch_application():
    """Launch the AI-enhanced PDF reader"""
    try:
        print("🚀 Starting AI-Enhanced PDF Reader...")
        from ai_pdf_reader import main
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure ai_pdf_reader.py and ml_features.py are in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def show_usage_tips():
    """Show usage tips and information"""
    print("\n💡 Usage Tips:")
    print("1. Load a PDF file using the Browse button")
    print("2. Click 'Analyze' to run AI analysis")
    print("3. Use 'Summarize' for quick document overview")
    print("4. Try 'Key Points' for main highlights")
    print("5. Use 'Optimize Voice' for best reading settings")
    print("6. Select reading mode: Full Text, Summary, or Key Points")
    print("\n⚠️ Note: First run may be slow as AI models download")
    print("📊 Estimated download: 1-2GB for full AI features")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Check and install dependencies
    if not check_dependencies():
        print("\n❌ Dependency installation failed")
        print("Try running: pip install pypdf pyttsx3 transformers torch scikit-learn textstat nltk")
        input("Press Enter to exit...")
        return
    
    # Show usage tips
    show_usage_tips()
    
    print("\n" + "=" * 50)
    input("Press Enter to launch the application...")
    
    # Launch application
    if not launch_application():
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 