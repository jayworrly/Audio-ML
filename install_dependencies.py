#!/usr/bin/env python3
"""
Dependency installer for PDF Text-to-Speech application.
This script handles installation issues with Python 3.13 and newer versions.
"""

import sys
import subprocess
import importlib.util
import os

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"Running: {command}")
    if description:
        print(f"Purpose: {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        if package_name == 'tkinter':
            import tkinter
        else:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies with multiple fallback methods"""
    print("PDF Text-to-Speech Dependency Installer")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Dependencies to install
    dependencies = [
        ('pypdf', 'pypdf==4.0.1'),
        ('pyttsx3', 'pyttsx3==2.90'),
        ('transformers', 'transformers==4.35.2'),
        ('torch', 'torch==2.1.0'),
        ('scikit-learn', 'scikit-learn==1.3.2'),
        ('textstat', 'textstat==0.7.3'),
        ('nltk', 'nltk==3.8.1'),
    ]
    
    # Check tkinter separately
    if not check_package('tkinter'):
        print("\n⚠️  tkinter is not available!")
        if sys.platform.startswith('win'):
            print("On Windows, tkinter should be included with Python.")
            print("Try reinstalling Python with the 'Add to PATH' option.")
        elif sys.platform.startswith('darwin'):
            print("On macOS, try: brew install python-tk")
        else:
            print("On Linux, try: sudo apt-get install python3-tk")
        return False
    else:
        print("✓ tkinter is available")
    
    # Try to install each dependency
    failed_packages = []
    
    for package_name, package_spec in dependencies:
        print(f"\n--- Installing {package_name} ---")
        
        # Check if already installed
        if check_package(package_name):
            print(f"✓ {package_name} is already installed")
            continue
        
        # Try different installation methods
        install_methods = [
            f"pip install {package_spec}",
            f"pip install --user {package_spec}",
            f"pip install --no-cache-dir {package_spec}",
            f"pip install --upgrade pip && pip install {package_spec}",
            f"python -m pip install {package_spec}",
        ]
        
        installed = False
        for method in install_methods:
            print(f"\nTrying method: {method}")
            if run_command(method, f"Install {package_name}"):
                if check_package(package_name):
                    print(f"✓ {package_name} successfully installed!")
                    installed = True
                    break
                else:
                    print(f"Command succeeded but {package_name} still not importable")
        
        if not installed:
            failed_packages.append(package_name)
            print(f"✗ Failed to install {package_name}")
    
    # Summary
    print("\n" + "=" * 50)
    if failed_packages:
        print("❌ Installation completed with errors")
        print(f"Failed packages: {', '.join(failed_packages)}")
        print("\nManual installation commands:")
        for package_name, package_spec in dependencies:
            if package_name in failed_packages:
                print(f"  pip install {package_spec}")
        
        print(f"\nAlternative commands to try:")
        print(f"  python -m pip install --upgrade pip")
        print(f"  python -m pip install --user pypdf pyttsx3")
        return False
    else:
        print("✅ All dependencies installed successfully!")
        return True

def test_import():
    """Test if all modules can be imported"""
    print("\n--- Testing imports ---")
    
    modules_to_test = [
        ('tkinter', 'tkinter'),
        ('pypdf', 'pypdf'),
        ('pyttsx3', 'pyttsx3'),
    ]
    
    all_good = True
    for module_name, import_name in modules_to_test:
        try:
            if import_name == 'tkinter':
                import tkinter
            elif import_name == 'pypdf':
                import pypdf
            elif import_name == 'pyttsx3':
                import pyttsx3
            print(f"✓ {module_name} imports successfully")
        except ImportError as e:
            print(f"✗ {module_name} import failed: {e}")
            all_good = False
    
    return all_good

def main():
    """Main function"""
    success = install_dependencies()
    
    if success:
        print("\n--- Testing installation ---")
        if test_import():
            print("\n🎉 Setup complete! You can now run:")
            print("   python pdf_text_to_speech.py")
        else:
            print("\n⚠️  Some imports failed. Check the errors above.")
    else:
        print("\n❌ Installation failed. Please try manual installation.")
        print("\nIf you continue having issues, try:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Try with --user flag: pip install --user pypdf pyttsx3")
        print("3. Use conda instead: conda install pypdf pyttsx3")

if __name__ == "__main__":
    main() 