#!/usr/bin/env python3
"""
AI-Enhanced PDF Text-to-Speech Reader
Advanced PDF reader with machine learning features including:
- Document analysis and classification
- Sentiment analysis
- Text summarization
- Key point extraction
- Smart voice settings based on content
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pypdf
import pyttsx3
import re
import os
import tempfile
import wave
import subprocess
import sys
from ml_features_optimized import ml_features_optimized

class AIEnhancedPDFReader:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI-Enhanced PDF Text-to-Speech Reader")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize components
        self.tts_engine = pyttsx3.init()
        self.is_speaking = False
        self.is_paused = False
        self.current_text = ""
        self.summary = ""
        self.key_points = []
        self.document_analysis = {}
        self.chapters = []
        self.selected_chapter = None
        
        # Setup TTS
        self.setup_tts()
        
        # Create GUI
        self.create_interface()
        
        # Start ML status monitoring
        self.update_ml_status()
        
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)
    
    def setup_tts(self):
        """Setup text-to-speech engine with better voice selection"""
        self.available_voices = []
        voices = self.tts_engine.getProperty('voices')
        
        if voices:
            # Store available voices for selection
            for voice in voices:
                voice_info = {
                    'id': getattr(voice, 'id', ''),
                    'name': getattr(voice, 'name', 'Unknown Voice'),
                    'languages': getattr(voice, 'languages', []) or [],
                    'gender': getattr(voice, 'gender', 'Unknown') or 'Unknown'
                }
                self.available_voices.append(voice_info)
            
            # Try to find a better default voice (prefer female, English)
            best_voice = self.find_best_voice()
            self.tts_engine.setProperty('voice', best_voice)
            
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.8)
    
    def find_best_voice(self):
        """Find the best available voice (prefer female English voices)"""
        if not self.available_voices:
            return None
            
        # Preference order: female English > male English > any English > any voice
        preferences = [
            lambda v: 'female' in str(v.get('gender', '')).lower() and any('en' in str(lang).lower() for lang in v.get('languages', [])),
            lambda v: 'male' in str(v.get('gender', '')).lower() and any('en' in str(lang).lower() for lang in v.get('languages', [])),
            lambda v: any('en' in str(lang).lower() for lang in v.get('languages', [])),
            lambda v: 'zira' in str(v.get('name', '')).lower(),  # Windows default female voice
            lambda v: 'hazel' in str(v.get('name', '')).lower(),  # Another good Windows voice
            lambda v: True  # Fallback to any voice
        ]
        
        for preference in preferences:
            matching_voices = [v for v in self.available_voices if preference(v)]
            if matching_voices:
                return matching_voices[0]['id']
        
        return self.available_voices[0]['id']
    
    def create_interface(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # File selection
        self.create_file_selection(main_frame)
        
        # AI features
        self.create_ai_panel(main_frame)
        
        # Playback controls
        self.create_playback_controls(main_frame)
        
        # Text display
        self.create_text_display(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please select a PDF file")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_file_selection(self, parent):
        """Create file selection section"""
        file_frame = ttk.LabelFrame(parent, text="📁 PDF File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.file_path_var, 
                 state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_file).grid(row=0, column=2)
    
    def create_ai_panel(self, parent):
        """Create AI analysis panel"""
        ai_frame = ttk.LabelFrame(parent, text="🧠 AI Analysis", padding="5")
        ai_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ai_frame.columnconfigure(0, weight=1)
        
        # Status
        self.ml_status_var = tk.StringVar(value="Loading AI models...")
        ttk.Label(ai_frame, textvariable=self.ml_status_var).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Buttons
        btn_frame = ttk.Frame(ai_frame)
        btn_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.ai_buttons = {
            'analyze': ttk.Button(btn_frame, text="🔍 Analyze", 
                                command=self.run_analysis, state='disabled'),
            'summarize': ttk.Button(btn_frame, text="📄 Summarize", 
                                  command=self.generate_summary, state='disabled'),
            'keypoints': ttk.Button(btn_frame, text="🎯 Key Points", 
                                  command=self.extract_keypoints, state='disabled'),
            'chapters': ttk.Button(btn_frame, text="📚 Chapters", 
                                 command=self.detect_chapters, state='disabled'),
            'optimize': ttk.Button(btn_frame, text="⚙️ Optimize Voice", 
                                 command=self.optimize_voice, state='disabled')
        }
        
        for i, (key, btn) in enumerate(self.ai_buttons.items()):
            btn.grid(row=0, column=i, padx=(0, 5))
        
        # Results display
        self.analysis_display = tk.Text(ai_frame, height=4, wrap=tk.WORD)
        self.analysis_display.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def create_playback_controls(self, parent):
        """Create playback controls"""
        control_frame = ttk.LabelFrame(parent, text="🎵 Playback", padding="5")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # Playback buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=0, column=0, sticky=tk.W)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", 
                                  command=self.play_audio, state='disabled')
        self.play_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", 
                                   command=self.pause_audio, state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", 
                                  command=self.stop_audio, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Export button
        self.export_btn = ttk.Button(btn_frame, text="💾 Export MP3", 
                                    command=self.export_audio, state='disabled')
        self.export_btn.grid(row=0, column=3, padx=(0, 15))
        
        # Reading mode
        ttk.Label(btn_frame, text="Mode:").grid(row=0, column=4, padx=(0, 5))
        self.reading_mode = tk.StringVar(value="full text")
        mode_combo = ttk.Combobox(btn_frame, textvariable=self.reading_mode,
                                 values=["full text", "summary", "key points", "selected chapter"],
                                 state="readonly", width=15)
        mode_combo.grid(row=0, column=5)
        
        # Voice settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Speed
        ttk.Label(settings_frame, text="Speed:").grid(row=0, column=0, padx=(0, 5))
        self.speed_var = tk.IntVar(value=180)
        speed_scale = ttk.Scale(settings_frame, from_=80, to=300, 
                               variable=self.speed_var, orient=tk.HORIZONTAL,
                               length=150, command=self.update_speed)
        speed_scale.grid(row=0, column=1, padx=(0, 10))
        self.speed_label = ttk.Label(settings_frame, text="180 WPM")
        self.speed_label.grid(row=0, column=2, padx=(0, 20))
        
        # Volume
        ttk.Label(settings_frame, text="Volume:").grid(row=0, column=3, padx=(0, 5))
        self.volume_var = tk.DoubleVar(value=0.8)
        volume_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0,
                                variable=self.volume_var, orient=tk.HORIZONTAL,
                                length=150, command=self.update_volume)
        volume_scale.grid(row=0, column=4, padx=(0, 10))
        self.volume_label = ttk.Label(settings_frame, text="80%")
        self.volume_label.grid(row=0, column=5)
        
        # Voice selection (second row)
        voice_frame = ttk.Frame(control_frame)
        voice_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(voice_frame, text="Voice:").grid(row=0, column=0, padx=(0, 5))
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(voice_frame, textvariable=self.voice_var,
                                       state="readonly", width=30)
        self.voice_combo.grid(row=0, column=1, padx=(0, 10))
        self.voice_combo.bind('<<ComboboxSelected>>', self.change_voice)
        
        # Populate voice dropdown
        self.populate_voice_dropdown()
        
        # Voice preview button
        self.preview_btn = ttk.Button(voice_frame, text="🔊 Preview", 
                                     command=self.preview_voice)
        self.preview_btn.grid(row=0, column=2, padx=(5, 0))
    
    def create_text_display(self, parent):
        """Create text display with tabs"""
        text_frame = ttk.LabelFrame(parent, text="📖 Content", padding="5")
        text_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(text_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Full text tab
        full_frame = ttk.Frame(self.notebook)
        self.notebook.add(full_frame, text="📄 Full Text")
        self.full_text_widget = scrolledtext.ScrolledText(full_frame, wrap=tk.WORD)
        self.full_text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        full_frame.columnconfigure(0, weight=1)
        full_frame.rowconfigure(0, weight=1)
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📝 Summary")
        self.summary_widget = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD)
        self.summary_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        
        # Key points tab
        keypoints_frame = ttk.Frame(self.notebook)
        self.notebook.add(keypoints_frame, text="🎯 Key Points")
        self.keypoints_widget = scrolledtext.ScrolledText(keypoints_frame, wrap=tk.WORD)
        self.keypoints_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        keypoints_frame.columnconfigure(0, weight=1)
        keypoints_frame.rowconfigure(0, weight=1)
        
        # Chapters tab
        chapters_frame = ttk.Frame(self.notebook)
        self.notebook.add(chapters_frame, text="📚 Chapters")
        chapters_frame.columnconfigure(0, weight=1)
        chapters_frame.rowconfigure(1, weight=1)
        
        # Chapter controls
        chapter_controls = ttk.Frame(chapters_frame)
        chapter_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        chapter_controls.columnconfigure(1, weight=1)
        
        ttk.Label(chapter_controls, text="Select Chapter:").grid(row=0, column=0, padx=(0, 5))
        self.chapter_var = tk.StringVar()
        self.chapter_combo = ttk.Combobox(chapter_controls, textvariable=self.chapter_var,
                                         state="readonly", width=50)
        self.chapter_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.chapter_combo.bind('<<ComboboxSelected>>', self.on_chapter_selected)
        
        # Chapter export button
        self.export_chapter_btn = ttk.Button(chapter_controls, text="💾 Export Chapter", 
                                           command=self.export_chapter, state='disabled')
        self.export_chapter_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Chapter text display
        self.chapter_widget = scrolledtext.ScrolledText(chapters_frame, wrap=tk.WORD)
        self.chapter_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    
    def update_ml_status(self):
        """Update ML model loading status"""
        status = ml_features_optimized.get_load_status()
        self.ml_status_var.set(f"AI: {status}")
        
        if ml_features_optimized.models_loaded:
            for btn in self.ai_buttons.values():
                btn.config(state='normal')
        else:
            self.root.after(2000, self.update_ml_status)
    
    def browse_file(self):
        """Browse and load PDF file"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.load_pdf(file_path)
    
    def load_pdf(self, file_path):
        """Load and extract text from PDF"""
        try:
            self.status_var.set("Loading PDF...")
            self.clear_displays()
            
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"--- Page {page_num + 1} ---\n"
                            text_content += page_text + "\n\n"
                    except Exception:
                        text_content += f"--- Page {page_num + 1} (Error) ---\n\n"
            
            if text_content.strip():
                self.current_text = self.clean_text(text_content)
                self.full_text_widget.insert(1.0, self.current_text)
                self.play_btn.config(state='normal')
                self.status_var.set(f"Loaded {len(pdf_reader.pages)} pages successfully")
                self.reset_analysis()
            else:
                self.status_var.set("No readable text found")
                messagebox.showwarning("Warning", 
                    "No readable text found. PDF might be image-based or encrypted.")
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    
    def clean_text(self, text):
        """Clean extracted text for better TTS"""
        # Remove page markers for cleaner reading
        text = re.sub(r'--- Page \d+ ---\n', '', text)
        # Normalize whitespace but preserve line breaks for chapter detection
        text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces and tabs, not newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines to double newlines
        # Fix sentence boundaries
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
        return text.strip()
    
    def clear_displays(self):
        """Clear all text displays"""
        self.full_text_widget.delete(1.0, tk.END)
        self.summary_widget.delete(1.0, tk.END)
        self.keypoints_widget.delete(1.0, tk.END)
        self.chapter_widget.delete(1.0, tk.END)
        self.analysis_display.delete(1.0, tk.END)
    
    def reset_analysis(self):
        """Reset analysis data"""
        self.document_analysis = {}
        self.summary = ""
        self.key_points = []
        self.chapters = []
        self.selected_chapter = None
        # Reset chapter UI
        self.chapter_combo.config(values=[])
        self.export_chapter_btn.config(state='disabled')
    
    def run_analysis(self):
        """Run comprehensive document analysis"""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a PDF first.")
            return
        
        if not ml_features_optimized.models_loaded:
            messagebox.showwarning("Warning", "AI models still loading. Please wait.")
            return
        
        self.status_var.set("Analyzing document...")
        
        def analyze():
            try:
                # Run all analyses
                sentiment = ml_features_optimized.analyze_sentiment(self.current_text)
                classification = ml_features_optimized.classify_document(self.current_text)
                readability = ml_features_optimized.analyze_readability(self.current_text)
                
                self.document_analysis = {
                    'sentiment': sentiment,
                    'classification': classification,
                    'readability': readability
                }
                
                self.root.after(0, self.display_analysis)
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Analysis error: {str(e)}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_analysis(self):
        """Display analysis results"""
        if not self.document_analysis:
            return
        
        self.analysis_display.delete(1.0, tk.END)
        
        results = []
        sentiment = self.document_analysis['sentiment']
        classification = self.document_analysis['classification']
        readability = self.document_analysis['readability']
        
        results.append(f"📊 Sentiment: {sentiment['sentiment'].title()} "
                      f"({sentiment['confidence']:.1%})")
        results.append(f"📋 Type: {classification['category'].title()} "
                      f"({classification['confidence']:.1%})")
        results.append(f"📚 Difficulty: {readability['difficulty'].title()} "
                      f"(Grade {readability['grade_level']:.1f})")
        results.append(f"⚡ Suggested Speed: {readability['recommended_speed']} WPM")
        
        self.analysis_display.insert(1.0, '\n'.join(results))
        self.status_var.set("Analysis complete")
    
    def generate_summary(self):
        """Generate document summary"""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a PDF first.")
            return
        
        if not ml_features_optimized.models_loaded:
            messagebox.showwarning("Warning", "AI models still loading.")
            return
        
        self.status_var.set("Generating summary...")
        
        def summarize():
            try:
                summary = ml_features_optimized.summarize_text(self.current_text, max_length=200)
                self.summary = summary
                
                self.root.after(0, lambda: self.summary_widget.delete(1.0, tk.END))
                self.root.after(0, lambda: self.summary_widget.insert(1.0, summary))
                self.root.after(0, lambda: self.notebook.select(1))
                self.root.after(0, lambda: self.status_var.set("Summary generated"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Summary error: {str(e)}"))
        
        threading.Thread(target=summarize, daemon=True).start()
    
    def extract_keypoints(self):
        """Extract key points from document"""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a PDF first.")
            return
        
        if not ml_features_optimized.models_loaded:
            messagebox.showwarning("Warning", "AI models still loading.")
            return
        
        self.status_var.set("Extracting key points...")
        
        def extract():
            try:
                points = ml_features_optimized.extract_key_points(self.current_text, num_points=7)
                self.key_points = points
                
                formatted = '\n\n'.join([f"{i+1}. {point}" 
                                       for i, point in enumerate(points)])
                
                self.root.after(0, lambda: self.keypoints_widget.delete(1.0, tk.END))
                self.root.after(0, lambda: self.keypoints_widget.insert(1.0, formatted))
                self.root.after(0, lambda: self.notebook.select(2))
                self.root.after(0, lambda: self.status_var.set("Key points extracted"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Extraction error: {str(e)}"))
        
        threading.Thread(target=extract, daemon=True).start()
    
    def detect_chapters(self):
        """Detect and display chapters"""
        if not self.current_text:
            messagebox.showwarning("Warning", "Please load a PDF first.")
            return
        
        if not ml_features_optimized.models_loaded:
            messagebox.showwarning("Warning", "AI models still loading.")
            return
        
        self.status_var.set("Detecting chapters...")
        
        def detect():
            try:
                print("Starting chapter detection...")
                chapters = ml_features_optimized.detect_chapters(self.current_text)
                print(f"Chapter detection complete. Found {len(chapters)} chapters")
                self.chapters = chapters
                
                if not chapters:
                    self.root.after(0, lambda: self.status_var.set("No chapters detected"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Chapter Detection", "No chapters could be detected in this document."))
                    return
                
                # Update chapter dropdown
                chapter_options = []
                for i, ch in enumerate(chapters):
                    title = ch["title"]
                    pages = ch.get("page_estimate", 1)
                    chapter_options.append(f"{i+1}. {title} ({pages}p)")
                
                print(f"Created {len(chapter_options)} chapter options")
                
                self.root.after(0, lambda: self.chapter_combo.config(values=chapter_options))
                self.root.after(0, lambda: self.export_chapter_btn.config(state='normal'))
                self.root.after(0, lambda: self.notebook.select(3))  # Switch to chapters tab
                self.root.after(0, lambda: self.status_var.set(f"Found {len(chapters)} chapters"))
                
                # Show chapter summary
                summary_text = f"📚 CHAPTER BREAKDOWN ({len(chapters)} chapters found)\n\n"
                for i, ch in enumerate(chapters):
                    summary_text += f"{i+1}. {ch['title']}\n"
                    summary_text += f"   Type: {ch.get('type', 'Unknown')}\n"
                    summary_text += f"   Pages: ~{ch.get('page_estimate', 1)}\n"
                    summary_text += f"   Words: {ch.get('word_count', 0):,}\n\n"
                
                self.root.after(0, lambda: self.chapter_widget.delete(1.0, tk.END))
                self.root.after(0, lambda: self.chapter_widget.insert(1.0, summary_text))
                
            except Exception as e:
                print(f"Chapter detection error: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.status_var.set(f"Chapter detection error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Chapter Detection Error", f"Failed to detect chapters:\n{str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def on_chapter_selected(self, event=None):
        """Handle chapter selection"""
        selection = self.chapter_combo.get()
        if not selection:
            return
        
        try:
            # Extract chapter index from selection
            chapter_index = int(selection.split('.')[0]) - 1
            if 0 <= chapter_index < len(self.chapters):
                self.selected_chapter = chapter_index
                chapter = self.chapters[chapter_index]
                
                # Display chapter content
                chapter_text = f"📖 {chapter['title']}\n"
                chapter_text += f"Type: {chapter.get('type', 'Unknown')} | "
                chapter_text += f"Pages: ~{chapter.get('page_estimate', 1)} | "
                chapter_text += f"Words: {chapter.get('word_count', 0):,}\n"
                chapter_text += "=" * 80 + "\n\n"
                chapter_text += chapter['content']
                
                self.chapter_widget.delete(1.0, tk.END)
                self.chapter_widget.insert(1.0, chapter_text)
                
                self.status_var.set(f"Selected: {chapter['title']}")
                
        except (ValueError, IndexError):
            pass
    
    def export_chapter(self):
        """Export selected chapter as audio"""
        if self.selected_chapter is None:
            messagebox.showwarning("Warning", "Please select a chapter first.")
            return
        
        if not (0 <= self.selected_chapter < len(self.chapters)):
            messagebox.showwarning("Warning", "Invalid chapter selection.")
            return
        
        chapter = self.chapters[self.selected_chapter]
        chapter_text = chapter['content']
        
        if not chapter_text.strip():
            messagebox.showwarning("Warning", "Selected chapter has no content.")
            return
        
        # Get filename from user
        safe_title = "".join(c for c in chapter['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
        default_name = f"Chapter_{self.selected_chapter+1}_{safe_title[:30]}.wav"
        
        file_path = filedialog.asksaveasfilename(
            title="Export Chapter Audio",
            defaultextension=".wav",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ],
            initialfile=default_name
        )
        
        if not file_path:
            return
        
        self.status_var.set(f"Exporting chapter: {chapter['title']}...")
        
        def export():
            try:
                # Method 1: Try using Windows PowerShell TTS (most reliable)
                if sys.platform == "win32":
                    success = self.export_chapter_with_powershell(chapter_text, file_path, chapter['title'])
                    if success:
                        return
                
                # Method 2: Try using espeak if available
                success = self.export_with_espeak(chapter_text, file_path)
                if success:
                    return
                
                # Method 3: Fallback - save as text file with instructions
                self.export_chapter_as_text_fallback(chapter_text, file_path, chapter)
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Export error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Export Error", f"Failed to export chapter:\n{str(e)}"))
        
        threading.Thread(target=export, daemon=True).start()
    
    def export_chapter_with_powershell(self, text, file_path, chapter_title):
        """Export chapter using Windows PowerShell TTS"""
        try:
            # Create PowerShell script for TTS
            ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("{file_path}")
$synth.Rate = {self.speed_var.get() // 10 - 10}
$synth.Volume = {int(self.volume_var.get() * 100)}
$synth.Speak(@"
{text.replace('"', '""')}
"@)
$synth.Dispose()
'''
            
            # Save script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as script_file:
                script_file.write(ps_script)
                script_path = script_file.name
            
            # Execute PowerShell script
            result = subprocess.run([
                'powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path
            ], capture_output=True, text=True, timeout=600)  # Longer timeout for chapters
            
            # Clean up script file
            os.remove(script_path)
            
            if result.returncode == 0 and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                self.root.after(0, lambda: self.status_var.set(
                    f"Chapter exported: {os.path.basename(file_path)} ({size_mb:.1f} MB)"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Chapter Export Complete", 
                    f"Chapter audio saved successfully!\n\n" +
                    f"Chapter: {chapter_title}\n" +
                    f"File: {file_path}\n" +
                    f"Size: {size_mb:.1f} MB\n\n" +
                    f"Perfect for listening on your walks!"))
                return True
            else:
                return False
                
        except Exception as e:
            print(f"PowerShell chapter export failed: {e}")
            return False
    
    def export_chapter_as_text_fallback(self, text, file_path, chapter):
        """Fallback: save chapter as text file with TTS instructions"""
        try:
            # Change extension to .txt
            base_path = os.path.splitext(file_path)[0]
            text_path = base_path + "_chapter_script.txt"
            
            # Save text with instructions
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("PDF CHAPTER TEXT-TO-SPEECH EXPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"CHAPTER: {chapter['title']}\n")
                f.write(f"TYPE: {chapter.get('type', 'Unknown')}\n")
                f.write(f"PAGES: ~{chapter.get('page_estimate', 1)}\n")
                f.write(f"WORDS: {chapter.get('word_count', 0):,}\n\n")
                f.write("INSTRUCTIONS:\n")
                f.write("This text can be converted to audio using:\n")
                f.write("1. Windows Narrator (built-in)\n")
                f.write("2. Online TTS services (naturalreaders.com, etc.)\n")
                f.write("3. Mobile TTS apps\n\n")
                f.write("SETTINGS:\n")
                f.write(f"Speed: {self.speed_var.get()} WPM\n")
                f.write(f"Volume: {int(self.volume_var.get() * 100)}%\n\n")
                f.write("CHAPTER CONTENT:\n")
                f.write("-" * 50 + "\n\n")
                f.write(text)
            
            self.root.after(0, lambda: self.status_var.set(f"Chapter text exported: {os.path.basename(text_path)}"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Chapter Export Alternative", 
                f"Audio export not available, but chapter saved as:\n{text_path}\n\n" +
                f"Chapter: {chapter['title']}\n\n" +
                "You can use this with:\n" +
                "• Windows Narrator\n" +
                "• Online TTS services\n" +
                "• Mobile TTS apps"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Export failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror(
                "Export Error", f"Failed to export chapter:\n{str(e)}"))
    
    def optimize_voice(self):
        """Apply AI-optimized voice settings"""
        if not self.document_analysis:
            messagebox.showinfo("Info", "Please analyze the document first.")
            return
        
        sentiment = self.document_analysis['sentiment']['sentiment']
        difficulty = self.document_analysis['readability']['difficulty']
        
        settings = ml_features_optimized
        
        self.speed_var.set(settings['speed'])
        self.volume_var.set(settings['volume'])
        
        self.update_speed(settings['speed'])
        self.update_volume(settings['volume'])
        
        self.status_var.set(f"Optimized: {settings['speed']} WPM, "
                           f"{int(settings['volume']*100)}% volume")
    
    def get_text_to_read(self):
        """Get text based on selected reading mode"""
        mode = self.reading_mode.get()
        
        if mode == "summary" and self.summary:
            return self.summary
        elif mode == "key points" and self.key_points:
            return '\n\n'.join([f"Point {i+1}: {point}" 
                              for i, point in enumerate(self.key_points)])
        elif mode == "selected chapter" and self.selected_chapter is not None and self.chapters:
            if 0 <= self.selected_chapter < len(self.chapters):
                return self.chapters[self.selected_chapter]['content']
            else:
                return ""
        else:
            return self.current_text
    
    def play_audio(self):
        """Start or resume audio playback"""
        text = self.get_text_to_read()
        if not text:
            messagebox.showwarning("Warning", "No text available for selected mode.")
            return
        
        if self.is_paused:
            self.is_paused = False
            self.status_var.set("Resuming...")
        
        self.is_speaking = True
        self.update_playback_buttons()
        
        def speak():
            try:
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if not self.is_speaking:
                        break
                    
                    sentence = sentence.strip()
                    if sentence:
                        self.root.after(0, lambda s=sentence: 
                                      self.status_var.set(f"Speaking: {s[:40]}..."))
                        self.tts_engine.say(sentence)
                        self.tts_engine.runAndWait()
                
                if self.is_speaking:
                    self.root.after(0, lambda: self.status_var.set("Playback complete"))
                    self.is_speaking = False
                    self.root.after(0, self.update_playback_buttons)
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Playback error: {str(e)}"))
                self.is_speaking = False
                self.root.after(0, self.update_playback_buttons)
        
        threading.Thread(target=speak, daemon=True).start()
    
    def pause_audio(self):
        """Pause audio playback"""
        if self.is_speaking:
            self.is_paused = True
            self.is_speaking = False
            self.tts_engine.stop()
            self.status_var.set("Paused")
            self.update_playback_buttons()
    
    def stop_audio(self):
        """Stop audio playback"""
        self.is_speaking = False
        self.is_paused = False
        self.tts_engine.stop()
        self.status_var.set("Stopped")
        self.update_playback_buttons()
    
    def update_playback_buttons(self):
        """Update playback button states"""
        text_available = bool(self.get_text_to_read())
        
        if self.is_speaking:
            self.play_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.stop_btn.config(state='normal')
        elif self.is_paused:
            self.play_btn.config(state='normal', text="▶ Resume")
            self.pause_btn.config(state='disabled')  
            self.stop_btn.config(state='normal')
        else:
            self.play_btn.config(state='normal' if text_available else 'disabled', 
                               text="▶ Play")
            self.pause_btn.config(state='disabled')
            self.stop_btn.config(state='disabled')
        
        # Export button is always available when text is available
        self.export_btn.config(state='normal' if text_available else 'disabled')
    
    def update_speed(self, value):
        """Update TTS speed"""
        speed = int(float(value))
        self.tts_engine.setProperty('rate', speed)
        self.speed_label.config(text=f"{speed} WPM")
    
    def update_volume(self, value):
        """Update TTS volume"""
        volume = float(value)
        self.tts_engine.setProperty('volume', volume)
        self.volume_label.config(text=f"{int(volume*100)}%")
    
    def populate_voice_dropdown(self):
        """Populate the voice selection dropdown"""
        if not hasattr(self, 'available_voices') or not self.available_voices:
            return
            
        voice_options = []
        current_voice_id = self.tts_engine.getProperty('voice')
        current_selection = 0
        
        for i, voice in enumerate(self.available_voices):
            # Create a nice display name
            name = str(voice.get('name', f'Voice {i+1}'))
            # Clean up the voice name
            if 'Microsoft' in name:
                name = name.replace('Microsoft ', '')
            if 'Desktop' in name:
                name = name.replace(' Desktop', '')
            
            # Add gender info if available
            gender = str(voice.get('gender', ''))
            if gender and gender != 'Unknown' and gender.lower() != 'none':
                name += f" ({gender})"
            
            voice_options.append(name)
            
            # Track current selection
            if voice['id'] == current_voice_id:
                current_selection = i
        
        self.voice_combo.config(values=voice_options)
        if voice_options:
            self.voice_combo.current(current_selection)
    
    def change_voice(self, event=None):
        """Change the TTS voice"""
        selection = self.voice_combo.current()
        if 0 <= selection < len(self.available_voices):
            voice_id = self.available_voices[selection]['id']
            self.tts_engine.setProperty('voice', voice_id)
            self.status_var.set(f"Voice changed to: {self.available_voices[selection]['name']}")
    
    def preview_voice(self):
        """Preview the current voice with a sample text"""
        if self.is_speaking:
            return
            
        sample_texts = [
            "Hello! I'm your AI-enhanced PDF reader. I can help you listen to documents with natural-sounding speech.",
            "This is a preview of how I sound when reading your PDF content. You can adjust my speed and volume as needed.",
            "I specialize in reading technical documents, research papers, and books with proper pronunciation and pacing."
        ]
        
        import random
        sample_text = random.choice(sample_texts)
        
        def preview_speak():
            try:
                self.is_speaking = True
                self.update_playback_buttons()
                self.tts_engine.say(sample_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Preview error: {e}")
            finally:
                self.is_speaking = False
                self.root.after(0, self.update_playback_buttons)
        
        threading.Thread(target=preview_speak, daemon=True).start()
    
    def export_audio(self):
        """Export current text as audio file using Windows TTS"""
        text = self.get_text_to_read()
        if not text:
            messagebox.showwarning("Warning", "No text available for selected mode.")
            return
        
        # Get filename from user
        mode = self.reading_mode.get().replace(" ", "_")
        default_name = f"pdf_audio_{mode}.wav"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Audio As",
            defaultextension=".wav",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ],
            initialfile=default_name
        )
        
        if not file_path:
            return
        
        self.status_var.set("Exporting audio...")
        
        def export():
            try:
                # Method 1: Try using Windows PowerShell TTS (most reliable)
                if sys.platform == "win32":
                    success = self.export_with_powershell(text, file_path)
                    if success:
                        return
                
                # Method 2: Try using espeak if available
                success = self.export_with_espeak(text, file_path)
                if success:
                    return
                
                # Method 3: Fallback - save as text file with instructions
                self.export_as_text_fallback(text, file_path)
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Export error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Export Error", f"Failed to export audio:\n{str(e)}"))
        
        threading.Thread(target=export, daemon=True).start()
    
    def export_with_powershell(self, text, file_path):
        """Export using Windows PowerShell TTS"""
        try:
            # Create PowerShell script for TTS
            ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("{file_path}")
$synth.Rate = {self.speed_var.get() // 10 - 10}
$synth.Volume = {int(self.volume_var.get() * 100)}
$synth.Speak(@"
{text.replace('"', '""')}
"@)
$synth.Dispose()
'''
            
            # Save script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as script_file:
                script_file.write(ps_script)
                script_path = script_file.name
            
            # Execute PowerShell script
            result = subprocess.run([
                'powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path
            ], capture_output=True, text=True, timeout=300)
            
            # Clean up script file
            os.remove(script_path)
            
            if result.returncode == 0 and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                self.root.after(0, lambda: self.status_var.set(
                    f"Audio exported: {os.path.basename(file_path)} ({size_mb:.1f} MB)"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Export Complete", 
                    f"Audio saved successfully!\n\nFile: {file_path}\nSize: {size_mb:.1f} MB\n\nYou can now listen to this on your walks!"))
                return True
            else:
                return False
                
        except Exception as e:
            print(f"PowerShell TTS failed: {e}")
            return False
    
    def export_with_espeak(self, text, file_path):
        """Export using espeak if available"""
        try:
            # Try espeak command
            result = subprocess.run([
                'espeak', '-s', str(self.speed_var.get()), 
                '-a', str(int(self.volume_var.get() * 200)),
                '-w', file_path, text
            ], capture_output=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                self.root.after(0, lambda: self.status_var.set(
                    f"Audio exported: {os.path.basename(file_path)} ({size_mb:.1f} MB)"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Export Complete", 
                    f"Audio saved successfully!\n\nFile: {file_path}\nSize: {size_mb:.1f} MB"))
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def export_as_text_fallback(self, text, file_path):
        """Fallback: save as text file with TTS instructions"""
        try:
            # Change extension to .txt
            base_path = os.path.splitext(file_path)[0]
            text_path = base_path + "_script.txt"
            
            # Save text with instructions
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("PDF TEXT-TO-SPEECH EXPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write("INSTRUCTIONS:\n")
                f.write("This text can be converted to audio using:\n")
                f.write("1. Windows Narrator (built-in)\n")
                f.write("2. Online TTS services (naturalreaders.com, etc.)\n")
                f.write("3. Mobile TTS apps\n\n")
                f.write("SETTINGS:\n")
                f.write(f"Speed: {self.speed_var.get()} WPM\n")
                f.write(f"Volume: {int(self.volume_var.get() * 100)}%\n\n")
                f.write("TEXT CONTENT:\n")
                f.write("-" * 50 + "\n\n")
                f.write(text)
            
            self.root.after(0, lambda: self.status_var.set(f"Text exported: {os.path.basename(text_path)}"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Export Alternative", 
                f"Audio export not available, but text saved as:\n{text_path}\n\n" +
                "You can use this with:\n" +
                "• Windows Narrator\n" +
                "• Online TTS services\n" +
                "• Mobile TTS apps"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Export failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror(
                "Export Error", f"Failed to export:\n{str(e)}"))
    
    def cleanup_and_exit(self):
        """Clean up and exit application"""
        if self.is_speaking:
            self.stop_audio()
        self.root.destroy()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = AIEnhancedPDFReader(root)
    root.mainloop()

if __name__ == "__main__":
    main() 