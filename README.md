# 🤖 AI-Enhanced PDF Text-to-Speech Reader

An intelligent PDF reader that converts documents to speech with advanced AI analysis features. Perfect for listening to PDFs on walks, during commutes, or while multitasking.

## ✨ Features

### 📖 Core Functionality
- **PDF Text Extraction**: Reads and processes PDF documents
- **High-Quality Text-to-Speech**: Natural voice synthesis with customizable settings
- **Audio Export**: Save TTS audio as WAV files for offline listening
- **Multiple Reading Modes**: Full text, summary, or key points

### 🧠 AI-Powered Analysis
- **Document Summarization**: Intelligent text summarization using BART models
- **Sentiment Analysis**: Understand document tone and emotion
- **Document Classification**: Automatic categorization of content type
- **Key Point Extraction**: Identify and highlight important information
- **Readability Analysis**: Assess document difficulty and suggest optimal reading speed
- **Smart Voice Optimization**: AI-recommended speed and volume settings

### 🎵 Audio Features
- **Playback Controls**: Play, pause, stop, and resume functionality
- **Adjustable Speed**: 80-300 words per minute
- **Volume Control**: 0-100% volume adjustment
- **Voice Selection**: System voices + premium Azure neural voices
- **☁️ Azure Integration**: High-quality AI voices (5 hours free/month)
- **Export to WAV**: Save audio files for portable listening
- **Multiple Content Modes**: Export full text, summaries, or key points

## 🚀 Quick Start

### Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   python install_dependencies.py
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python run_ai_app.py
   ```

### Usage

1. **Load PDF**: Click "Browse" to select your PDF file
2. **AI Analysis**: Use the AI buttons to analyze your document:
   - 🔍 **Analyze**: Get sentiment, classification, and readability analysis
   - 📄 **Summarize**: Generate a concise summary
   - 🎯 **Key Points**: Extract main points and highlights
   - ⚙️ **Optimize Voice**: Apply AI-recommended voice settings

3. **Premium Voices** (Optional): Click "☁️ Azure Voices" to set up premium neural voices
4. **Listen**: Choose your reading mode and click "Play"
5. **Export**: Click "💾 Export MP3" to save audio for offline listening

## 📁 Project Structure

```
pdf/
├── ai_pdf_reader.py              # Main application with GUI
├── ml_features_optimized.py      # Optimized AI/ML features
├── run_ai_app.py                 # Application launcher
├── azure_tts.py                 # Azure neural voice integration
├── voice_config.py               # Voice configuration helpers
├── install_dependencies.py       # Dependency installer
├── requirements.txt              # Python dependencies
├── AZURE_SETUP.md               # Azure setup guide
├── README.md                     # This file
├── cache/                        # AI model cache (auto-created)
└── *.pdf                        # Your PDF files
```

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended for AI features)
- **Storage**: 2GB free space for AI models
- **Audio**: Speakers or headphones for playback

## 📦 Dependencies

### Core Libraries
- `pypdf` - PDF text extraction
- `pyttsx3` - Text-to-speech synthesis
- `tkinter` - GUI framework (included with Python)

### AI/ML Libraries
- `transformers` - Hugging Face transformers for NLP
- `torch` - PyTorch for deep learning models
- `scikit-learn` - Machine learning utilities
- `textstat` - Readability analysis
- `nltk` - Natural language processing

## 🎯 Use Cases

- **Students**: Listen to research papers and textbooks while commuting
- **Professionals**: Review documents during walks or exercise
- **Accessibility**: Assist users with visual impairments or reading difficulties
- **Multitasking**: Consume written content while doing other activities
- **Learning**: Improve comprehension through audio-visual learning

## 🔊 Audio Export Features

The app can export audio in the following modes:
- **Full Text**: Complete document narration
- **Summary**: Condensed version highlighting main points
- **Key Points**: Bulleted list of important information

Exported files are saved as high-quality WAV files that can be:
- Transferred to mobile devices
- Played in any audio player
- Used offline during walks, commutes, or workouts

## ☁️ Premium Azure Neural Voices

Get studio-quality AI voices with Azure integration:

### 🎵 **Voice Quality**
- **System Voices (🔊)**: Good quality, always free
- **Azure Neural (☁️)**: Excellent quality, 5 hours free/month

### 🚀 **Setup** (Optional)
1. Click "☁️ Azure Voices" in the app
2. Follow the setup wizard
3. Get 5 hours of premium voices free every month
4. See [AZURE_SETUP.md](AZURE_SETUP.md) for detailed instructions

### 🌟 **Available Voices**
- **Jenny (US)** - Natural, friendly female voice ⭐ Recommended
- **Guy (US)** - Professional male voice
- **Sonia (UK)** - Elegant British female
- **Ryan (UK)** - Professional British male
- **Natasha (AU)** - Australian female
- And many more!

## 🧠 AI Model Information

The application uses optimized, lightweight AI models:
- **Summarization**: DistilBART for fast, quality summaries
- **Sentiment**: RoBERTa for accurate emotion detection
- **Classification**: Zero-shot classification for content categorization
- **Caching**: Results are cached to improve performance on repeat use

## 🛠️ Troubleshooting

### Common Issues

**"AI models still loading"**
- Wait for models to download (first run only)
- Check internet connection
- Ensure sufficient disk space

**"No audio output"**
- Check system audio settings
- Verify speakers/headphones are connected
- Try adjusting volume in the app

**"Export failed"**
- Check file permissions in save location
- Ensure sufficient disk space
- Try saving to a different location

**Performance Issues**
- Close other applications to free memory
- Use shorter text segments
- Clear cache folder if needed

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

**Enjoy your AI-enhanced PDF listening experience! 🎧📚** 