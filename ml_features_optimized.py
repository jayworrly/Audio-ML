#!/usr/bin/env python3
"""
Optimized Machine Learning Features for PDF Text-to-Speech Application
Performance-focused version with lazy loading and caching.
"""

import re
import warnings
import threading
from typing import List, Dict, Tuple, Optional
import logging
import pickle
import hashlib
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

class OptimizedMLFeatures:
    def __init__(self):
        self.models_loaded = False
        self.summarizer = None
        self.sentiment_analyzer = None
        self.classifier = None
        self.loading_thread = None
        self.load_status = "Initializing..."
        
        # Performance optimizations
        self.use_fast_models = True  # Use smaller, faster models
        self.cache_enabled = True    # Enable result caching
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model loading flags
        self.models_to_load = {
            'summarizer': True,   # Auto-load on startup
            'sentiment': True,    # Auto-load on startup
            'classifier': True    # Auto-load on startup
        }
        
        # Chapter detection
        self.chapters = []
        self.chapter_cache = {}
        
        # Start loading models automatically
        self.start_loading_models()
    
    def start_loading_models(self):
        """Start loading models in background thread"""
        if not self.loading_thread or not self.loading_thread.is_alive():
            self.loading_thread = threading.Thread(target=self._load_specific_models, daemon=True)
            self.loading_thread.start()
    
    def enable_fast_mode(self):
        """Enable fast mode with smaller models"""
        self.use_fast_models = True
        self.load_status = "Fast mode enabled"
    
    def load_model_on_demand(self, model_type: str):
        """Load specific model only when needed"""
        if model_type in self.models_to_load:
            self.models_to_load[model_type] = True
            if not self.loading_thread or not self.loading_thread.is_alive():
                self.loading_thread = threading.Thread(target=self._load_specific_models, daemon=True)
                self.loading_thread.start()
    
    def _load_specific_models(self):
        """Load only the models that are requested"""
        try:
            self.load_status = "Downloading AI models (first time only)..."
            from transformers import pipeline
            import nltk
            import textstat
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if self.models_to_load['sentiment'] and not self.sentiment_analyzer:
                self.load_status = "Downloading sentiment model (~500MB)..."
                if self.use_fast_models:
                    # Use smaller, faster model
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True,
                        device=-1  # Force CPU for consistency
                    )
                else:
                    self.sentiment_analyzer = pipeline("sentiment-analysis")
            
            if self.models_to_load['summarizer'] and not self.summarizer:
                self.load_status = "Downloading summarizer model (~250MB)..."
                if self.use_fast_models:
                    # Use smaller, faster model
                    self.summarizer = pipeline(
                        "summarization",
                        model="sshleifer/distilbart-cnn-6-6",  # Smaller than BART-large
                        max_length=100,
                        min_length=20,
                        device=-1
                    )
                else:
                    self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            if self.models_to_load['classifier'] and not self.classifier:
                self.load_status = "Downloading classifier model (~250MB)..."
                if self.use_fast_models:
                    # Use DistilBERT instead of BART for classification
                    self.classifier = pipeline(
                        "zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli",
                        device=-1
                    )
                else:
                    self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            
            # Download NLTK data if needed (small files)
            self.load_status = "Downloading language data..."
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                pass
            
            # Initialize other tools
            self.load_status = "Initializing AI features..."
            self.vectorizer = TfidfVectorizer(max_features=50, stop_words='english')  # Reduced features
            
            loaded_models = [k for k, v in self.models_to_load.items() if v]
            self.load_status = f"✅ AI models ready! ({', '.join(loaded_models)})"
            self.models_loaded = True
            
        except Exception as e:
            self.load_status = f"❌ Error loading models: {str(e)}"
            self.models_loaded = False
    
    def get_cache_key(self, text: str, operation: str) -> str:
        """Generate cache key for text and operation"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{operation}_{text_hash[:16]}"
    
    def get_cached_result(self, cache_key: str):
        """Get cached result if available"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def save_cached_result(self, cache_key: str, result):
        """Save result to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass
    
    def get_load_status(self):
        """Get current loading status"""
        return self.load_status
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Generate summary with optimizations"""
        # Check cache first
        cache_key = self.get_cache_key(text[:1000], f"summary_{max_length}")
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Load model on demand
        self.load_model_on_demand('summarizer')
        
        if not self.summarizer:
            return "Summarization model not loaded. Enable in settings."
        
        try:
            # Optimize text preprocessing
            text = self._fast_clean_text(text)
            
            # Limit text length for faster processing
            max_input_length = 2000 if self.use_fast_models else 4000
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=max(10, max_length // 4),
                do_sample=False
            )
            
            result = summary[0]['summary_text']
            
            # Cache result
            self.save_cached_result(cache_key, result)
            
            return result
            
        except Exception as e:
            return f"Summarization error: {str(e)}"
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Fast sentiment analysis"""
        # Check cache
        cache_key = self.get_cache_key(text[:500], "sentiment")
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Load model on demand
        self.load_model_on_demand('sentiment')
        
        if not self.sentiment_analyzer:
            return {"sentiment": "neutral", "confidence": 0.5, "explanation": "Model not loaded"}
        
        try:
            # Use only first 500 characters for speed
            sample_text = text[:500]
            
            result = self.sentiment_analyzer(sample_text)
            
            # Process result
            if isinstance(result[0], list):
                scores = result[0]
                best_score = max(scores, key=lambda x: x['score'])
                sentiment = best_score['label'].lower()
                confidence = best_score['score']
            else:
                sentiment = result[0]['label'].lower()
                confidence = result[0]['score']
            
            # Normalize sentiment labels
            if 'positive' in sentiment or 'pos' in sentiment:
                sentiment = 'positive'
            elif 'negative' in sentiment or 'neg' in sentiment:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            final_result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "explanation": f"Analysis based on first 500 characters",
                "details": {"processed_length": len(sample_text)}
            }
            
            # Cache result
            self.save_cached_result(cache_key, final_result)
            
            return final_result
            
        except Exception as e:
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Error: {str(e)}"}
    
    def classify_document(self, text: str) -> Dict[str, any]:
        """Fast document classification"""
        # Check cache
        cache_key = self.get_cache_key(text[:300], "classify")
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Load model on demand
        self.load_model_on_demand('classifier')
        
        if not self.classifier:
            return {"category": "document", "confidence": 0.5, "explanation": "Model not loaded"}
        
        try:
            # Simplified category list for faster processing
            candidate_labels = [
                "academic", "business", "news", "technical", "legal", "educational"
            ]
            
            # Use only first 300 characters
            sample_text = text[:300]
            
            result = self.classifier(sample_text, candidate_labels)
            
            final_result = {
                "category": result['labels'][0],
                "confidence": result['scores'][0],
                "explanation": f"Fast classification based on first 300 characters",
                "all_scores": dict(zip(result['labels'][:3], result['scores'][:3]))  # Top 3 only
            }
            
            # Cache result
            self.save_cached_result(cache_key, final_result)
            
            return final_result
            
        except Exception as e:
            return {"category": "document", "confidence": 0.0, "explanation": f"Error: {str(e)}"}
    
    def analyze_readability(self, text: str) -> Dict[str, any]:
        """Fast readability analysis (lightweight)"""
        try:
            import textstat
            
            # Use sample for large texts
            sample_text = text[:2000] if len(text) > 2000 else text
            
            # Calculate basic scores only
            flesch_score = textstat.flesch_reading_ease(sample_text)
            word_count = len(sample_text.split())
            sentence_count = sample_text.count('.') + sample_text.count('!') + sample_text.count('?')
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Simple grade level estimation
            grade_level = max(1, min(16, (avg_sentence_length - 15) / 2 + 8))
            
            # Determine difficulty and speed
            if flesch_score >= 80:
                difficulty = "easy"
                recommended_speed = 200
            elif flesch_score >= 60:
                difficulty = "standard"
                recommended_speed = 180
            elif flesch_score >= 40:
                difficulty = "difficult"
                recommended_speed = 160
            else:
                difficulty = "very difficult"
                recommended_speed = 140
            
            return {
                "flesch_score": flesch_score,
                "grade_level": grade_level,
                "difficulty": difficulty,
                "recommended_speed": recommended_speed,
                "word_count": word_count,
                "explanation": f"Fast analysis of {len(sample_text)} characters"
            }
            
        except Exception as e:
            return {
                "flesch_score": 60,
                "grade_level": 10,
                "difficulty": "standard",
                "recommended_speed": 180,
                "explanation": f"Error: {str(e)}"
            }
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Fast key point extraction using simple TF-IDF"""
        try:
            # Simple sentence extraction without heavy NLP
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= num_points:
                return sentences
            
            # Simple scoring based on sentence length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                # Score based on length (prefer medium-length sentences)
                length_score = min(len(sentence) / 100, 1.0)
                
                # Score based on position (prefer beginning and end)
                position_score = 1.0 if i < 3 else (0.8 if i >= len(sentences) - 3 else 0.5)
                
                # Simple keyword scoring
                keyword_score = len([w for w in sentence.lower().split() 
                                   if w in ['important', 'key', 'main', 'significant', 'critical']]) * 0.2
                
                total_score = length_score + position_score + keyword_score
                scored_sentences.append((total_score, sentence))
            
            # Sort and return top sentences
            scored_sentences.sort(reverse=True)
            return [sentence for _, sentence in scored_sentences[:num_points]]
            
        except Exception as e:
            # Fallback: return first few sentences
            sentences = text.split('.')[:num_points]
            return [s.strip() + '.' for s in sentences if s.strip()]
    
    def get_recommended_voice_settings(self, sentiment: str, difficulty: str) -> Dict[str, any]:
        """Get recommended voice settings (fast, no ML needed)"""
        settings = {
            "speed": 180,
            "volume": 0.8,
            "voice_type": "neutral"
        }
        
        # Quick adjustments based on analysis
        if sentiment == "positive":
            settings["speed"] += 15
        elif sentiment == "negative":
            settings["speed"] -= 10
        
        if difficulty in ["difficult", "very difficult"]:
            settings["speed"] -= 25
        elif difficulty in ["easy", "very easy"]:
            settings["speed"] += 15
        
        settings["speed"] = max(120, min(250, settings["speed"]))
        
        return settings
    
    def _fast_clean_text(self, text: str) -> str:
        """Fast text cleaning for ML processing"""
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        # Basic whitespace normalization
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def detect_chapters(self, text: str) -> List[Dict[str, any]]:
        """Detect chapters in the document using multiple methods"""
        # Check cache first
        cache_key = self.get_cache_key(text[:2000], "chapters")
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            self.chapters = cached_result
            return cached_result
        
        chapters = []
        lines = text.split('\n')
        current_chapter = {"title": "Introduction", "start_line": 0, "content": "", "page_estimate": 1}
        
        # First, split by References sections to get natural chapter boundaries
        reference_chapters = self._split_by_references(text)
        if len(reference_chapters) > 1:
            return reference_chapters
        
        # Fallback to pattern-based detection if References method doesn't work
        # Chapter detection patterns (ordered by priority)
        chapter_patterns = [
            # Standard chapter formats
            (r'^Chapter\s+(\d+|[IVX]+)[\s\.:]\s*(.+)$', 'Chapter'),
            (r'^CHAPTER\s+(\d+|[IVX]+)[\s\.:]\s*(.+)$', 'Chapter'),
            (r'^(\d+)\.\s+([A-Z][^.]{10,80})$', 'Numbered Section'),
            
            # Academic paper sections
            (r'^(\d+)\.\s*(Introduction|Methodology|Results|Discussion|Conclusion|Abstract|References)$', 'Academic Section'),
            (r'^(Introduction|Methodology|Results|Discussion|Conclusion|Abstract|References)$', 'Academic Section'),
            
            # Book/document sections
            (r'^(Preface|Foreword|Introduction|Overview|Summary|Conclusion|Appendix|Bibliography|Index)$', 'Document Section'),
            (r'^Part\s+([IVX]+|\d+)[\s\.:]\s*(.+)$', 'Part'),
            (r'^Section\s+(\d+)[\s\.:]\s*(.+)$', 'Section'),
            
            # Technical document patterns
            (r'^(\d+\.\d+)\s+([A-Z][^.]{10,80})$', 'Subsection'),
            (r'^([A-Z][A-Z\s]{10,80}[A-Z])$', 'Major Heading'),
            
            # AI/ML specific patterns (for your PDF)
            (r'^(Machine Learning|Deep Learning|Neural Networks|Artificial Intelligence|Data Science|Computer Vision|Natural Language Processing).*$', 'AI Topic'),
            (r'^(\d+\.\d+)\s*(Machine Learning|Deep Learning|Neural Networks|Artificial Intelligence).*$', 'AI Section'),
        ]
        
        line_num = 0
        chapter_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check each pattern
            for pattern, chapter_type in chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous chapter if it has content
                    if current_chapter["content"].strip():
                        current_chapter["end_line"] = i - 1
                        current_chapter["word_count"] = len(current_chapter["content"].split())
                        current_chapter["page_estimate"] = max(1, current_chapter["word_count"] // 250)
                        chapters.append(current_chapter)
                        chapter_count += 1
                    
                    # Start new chapter
                    if match.groups():
                        if len(match.groups()) >= 2:
                            title = f"{match.group(1)} - {match.group(2)}"
                        else:
                            title = match.group(1)
                    else:
                        title = line
                    
                    current_chapter = {
                        "title": title[:100],  # Limit title length
                        "type": chapter_type,
                        "start_line": i,
                        "content": "",
                        "chapter_number": chapter_count + 1
                    }
                    break
            
            # Add line to current chapter content
            current_chapter["content"] += line + "\n"
        
        # Don't forget the last chapter
        if current_chapter["content"].strip():
            current_chapter["end_line"] = len(lines) - 1
            current_chapter["word_count"] = len(current_chapter["content"].split())
            current_chapter["page_estimate"] = max(1, current_chapter["word_count"] // 250)
            chapters.append(current_chapter)
        
        # If no chapters found, create automatic chunks
        if len(chapters) <= 1:
            chapters = self._create_automatic_chapters(text)
        
        # Post-process chapters
        processed_chapters = []
        for chapter in chapters:
            # Ensure reasonable chapter length (split if too long)
            if chapter.get("word_count", 0) > 5000:  # ~20 pages
                sub_chapters = self._split_long_chapter(chapter)
                processed_chapters.extend(sub_chapters)
            else:
                processed_chapters.append(chapter)
        
        # Cache result
        self.save_cached_result(cache_key, processed_chapters)
        self.chapters = processed_chapters
        return processed_chapters
    
    def _create_automatic_chapters(self, text: str, target_words_per_chapter: int = 2500) -> List[Dict[str, any]]:
        """Create automatic chapters based on text length"""
        chapters = []
        words = text.split()
        total_words = len(words)
        
        chapter_count = max(1, total_words // target_words_per_chapter)
        words_per_chapter = total_words // chapter_count
        
        for i in range(chapter_count):
            start_idx = i * words_per_chapter
            end_idx = min((i + 1) * words_per_chapter, total_words)
            
            chapter_words = words[start_idx:end_idx]
            chapter_content = ' '.join(chapter_words)
            
            # Try to find a good title from the beginning of the chapter
            first_sentences = chapter_content[:200].split('.')
            title = first_sentences[0][:50] if first_sentences else f"Chapter {i+1}"
            
            chapters.append({
                "title": f"Auto-Chapter {i+1}: {title}...",
                "type": "Auto-Generated",
                "start_line": start_idx,
                "end_line": end_idx,
                "content": chapter_content,
                "word_count": len(chapter_words),
                "page_estimate": max(1, len(chapter_words) // 250),
                "chapter_number": i + 1
            })
        
        return chapters
    
    def _split_long_chapter(self, chapter: Dict[str, any]) -> List[Dict[str, any]]:
        """Split a long chapter into smaller parts"""
        content = chapter["content"]
        words = content.split()
        target_size = 2500  # ~10 pages
        
        if len(words) <= target_size:
            return [chapter]
        
        parts = []
        num_parts = (len(words) + target_size - 1) // target_size
        
        for i in range(num_parts):
            start_idx = i * target_size
            end_idx = min((i + 1) * target_size, len(words))
            
            part_words = words[start_idx:end_idx]
            part_content = ' '.join(part_words)
            
            part = {
                "title": f"{chapter['title']} (Part {i+1})",
                "type": chapter.get("type", "Unknown"),
                "start_line": chapter.get("start_line", 0) + start_idx,
                "end_line": chapter.get("start_line", 0) + end_idx,
                "content": part_content,
                "word_count": len(part_words),
                "page_estimate": max(1, len(part_words) // 250),
                "chapter_number": f"{chapter.get('chapter_number', 1)}.{i+1}"
            }
            parts.append(part)
        
        return parts
    
    def _split_by_references(self, text: str) -> List[Dict[str, any]]:
        """Split text by References sections to create natural chapter boundaries"""
        chapters = []
        lines = text.split('\n')
        
        # Find all References sections with more flexible patterns
        reference_positions = []
        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            # Look for References section headers (ONLY major section headers, not individual citations)
            if (line_upper == "REFERENCES" or 
                line_upper == "REFERENCE" or
                line_upper == "BIBLIOGRAPHY" or
                line_clean == "References" or
                line_clean == "Reference" or
                line_clean == "Bibliography" or
                re.match(r'^REFERENCES?$', line_upper) or
                re.match(r'^BIBLIOGRAPHY$', line_upper) or
                re.match(r'^\d+\.\s*REFERENCES?$', line_clean, re.IGNORECASE) or
                re.match(r'^\d+\.\s*BIBLIOGRAPHY$', line_clean, re.IGNORECASE) or
                re.match(r'^Chapter\s+\d+\s+References?$', line_clean, re.IGNORECASE) or
                # Look for centered References (common in academic papers)
                (len(line_clean) < 20 and line_upper.strip() in ["REFERENCES", "REFERENCE", "BIBLIOGRAPHY"]) or
                # Look for References with page numbers or formatting
                re.match(r'^\s*REFERENCES?\s*\d*\s*$', line_upper)):
                # NOTE: Removed [1] detection to avoid too many small sections
                reference_positions.append(i)
                print(f"Found reference section at line {i}: '{line_clean}'")  # Debug output
        
        print(f"Found {len(reference_positions)} reference positions")
        if len(reference_positions) < 2:
            print("Not enough reference sections found for References-based detection")
            return []  # Need at least 2 references sections to create chapters
        
        print(f"Found {len(reference_positions)} reference sections, creating merged chapters...")
        
        # Create initial chapters based on References boundaries
        raw_chapters = []
        for i in range(len(reference_positions)):
            start_pos = 0 if i == 0 else reference_positions[i-1]
            end_pos = reference_positions[i]
            
            # Extract chapter content
            chapter_lines = lines[start_pos:end_pos]
            chapter_content = '\n'.join(chapter_lines)
            
            # Calculate chapter stats
            word_count = len(chapter_content.split())
            
            # Only add if chapter has substantial content
            if word_count > 100:  # At least 100 words
                raw_chapters.append({
                    "start_line": start_pos,
                    "end_line": end_pos,
                    "content": chapter_content.strip(),
                    "word_count": word_count,
                    "lines": chapter_lines
                })
        
        # Merge small chapters into larger ones (target: 2000-4000 words per chapter)
        merged_chapters = []
        current_merged = None
        target_min_words = 2000
        target_max_words = 4000
        
        for raw_chapter in raw_chapters:
            if current_merged is None:
                # Start a new merged chapter
                current_merged = {
                    "start_line": raw_chapter["start_line"],
                    "end_line": raw_chapter["end_line"],
                    "content": raw_chapter["content"],
                    "word_count": raw_chapter["word_count"],
                    "lines": raw_chapter["lines"]
                }
            elif current_merged["word_count"] < target_min_words:
                # Merge with current chapter if it's still too small
                current_merged["end_line"] = raw_chapter["end_line"]
                current_merged["content"] += "\n" + raw_chapter["content"]
                current_merged["word_count"] += raw_chapter["word_count"]
                current_merged["lines"].extend(raw_chapter["lines"])
            else:
                # Current chapter is big enough, finalize it and start a new one
                merged_chapters.append(current_merged)
                current_merged = {
                    "start_line": raw_chapter["start_line"],
                    "end_line": raw_chapter["end_line"],
                    "content": raw_chapter["content"],
                    "word_count": raw_chapter["word_count"],
                    "lines": raw_chapter["lines"]
                }
        
        # Don't forget the last merged chapter
        if current_merged:
            merged_chapters.append(current_merged)
        
        # Convert merged chapters to final format
        for i, merged_chapter in enumerate(merged_chapters):
            # Try to find a good chapter title from the beginning
            chapter_title = self._extract_chapter_title(merged_chapter["lines"])
            if not chapter_title:
                chapter_title = f"Chapter {i+1}"
            
            page_estimate = max(1, merged_chapter["word_count"] // 250)
            
            chapter = {
                "title": chapter_title,
                "type": "Reference-Bounded Chapter",
                "start_line": merged_chapter["start_line"],
                "end_line": merged_chapter["end_line"],
                "content": merged_chapter["content"],
                "word_count": merged_chapter["word_count"],
                "page_estimate": page_estimate,
                "chapter_number": i + 1
            }
            chapters.append(chapter)
        
        # Add the final chapter (after last References to end of document)
        if reference_positions:
            last_ref_pos = reference_positions[-1]
            final_lines = lines[last_ref_pos:]
            final_content = '\n'.join(final_lines)
            final_word_count = len(final_content.split())
            
            if final_word_count > 100:  # Only add if substantial
                final_title = self._extract_chapter_title(final_lines[:20])  # Look in first 20 lines
                if not final_title:
                    final_title = f"Chapter {len(chapters) + 1}"
                
                final_chapter = {
                    "title": final_title,
                    "type": "Reference-Bounded Chapter",
                    "start_line": last_ref_pos,
                    "end_line": len(lines),
                    "content": final_content.strip(),
                    "word_count": final_word_count,
                    "page_estimate": max(1, final_word_count // 250),
                    "chapter_number": len(chapters) + 1
                }
                chapters.append(final_chapter)
        
        # Cache and return if we found good chapters
        if len(chapters) >= 2:
            cache_key = self.get_cache_key(text[:2000], "ref_chapters")
            self.save_cached_result(cache_key, chapters)
            self.chapters = chapters
            return chapters
        
        return []  # Return empty if not enough chapters found
    
    def _extract_chapter_title(self, lines: List[str]) -> str:
        """Extract a meaningful chapter title from the beginning of chapter lines"""
        # Look for various title patterns in the first 20 lines
        for line in lines[:20]:
            line = line.strip()
            if not line:
                continue
            
            # Skip common non-title patterns
            if (line.upper() in ["REFERENCES", "REFERENCE", "BIBLIOGRAPHY"] or
                line.startswith('[') or  # Reference citations
                line.startswith('http') or  # URLs
                len(line) < 10 or  # Too short
                line.count('.') > 3 or  # Likely a sentence, not title
                line.startswith('Complimentary Contributor Copy') or
                line.startswith('Page ') or
                line.isdigit()):  # Just numbers
                continue
            
            # Look for chapter-like patterns first
            chapter_patterns = [
                r'^Chapter\s+(\d+|[IVX]+)[\s\.:]\s*(.+)$',
                r'^CHAPTER\s+(\d+|[IVX]+)[\s\.:]\s*(.+)$',
                r'^(\d+)\.\s+([A-Z][^.]{10,80})$',
                r'^Part\s+([IVX]+|\d+)[\s\.:]\s*(.+)$',
                r'^Section\s+(\d+)[\s\.:]\s*(.+)$'
            ]
            
            for pattern in chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2:
                        return f"{match.group(1)} - {match.group(2)}"[:80]
                    else:
                        return match.group(1)[:80]
            
            # Look for AI/ML topic titles
            ai_keywords = [
                'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
                'data science', 'computer vision', 'natural language processing', 'robotics',
                'algorithm', 'classification', 'regression', 'clustering', 'optimization',
                'supervised learning', 'unsupervised learning', 'reinforcement learning',
                'convolutional', 'recurrent', 'transformer', 'gradient descent', 'backpropagation',
                'feature engineering', 'model training', 'prediction', 'inference', 'automation',
                'intelligence', 'learning', 'neural', 'network', 'data', 'analysis', 'model',
                'system', 'application', 'implementation', 'methodology', 'approach', 'framework'
            ]
            
            # Look for title-like patterns with AI/ML keywords
            line_lower = line.lower()
            if (line[0].isupper() and  # Starts with capital
                len(line) < 120 and  # Reasonable title length
                not line.endswith('.') and  # Doesn't end with period
                any(keyword in line_lower for keyword in ai_keywords)):
                
                # Clean up the title
                title = line
                # Remove common prefixes
                title = re.sub(r'^(Complimentary Contributor Copy\s*)', '', title)
                title = re.sub(r'^(Page \d+\s*)', '', title)
                
                if len(title.strip()) >= 10:
                    return title.strip()[:80]
            
            # Look for all-caps headings (common in academic papers)
            if (line.isupper() and 
                10 <= len(line) <= 80 and
                not line.startswith('HTTP') and
                not line.startswith('WWW') and
                any(keyword.upper() in line for keyword in ai_keywords)):
                return line.title()[:80]  # Convert to title case
        
        return ""  # No good title found
    
    def get_chapter_list(self) -> List[Dict[str, str]]:
        """Get a simplified list of chapters for UI display"""
        return [
            {
                "number": ch.get("chapter_number", i+1),
                "title": ch["title"],
                "pages": ch.get("page_estimate", 1),
                "words": ch.get("word_count", 0),
                "type": ch.get("type", "Unknown")
            }
            for i, ch in enumerate(self.chapters)
        ]
    
    def get_chapter_content(self, chapter_index: int) -> str:
        """Get content for a specific chapter"""
        if 0 <= chapter_index < len(self.chapters):
            return self.chapters[chapter_index]["content"]
        return ""
    
    def clear_cache(self):
        """Clear all cached results"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            return True
        except:
            return False

# Global instance
ml_features_optimized = OptimizedMLFeatures() 