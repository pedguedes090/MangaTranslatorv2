#!/usr/bin/env python3
"""
Multi-Language OCR System for Comic Translation
===============================================

A comprehensive OCR system that automatically selects the best OCR engine
based on the source language:

- manga-ocr: Specialized for Japanese manga text
- PaddleOCR: Optimized for Chinese manhua text  
- EasyOCR: Good for Korean manhwa and multilingual text
- TrOCR: General purpose fallback OCR

Author: MangaTranslator Team
License: MIT
"""

# Standard library imports
import cv2
import numpy as np
from PIL import Image
import torch

# OCR engine imports
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from manga_ocr import MangaOcr


class MultiLanguageOCR:
    """
    Multi-language OCR system that automatically selects the best OCR engine
    based on the target language for optimal text recognition.
    """
    
    def __init__(self):
        """Initialize multi-language OCR engines lazily for better performance"""
        print("🔧 Initializing Multi-Language OCR engines...")
        
        # OCR engines - initialized on demand for better memory usage
        self.manga_ocr = None          # Japanese OCR (manga-ocr) - Best for manga
        self.paddle_ocr = None         # Chinese OCR (PaddleOCR) - Best for manhua
        self.easy_ocr = None           # Multi-language OCR (EasyOCR) - Good for manhwa
        self.easy_ocr_ja = None        # Japanese EasyOCR (separate instance)
        self.trocr_processor = None    # General OCR (TrOCR) - Fallback
        self.trocr_model = None
        
        print("✅ OCR engines ready for initialization")

    def _init_manga_ocr(self):
        """Initialize Japanese manga OCR engine"""
        if self.manga_ocr is None:
            print("📚 Loading manga-ocr for Japanese...")
            self.manga_ocr = MangaOcr()
            print("✅ manga-ocr ready for Japanese text")

    def _init_paddle_ocr(self):
        """Initialize PaddleOCR for Chinese text"""
        """Initialize Chinese manhua OCR"""
        if self.paddle_ocr is None:
            print("🐼 Loading PaddleOCR for Chinese...")
            try:
                # New PaddleOCR API (v5+)
                self.paddle_ocr = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False, 
                    use_textline_orientation=False,
                    lang='ch'
                )
                print("✅ PaddleOCR ready for Chinese text")
            except Exception as e:
                print(f"❌ PaddleOCR initialization failed: {e}")
                print("💡 Trying fallback initialization...")
                try:
                    # Fallback to older API
                    self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                    print("✅ PaddleOCR ready (fallback mode)")
                except Exception as e2:
                    print(f"❌ PaddleOCR fallback failed: {e2}")
                    self.paddle_ocr = None

    def _init_easy_ocr(self):
        """Initialize Korean manhwa OCR"""
        if self.easy_ocr is None:
            print("👀 Loading EasyOCR for multi-language...")
            # Use only Korean and English to avoid compatibility issues
            # Japanese conflicts with other Asian languages in EasyOCR
            self.easy_ocr = easyocr.Reader(['ko', 'en'], gpu=False)
            print("✅ EasyOCR ready for Korean + English")

    def _init_easy_ocr_ja(self):
        """Initialize Japanese EasyOCR (separate from Korean OCR)"""
        if self.easy_ocr_ja is None:
            print("👀 Loading EasyOCR for Japanese...")
            # Japanese only works with English in EasyOCR
            self.easy_ocr_ja = easyocr.Reader(['ja', 'en'], gpu=False)
            print("✅ EasyOCR ready for Japanese + English")

    def _init_trocr(self):
        """Initialize TrOCR for general text"""
        if self.trocr_processor is None:
            print("🤖 Loading TrOCR for general text...")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            print("✅ TrOCR ready for general text")

    def extract_text(self, image, source_lang="auto", method="auto"):
        """
        Extract text from comic bubble image
        
        Args:
            image: PIL Image or numpy array
            source_lang: "ja", "zh", "ko", "en", "auto"
            method: "manga_ocr", "paddle", "easy", "trocr", "auto"
        """
        
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Auto-select OCR based on language
        if method == "auto":
            if source_lang == "ja":
                method = "manga_ocr"  # Best for Japanese manga
            elif source_lang == "zh":
                method = "paddle"     # Best for Chinese manhua
            elif source_lang == "ko":
                method = "easy"       # Good for Korean manhwa
            elif source_lang == "en":
                method = "easy"       # Good for English comics
            else:  # auto or unknown
                method = "easy"       # EasyOCR as general fallback

        try:
            if method == "manga_ocr":
                return self._extract_with_manga_ocr(image)
            elif method == "paddle":
                return self._extract_with_paddle_ocr(image)
            elif method == "easy":
                # Use appropriate EasyOCR based on source language
                if source_lang == "ja":
                    return self._extract_with_easy_ocr_ja(image)
                else:
                    return self._extract_with_easy_ocr(image)
            elif method == "trocr":
                return self._extract_with_trocr(image)
            else:
                # Fallback to appropriate EasyOCR
                if source_lang == "ja":
                    return self._extract_with_easy_ocr_ja(image)
                else:
                    return self._extract_with_easy_ocr(image)
                
        except Exception as e:
            print(f"❌ OCR failed with {method}: {e}")
            # Smart fallback based on language
            try:
                if source_lang == "ja":
                    # For Japanese: try EasyOCR-JA -> manga-ocr
                    if method != "easy_ja":
                        return self._extract_with_easy_ocr_ja(image)
                    elif method != "manga_ocr":
                        return self._extract_with_manga_ocr(image)
                elif source_lang == "zh":
                    # For Chinese: try EasyOCR -> TrOCR
                    if method != "easy":
                        return self._extract_with_easy_ocr(image)
                    else:
                        return self._extract_with_trocr(image)
                elif source_lang == "ko":
                    # For Korean: try TrOCR -> manga-ocr
                    if method != "trocr":
                        return self._extract_with_trocr(image)
                    else:
                        return self._extract_with_manga_ocr(image)
                else:
                    # For others: general fallback
                    return self._extract_with_easy_ocr(image)
            except:
                return "OCR_ERROR"

    def _extract_with_manga_ocr(self, image):
        """Extract Japanese text using manga-ocr"""
        self._init_manga_ocr()
        try:
            text = self.manga_ocr(image)
            return text.strip()
        except Exception as e:
            print(f"❌ manga-ocr error: {e}")
            return ""

    def _extract_with_paddle_ocr(self, image):
        """Extract Chinese text using PaddleOCR"""
        self._init_paddle_ocr()
        
        if self.paddle_ocr is None:
            print("❌ PaddleOCR not initialized")
            return ""
            
        try:
            # Convert PIL to numpy for PaddleOCR
            img_array = np.array(image)
            
            # Use new PaddleOCR API (predict)
            results = self.paddle_ocr.predict(img_array)
            
            if results:
                texts = []
                
                # Parse new PaddleOCR format - OCRResult object
                for result in results:
                    try:
                        rec_texts = result['rec_texts']
                        rec_scores = result['rec_scores']
                        
                        for text, score in zip(rec_texts, rec_scores):
                            if text.strip() and score > 0.5:  # Filter by confidence and non-empty
                                texts.append(text.strip())
                    except (KeyError, TypeError) as e:
                        print(f"❌ PaddleOCR result parsing error: {e}")
                        continue
                
                return " ".join(texts) if texts else ""
            
            return ""
            
        except Exception as e:
            print(f"❌ PaddleOCR error: {e}")
            return ""

    def _extract_with_easy_ocr(self, image):
        """Extract text using EasyOCR (Korean + English)"""
        self._init_easy_ocr()
        try:
            # Convert PIL to numpy for EasyOCR
            img_array = np.array(image)
            
            # EasyOCR returns [(box, text, confidence)] or [(box, text)]
            results = self.easy_ocr.readtext(img_array, paragraph=True)
            
            if results:
                texts = []
                for result in results:
                    if len(result) >= 2:  # Handle both formats
                        bbox, text = result[0], result[1]
                        conf = result[2] if len(result) > 2 else 1.0
                        
                        if conf > 0.5:  # confidence threshold
                            texts.append(text)
                return " ".join(texts)
            return ""
            
        except Exception as e:
            print(f"❌ EasyOCR error: {e}")
            return ""

    def _extract_with_easy_ocr_ja(self, image):
        """Extract Japanese text using EasyOCR (Japanese + English only)"""
        self._init_easy_ocr_ja()
        try:
            # Convert PIL to numpy for EasyOCR
            img_array = np.array(image)
            
            # EasyOCR returns [(box, text, confidence)] or [(box, text)]
            results = self.easy_ocr_ja.readtext(img_array, paragraph=True)
            
            if results:
                texts = []
                for result in results:
                    if len(result) >= 2:  # Handle both formats
                        bbox, text = result[0], result[1]
                        conf = result[2] if len(result) > 2 else 1.0
                        
                        if conf > 0.5:  # confidence threshold
                            texts.append(text)
                return " ".join(texts)
            return ""
            
        except Exception as e:
            print(f"❌ EasyOCR Japanese error: {e}")
            return ""

    def _extract_with_trocr(self, image):
        """Extract text using TrOCR (general purpose)"""
        self._init_trocr()
        try:
            # Preprocess image
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            
            # Generate text
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ TrOCR error: {e}")
            return ""

    def get_best_ocr_for_language(self, source_lang):
        """Get recommended OCR method for language"""
        recommendations = {
            "ja": ("manga_ocr", "🇯🇵 manga-ocr → EasyOCR-JA (Specialized for Japanese)"),
            "zh": ("paddle", "🇨🇳 PaddleOCR → EasyOCR (Optimized for Chinese)"),
            "ko": ("easy", "🇰🇷 EasyOCR → TrOCR (Good for Korean manhwa)"),
            "en": ("easy", "🇺🇸 EasyOCR (Multi-language support)"),
            "auto": ("easy", "🌍 EasyOCR → Smart fallback (Auto-detect)")
        }
        return recommendations.get(source_lang, ("easy", "🌍 EasyOCR (Fallback)"))

    def benchmark_ocr_methods(self, image, source_lang="auto"):
        """Compare all OCR methods on the same image"""
        print(f"\n🧪 OCR Benchmark for language: {source_lang}")
        print("=" * 60)
        
        methods = [
            ("manga_ocr", "🇯🇵 manga-ocr"),
            ("paddle", "🇨🇳 PaddleOCR"), 
            ("easy", "🇰🇷 EasyOCR"),
            ("trocr", "🤖 TrOCR")
        ]
        
        results = {}
        for method, name in methods:
            try:
                import time
                start_time = time.time()
                text = self.extract_text(image, source_lang, method)
                elapsed = time.time() - start_time
                
                results[method] = {
                    'text': text,
                    'time': elapsed,
                    'success': len(text.strip()) > 0
                }
                
                print(f"{name:20} | {elapsed:5.2f}s | {text[:50]}")
                
            except Exception as e:
                results[method] = {
                    'text': f"ERROR: {e}",
                    'time': 0,
                    'success': False
                }
                print(f"{name:20} | ERROR  | {str(e)[:50]}")
        
        return results


if __name__ == "__main__":
    # Test script
    print("🧪 Testing Multi-Language OCR")
    
    ocr = MultiLanguageOCR()
    
    # Test recommendations
    for lang in ["ja", "zh", "ko", "en", "auto"]:
        method, desc = ocr.get_best_ocr_for_language(lang)
        print(f"Language '{lang}': {desc}")
