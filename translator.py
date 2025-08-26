#!/usr/bin/env python3
"""
Enhanced Manga Translator Module
===============================

A comprehensive translation system for manga/comic text with context-aware AI translation.

🆕 NEW FEATURES:
- Context metadata support for smart pronoun/honorific selection
- Locked output format (translation only - no explanations)  
- Language-specific rules for JA/ZH/KO comics
- SFX and thought bubble specialized handling
- Bubble fitting with character limits
- Line preservation for multi-bubble text

Features:
- Multiple translation backends (Google, Gemini AI, HuggingFace, Sogou, Bing)
- Context-aware translation with relationship/formality/gender metadata
- Language-specific prompts (Japanese manga, Chinese manhua, Korean manhwa)
- Clean output guarantee (no AI explanations or multiple options)
- Automatic language detection
- Error handling and fallbacks

Author: MangaTranslator Team  
License: MIT
Version: 2.0 (Enhanced Prompt System)
"""

# Translation libraries
from deep_translator import GoogleTranslator
from transformers import pipeline
import translators as ts

# Standard library imports
import requests
import random
import time
import os
import json
from typing import List, Dict, Optional, Tuple

# Import API Key Manager
from api_key_manager import ApiKeyManager


class MangaTranslator:
    """
    Multi-service translator optimized for manga/comic text translation with context awareness.
    
    🆕 NEW: Context metadata support for intelligent translation:
    - Smart pronoun/honorific selection based on relationship and formality
    - Gender-aware translation for natural Vietnamese output  
    - Bubble fitting with character limits
    - SFX and internal thought specialized handling
    - Clean output guarantee (no AI explanations)
    """
    
    def __init__(self, gemini_api_key=None):
        """
        Initialize the translator with optional Gemini API key and API Key Manager
        
        Args:
            gemini_api_key (str, optional): Gemini API key for AI translation
        """
        self.target = "vi"  # Target language: Vietnamese
        
        # Supported source languages mapping
        self.supported_languages = {
            "auto": "Tự động nhận diện",
            "ja": "Tiếng Nhật (Manga)",
            "zh": "Tiếng Trung (Manhua)", 
            "ko": "Tiếng Hàn (Manhwa)",
            "en": "Tiếng Anh"
        }
        
        # Initialize API Key Manager
        try:
            self.api_key_manager = ApiKeyManager()
            print("✅ API Key Manager initialized")
        except Exception as e:
            print(f"⚠️ API Key Manager failed to initialize: {e}")
            self.api_key_manager = None
        
        # Initialize Gemini API key (fallback to single key mode)
        if not gemini_api_key:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key and gemini_api_key.strip():
            self.gemini_api_key = gemini_api_key.strip()
            print(f"✅ Fallback Gemini API key configured: {self.gemini_api_key[:10]}...")
        else:
            self.gemini_api_key = None
        
        # Translation methods mapping
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            "sogou": self._translate_with_sogou,
            "bing": self._translate_with_bing,
            "gemini": self._translate_with_gemini
        }

    def translate(self, text, method="google", source_lang="auto", context=None, custom_prompt=None):
        """
        Translate text to Vietnamese using the specified method with context support
        
        Args:
            text (str): Text to translate
            method (str): Translation method ("google", "gemini", "hf", "sogou", "bing")
            source_lang (str): Source language code - "auto", "ja", "zh", "ko", "en"
            context (dict, optional): Context metadata for better translation:
                - gender: 'male'/'female'/'neutral' (default: 'neutral')
                - relationship: 'friend'/'senior'/'junior'/'family'/'stranger' (default: 'neutral') 
                - formality: 'casual'/'polite'/'formal' (default: 'casual')
                - bubble_limit: int (character limit for bubble fitting)
                - is_thought: bool (internal monologue/thought bubble)
                - is_sfx: bool (sound effect)
                - scene_context: str (brief scene description)
            custom_prompt (str, optional): Custom translation style prompt to override defaults
            
        Returns:
            str: Translated text in Vietnamese
        """
        
        # Kiểm tra xem có API key Gemini không (từ manager hoặc fallback)
        has_gemini_key = False
        if method == "gemini":
            # Kiểm tra API Key Manager trước
            if self.api_key_manager:
                try:
                    manager_key = self.api_key_manager.get_api_key('gemini')
                    has_gemini_key = bool(manager_key)
                except:
                    has_gemini_key = False
            
            # Nếu không có từ manager, kiểm tra fallback key
            if not has_gemini_key:
                has_gemini_key = bool(self.gemini_api_key)
        
        if method == "gemini" and not has_gemini_key:
            print("⚠️ Gemini API not available, falling back to Google Translate")
            method = "google"
        elif method == "gemini" and has_gemini_key:
            print("🤖 Using Gemini 2.0 Flash for context-aware translation")
            
        translator_func = self.translators.get(method)

        if translator_func:
            if method == "gemini":
                return translator_func(self._preprocess_text(text), source_lang, context, custom_prompt)
            else:
                return translator_func(self._preprocess_text(text), source_lang)
        else:
            raise ValueError("Invalid translation method.")
            
    def _translate_with_google(self, text, source_lang="auto"):
        self._delay()
        
        # Map our language codes to Google's codes
        google_lang = source_lang
        if source_lang == "zh":
            google_lang = "zh-cn"
        
        translator = GoogleTranslator(source=google_lang, target=self.target)
        translated_text = translator.translate(text)
        return translated_text if translated_text is not None else text

    def _translate_with_hf(self, text, source_lang="auto"):
        # HF pipeline chỉ hỗ trợ Japanese to English, fallback to Google
        print("⚠️ Helsinki-NLP chỉ hỗ trợ Nhật → Anh, chuyển sang Google Translate")
        return self._translate_with_google(text, source_lang)

    def _translate_with_sogou(self, text, source_lang="auto"):
        self._delay()
        
        # Map to sogou language codes
        sogou_lang = "auto" if source_lang == "auto" else source_lang
        
        translated_text = ts.translate_text(text, translator="sogou",
                                            from_language=sogou_lang,
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_bing(self, text, source_lang="auto"):
        self._delay()
        
        # Map to bing language codes  
        bing_lang = "auto" if source_lang == "auto" else source_lang
        
        translated_text = ts.translate_text(text, translator="bing",
                                            from_language=bing_lang, 
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_gemini(self, text, source_lang="auto", context=None, custom_prompt=None):
        """
        Translate using Google Gemini 2.0 Flash with context metadata support.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language
            context (dict, optional): Context metadata with keys:
                - gender: 'male'/'female'/'neutral'
                - relationship: 'friend'/'senior'/'junior'/'family'/'stranger'
                - formality: 'casual'/'polite'/'formal'
                - bubble_limit: int (character limit)
                - is_thought: bool (internal monologue)
                - is_sfx: bool (sound effect)
                - scene_context: str (brief scene description)
        """
        # Lấy API key từ manager hoặc fallback
        api_key = None
        
        # Thử lấy từ API Key Manager trước
        if self.api_key_manager:
            try:
                api_key = self.api_key_manager.get_api_key('gemini')
            except Exception as e:
                print(f"⚠️ Không thể lấy API key từ manager: {e}")
        
        # Nếu không có từ manager, dùng fallback key
        if not api_key:
            api_key = self.gemini_api_key
        
        if not api_key:
            raise ValueError("Gemini API key not configured")
        
        # Clean input text
        text = text.strip() if text else ""
        if not text:
            print("⚠️ Empty text sent to Gemini, skipping")
            return ""
            
        # Debug logging
        print(f"🤖 Gemini input: '{text}' | Lang: {source_lang}")
        if context:
            print(f"📋 Context: {context}")
            
        try:
            # Use REST API directly for more reliable connection
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': api_key
            }
            
            # Get specialized prompt based on source language and context
            prompt = self._get_translation_prompt(text, source_lang, context, custom_prompt)
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 200,
                    "topP": 0.9,
                    "topK": 40
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    translated_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    
                    # Aggressive cleanup - remove any AI explanations
                    translated_text = self._clean_gemini_response(translated_text)
                    
                    return translated_text if translated_text else text
                else:
                    print("❌ No translation candidates in response")
                    return self._translate_with_google(text, source_lang)
            else:
                print(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return self._translate_with_google(text, source_lang)
            
        except Exception as e:
            print(f"Gemini translation failed: {e}")
            # Fallback to Google Translate
            return self._translate_with_google(text, source_lang)

    def _get_translation_prompt(self, text, source_lang, context=None, custom_prompt=None):
        """
        Generate enhanced translation prompt with context metadata support
        """
        # If custom prompt provided, use it as instruction, not text to translate
        if custom_prompt and custom_prompt.strip():
            return f"""Bạn là một chuyên gia dịch thuật manga/comic chuyên nghiệp.

INSTRUCTION: {custom_prompt.strip()}

Text cần dịch: "{text}"

CHỈ trả về bản dịch tiếng Việt của text trên, không giải thích gì thêm."""
        
        # Use default prompt system
        # Parse context metadata
        gender = context.get('gender', 'neutral') if context else 'neutral'
        relationship = context.get('relationship', 'neutral') if context else 'neutral'  
        formality = context.get('formality', 'casual') if context else 'casual'
        bubble_limit = context.get('bubble_limit', None) if context else None
        is_thought = context.get('is_thought', False) if context else False
        is_sfx = context.get('is_sfx', False) if context else False
        scene_context = context.get('scene_context', '') if context else ''
        
        # Build context info
        context_info = []
        if gender != 'neutral':
            context_info.append(f"GENDER: {gender}")
        if relationship != 'neutral':
            context_info.append(f"RELATIONSHIP: {relationship}")
        if formality != 'casual':
            context_info.append(f"FORMALITY: {formality}")
        if bubble_limit:
            context_info.append(f"BUBBLE_LIMIT: {bubble_limit} chars")
        if is_thought:
            context_info.append("TYPE: internal_thought")
        if is_sfx:
            context_info.append("TYPE: sound_effect")
        if scene_context:
            context_info.append(f"SCENE: {scene_context}")
            
        context_str = " | ".join(context_info) if context_info else "No specific context"
        
        # Get language-specific rules
        lang_rules = self._get_language_rules(source_lang)
        
        return f"""Dịch "{text}" sang tiếng Việt.

CONTEXT: {context_str}

{lang_rules}

GLOBAL RULES:
- Chỉ trả về chuỗi bản dịch, không nhãn, không ngoặc kép, không giải thích
- Một dòng vào → một dòng ra (bảo toàn số dòng)
- Xưng hô tự động theo relationship/formality: bạn bè→"tôi/cậu"; lịch sự→"tôi/anh(chị)"; thân mật→"tao/mày"
- Không sáng tác thêm, dịch trung thực nhưng mượt
- Tên riêng/ký hiệu: giữ nguyên
- Dấu câu Việt: "…" cho thở dài, "—" cho ngắt mạnh
- SFX: dịch ngắn mạnh (vd: "RẦM!", "BỤP!")
- Thought: dùng "…" mềm, tránh đại từ nặng
- Bubble fit: ưu tiên câu ngắn tự nhiên

CHỈ TRẢ VỀ BẢN DỊCH:"""
    def _get_language_rules(self, source_lang):
        """Get language-specific translation rules"""
        if source_lang == "ja":
            return """JA RULES:
- Keigo→"ạ/dạ"; thường→bỏ kính ngữ
- Senpai/kouhai→"tiền bối/hậu bối" hoặc giữ nguyên
- やばい→"Chết tiệt!/Tệ rồi!"; すごい→"Đỉnh quá!"  
- 技→"kỹ thuật/chiêu"; 必殺技→"tuyệt kỹ"; 変身→"biến hình"
- SFX: バン→"BÙNG!"; ドン→"RẦM!"; キラキラ→"lấp lánh" """

        elif source_lang == "zh":
            return """ZH RULES:
- 您→"Ngài/thưa"; 你→"anh/chị"; 朕→"Trẫm"; 本王→"Bản vương"
- 武功→"võ công"; 轻功→"khinh công"; 江湖→"giang hồ"
- 境界→"cảnh giới"; 丹药→"đan dược"; 法宝→"pháp bảo"  
- 哼→"Hừ!"; 哎呀→"Ôi trời!"; 天啊→"Trời ơi!"
- SFX: 轰→"BÙMM!"; 砰→"ĐỤC!"; 咔嚓→"KẮC!" """

        elif source_lang == "ko":
            return """KO RULES:
- Jondaetmal→"ạ/dạ"; banmal→bỏ kính ngữ
- 형/누나/오빠/언니→"anh/chị"; 선배→"tiền bối" 
- 아이고→"Ôi giời!"; 헐→"Hả?!"; 와→"Wow!"
- 능력→"năng lực"; 각성→"thức tỉnh"; 레벨업→"lên cấp"
- SFX: 쾅→"CẠCH!"; 쿵→"RẦM!"; 휘익→"VỪN!" """

        else:
            return """GENERAL RULES:
- Phân biệt formal/informal, nam/nữ, già/trẻ
- Cảm thán: "Ồ!", "Trời!", "Chết tiệt!"
- Hiệu ứng âm thanh: dịch phù hợp tiếng Việt"""

    def _clean_gemini_response(self, response):
        """Enhanced cleaning to remove any AI explanations and return only translation"""
        if not response:
            return ""
            
        # Remove quotes and common prefixes
        cleaned = response.strip().strip('"').strip("'")
        
        # Remove translation labels
        prefixes_to_remove = [
            "Bản dịch:", "Dịch:", "Translation:", "Vietnamese:",
            "Tiếng Việt:", "Câu dịch:", "Kết quả:", "Đáp án:",
            "Bản dịch tiếng Việt:", "Vietnamese translation:",
            "Tôi sẽ dịch:", "Đây là bản dịch:", "Câu trả lời:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Split by common explanation indicators and take first part
        explanation_splits = [
            " (", "[", "Hoặc", "Tùy", "Nếu", "* ", "• ",
            "- ", "Giải thích:", "Lưu ý:", "Chú thích:",
            "Có thể", "Tuỳ theo", "Tùy vào"
        ]
        
        for split_pattern in explanation_splits:
            if split_pattern in cleaned:
                parts = cleaned.split(split_pattern)
                if parts[0].strip():
                    cleaned = parts[0].strip()
                    break
        
        # Clean extra whitespace and newlines
        cleaned = " ".join(cleaned.split())
        
        # Final validation - if it contains typical AI response patterns, extract the core translation
        ai_patterns = [
            "có thể dịch", "tùy ngữ cảnh", "tuỳ theo", "hoặc là",
            "một cách khác", "phiên bản khác", "cách khác"
        ]
        
        for pattern in ai_patterns:
            if pattern in cleaned.lower():
                # Try to extract the first clean sentence before the pattern
                sentences = cleaned.split('.')
                if sentences and len(sentences[0]) > 3:
                    cleaned = sentences[0].strip()
                    break
                    
        return cleaned.rstrip('.,!?;:')

    def _preprocess_text(self, text):
        """Enhanced preprocessing for different comic types"""
        # Basic cleaning
        preprocessed_text = text.replace("．", ".")
        
        # Remove excessive whitespace
        preprocessed_text = " ".join(preprocessed_text.split())
        
        # Clean up common OCR artifacts
        preprocessed_text = preprocessed_text.replace("（", "(").replace("）", ")")
        preprocessed_text = preprocessed_text.replace("！", "!").replace("？", "?")
        
        return preprocessed_text

    def _delay(self):
        time.sleep(random.randint(3, 5))

    # ============================================================================
    # BATCH TRANSLATION METHODS - New Feature
    # ============================================================================
    
    def batch_translate(self, texts: List[str], method="gemini", source_lang="auto", 
                       context=None, custom_prompt=None) -> List[str]:
        """
        Dịch batch texts - tối ưu cho việc dịch nhiều text cùng lúc
        
        Args:
            texts (List[str]): Danh sách texts cần dịch
            method (str): Translation method
            source_lang (str): Source language
            context (dict, optional): Context metadata
            custom_prompt (str, optional): Custom prompt
            
        Returns:
            List[str]: Danh sách texts đã dịch
        """
        if not texts:
            return []
        
        # Lọc bỏ text rỗng và chuẩn bị
        clean_texts = []
        text_indices = []  # Track original indices
        
        for i, text in enumerate(texts):
            cleaned = self._preprocess_text(text) if text else ""
            if cleaned.strip():
                clean_texts.append(cleaned)
                text_indices.append(i)
        
        if not clean_texts:
            return [""] * len(texts)
        
        print(f"🔄 Batch translating {len(clean_texts)} texts using {method}")
        
        # Sử dụng API Key Manager nếu có
        if method == "gemini" and self.api_key_manager:
            return self._batch_translate_with_manager(texts, clean_texts, text_indices, 
                                                    source_lang, context, custom_prompt)
        
        # Fallback to individual translation
        results = []
        for text in texts:
            if text and text.strip():
                translated = self.translate(text, method, source_lang, context, custom_prompt)
                results.append(translated)
            else:
                results.append("")
        
        return results
    
    def _batch_translate_with_manager(self, original_texts: List[str], clean_texts: List[str], 
                                    text_indices: List[int], source_lang: str, 
                                    context: dict, custom_prompt: str) -> List[str]:
        """
        Batch translate using API Key Manager với rotation
        """
        # Tạo batch prompt cho tất cả texts
        batch_prompt = self._create_batch_prompt(clean_texts, source_lang, context, custom_prompt)
        
        # Thử translate toàn bộ batch với API Key Manager
        def translate_func(prompt, api_key):
            return self._translate_batch_with_gemini(prompt, api_key)
        
        batch_result = self.api_key_manager.batch_translate_with_rotation(
            [batch_prompt], translate_func, 'gemini', max_retries=2
        )
        
        if batch_result and batch_result[0]:
            # Parse kết quả batch
            individual_results = self._parse_batch_result(batch_result[0], len(clean_texts))
            
            if len(individual_results) == len(clean_texts):
                # Map kết quả về vị trí gốc
                final_results = [""] * len(original_texts)
                for i, text_idx in enumerate(text_indices):
                    final_results[text_idx] = individual_results[i]
                
                return final_results
        
        # Fallback: translate từng cái một
        print("⚠️ Batch translation failed, falling back to individual translation")
        return self._fallback_individual_translate(original_texts, source_lang, context, custom_prompt)
    
    def _create_batch_prompt(self, texts: List[str], source_lang: str, 
                           context: dict, custom_prompt: str) -> str:
        """
        Tạo prompt cho batch translation
        """
        # Get language rules
        lang_rules = self._get_language_rules(source_lang)
        
        # Build context
        context_info = []
        if context:
            gender = context.get('gender', 'neutral')
            relationship = context.get('relationship', 'neutral')
            formality = context.get('formality', 'casual')
            
            if gender != 'neutral':
                context_info.append(f"GENDER: {gender}")
            if relationship != 'neutral':
                context_info.append(f"RELATIONSHIP: {relationship}")
            if formality != 'casual':
                context_info.append(f"FORMALITY: {formality}")
        
        context_str = " | ".join(context_info) if context_info else "No specific context"
        
        # Custom prompt hoặc default
        if custom_prompt and custom_prompt.strip():
            instruction = f"""Bạn là chuyên gia dịch manga. Hãy dịch CHÍNH XÁC từng câu sau sang tiếng Việt.

INSTRUCTION: {custom_prompt.strip()}

BATCH RULES:
- Dịch từng dòng một cách riêng biệt
- Giữ nguyên thứ tự và số lượng dòng  
- Mỗi dòng input → một dòng output tương ứng
- Không thêm số thứ tự, không giải thích
- CHỈ trả về bản dịch, không bao gồm instruction"""
        else:
            instruction = f"""Dịch CHÍNH XÁC từng câu sau sang tiếng Việt.

CONTEXT: {context_str}

{lang_rules}

BATCH RULES:
- Dịch từng dòng một cách riêng biệt
- Giữ nguyên thứ tự và số lượng dòng
- Mỗi dòng input → một dòng output tương ứng
- Không thêm số thứ tự, không giải thích
- Tên riêng/ký hiệu: giữ nguyên
- Trả về định dạng: mỗi bản dịch trên 1 dòng, cách nhau bởi \\n"""
        
        # Format texts với số thứ tự để dễ parse
        numbered_texts = []
        for i, text in enumerate(texts, 1):
            numbered_texts.append(f"{i}. {text}")
        
        full_prompt = f"""{instruction}

TEXTS TO TRANSLATE:
{chr(10).join(numbered_texts)}

TRANSLATED RESULTS (one per line):"""
        
        return full_prompt
    
    def _translate_batch_with_gemini(self, prompt: str, api_key: str) -> str:
        """
        Translate batch prompt with specific API key
        """
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': api_key
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                print(f"❌ Gemini batch API error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Batch translation error: {e}")
            return ""
    
    def _parse_batch_result(self, batch_result: str, expected_count: int) -> List[str]:
        """
        Parse kết quả batch translation thành list individual results
        """
        if not batch_result:
            return []
        
        # Clean up response
        cleaned = batch_result.strip()
        
        # Remove any numbering if present
        lines = cleaned.split('\n')
        parsed_results = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering if present (1. 2. etc.)
            import re
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Remove quotes if present
            line = line.strip('"\'')
            
            if line:
                parsed_results.append(line)
        
        # Ensure we have the right number of results
        while len(parsed_results) < expected_count:
            parsed_results.append("")
        
        return parsed_results[:expected_count]
    
    def _fallback_individual_translate(self, texts: List[str], source_lang: str, 
                                     context: dict, custom_prompt: str) -> List[str]:
        """
        Fallback: translate individual texts khi batch fail
        """
        results = []
        
        # Get API key từ manager hoặc fallback
        api_key = None
        if self.api_key_manager:
            api_key = self.api_key_manager.get_api_key('gemini')
        
        if not api_key:
            api_key = self.gemini_api_key
        
        for text in texts:
            if text and text.strip():
                if api_key:
                    translated = self._translate_with_gemini_direct(text, source_lang, context, custom_prompt, api_key)
                else:
                    translated = self._translate_with_google(text, source_lang)
                results.append(translated)
            else:
                results.append("")
        
        return results
    
    def _translate_with_gemini_direct(self, text: str, source_lang: str, context: dict, 
                                    custom_prompt: str, api_key: str) -> str:
        """
        Direct Gemini translation với specific API key
        """
        prompt = self._get_translation_prompt(text, source_lang, context, custom_prompt)
        result = self._translate_batch_with_gemini(prompt, api_key)
        
        if result:
            return self._clean_gemini_response(result)
        else:
            return self._translate_with_google(text, source_lang)
    
    def get_api_key_status(self) -> Dict:
        """
        Lấy status của API keys
        """
        if not self.api_key_manager:
            return {"api_manager": False, "gemini_status": "No manager"}
        
        return {
            "api_manager": True,
            "gemini_status": self.api_key_manager.get_key_status('gemini')
        }
    
    def test_gemini_translation(self, test_text="こんにちは", source_lang="ja") -> Dict:
        """
        Kiểm tra xem dịch thuật bằng Gemini có hoạt động đúng không
        
        Args:
            test_text (str): Text để test (mặc định: "こんにちは" - "Xin chào" bằng tiếng Nhật)
            source_lang (str): Ngôn ngữ nguồn (mặc định: "ja" - tiếng Nhật)
            
        Returns:
            Dict: Kết quả test bao gồm:
                - success (bool): Có thành công không
                - translation (str): Kết quả dịch 
                - error (str): Lỗi nếu có
                - api_key_available (bool): API key có sẵn không
                - response_time (float): Thời gian phản hồi (giây)
                - fallback_used (bool): Có sử dụng fallback không
        """
        import time
        
        print(f"🔍 Testing Gemini translation with: '{test_text}' ({source_lang} → vi)")
        
        # Kiểm tra API key
        api_key_available = bool(self.gemini_api_key)
        if self.api_key_manager:
            manager_key = self.api_key_manager.get_api_key('gemini')
            api_key_available = api_key_available or bool(manager_key)
        
        result = {
            "success": False,
            "translation": "",
            "error": "",
            "api_key_available": api_key_available,
            "response_time": 0.0,
            "fallback_used": False,
            "test_input": test_text,
            "source_language": source_lang
        }
        
        if not api_key_available:
            result["error"] = "Không có API key Gemini được cấu hình"
            result["success"] = False
            print("❌ Gemini API key không có sẵn")
            return result
        
        try:
            start_time = time.time()
            
            # Test với context đơn giản
            test_context = {
                "gender": "neutral",
                "relationship": "friend", 
                "formality": "casual"
            }
            
            # Thử dịch bằng Gemini
            translation = self.translate(
                text=test_text,
                method="gemini",
                source_lang=source_lang,
                context=test_context
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            result["response_time"] = round(response_time, 2)
            result["translation"] = translation
            
            # Kiểm tra xem có fallback về Google Translate không
            if translation == test_text:
                # Nếu kết quả giống input, có thể là lỗi
                result["error"] = "Kết quả dịch giống với input, có thể có lỗi"
                result["success"] = False
                result["fallback_used"] = True
            elif not translation or translation.strip() == "":
                # Nếu kết quả rỗng
                result["error"] = "Kết quả dịch rỗng"
                result["success"] = False
            else:
                # Dịch thành công
                result["success"] = True
                
                # Kiểm tra xem có phải là kết quả từ Google Translate không
                # (thử dịch cùng text bằng Google để so sánh)
                try:
                    google_translation = self._translate_with_google(test_text, source_lang)
                    if translation.lower().strip() == google_translation.lower().strip():
                        result["fallback_used"] = True
                        result["error"] = "Có thể đã fallback về Google Translate"
                    else:
                        result["fallback_used"] = False
                except:
                    # Không thể so sánh với Google, vẫn coi là thành công
                    pass
            
            print(f"✅ Test hoàn thành trong {response_time:.2f}s")
            print(f"📝 Kết quả: '{translation}'")
            
            if result["fallback_used"]:
                print("⚠️ Có thể đã sử dụng fallback")
            
        except Exception as e:
            result["error"] = f"Lỗi khi test: {str(e)}"
            result["success"] = False
            print(f"❌ Test thất bại: {e}")
        
        return result
    
    def run_comprehensive_gemini_test(self) -> Dict:
        """
        Chạy bộ test toàn diện cho Gemini translation
        
        Returns:
            Dict: Kết quả test chi tiết cho nhiều trường hợp
        """
        print("🧪 Bắt đầu test toàn diện cho Gemini translation...")
        
        # Các test cases
        test_cases = [
            {"text": "こんにちは", "lang": "ja", "name": "Tiếng Nhật cơ bản"},
            {"text": "你好", "lang": "zh", "name": "Tiếng Trung cơ bản"}, 
            {"text": "안녕하세요", "lang": "ko", "name": "Tiếng Hàn cơ bản"},
            {"text": "Hello", "lang": "en", "name": "Tiếng Anh cơ bản"},
            {"text": "ありがとうございます", "lang": "ja", "name": "Tiếng Nhật lịch sự"},
            {"text": "私は学生です", "lang": "ja", "name": "Tiếng Nhật câu dài"},
            {"text": "バン！", "lang": "ja", "name": "SFX tiếng Nhật"},
            {"text": "すごい...", "lang": "ja", "name": "Thought bubble"},
        ]
        
        results = {
            "overall_success": True,
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "summary": "",
            "recommendations": []
        }
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}/{len(test_cases)}: {test_case['name']}")
            
            # Thêm context tùy theo loại test
            context = {"formality": "casual"}
            if "SFX" in test_case['name']:
                context["is_sfx"] = True
            elif "thought" in test_case['name'].lower():
                context["is_thought"] = True
            elif "lịch sự" in test_case['name']:
                context["formality"] = "polite"
            
            test_result = self.test_gemini_translation(
                test_text=test_case["text"],
                source_lang=test_case["lang"]
            )
            
            test_result["test_name"] = test_case["name"]
            test_result["test_number"] = i
            
            results["test_results"].append(test_result)
            
            if test_result["success"]:
                results["passed_tests"] += 1
                print(f"✅ {test_case['name']}: PASSED")
            else:
                results["failed_tests"] += 1
                results["overall_success"] = False
                print(f"❌ {test_case['name']}: FAILED - {test_result['error']}")
        
        # Tạo summary
        success_rate = (results["passed_tests"] / results["total_tests"]) * 100
        results["success_rate"] = round(success_rate, 1)
        
        if results["overall_success"]:
            results["summary"] = f"🎉 TẤT CẢ TESTS PASSED! ({results['passed_tests']}/{results['total_tests']})"
        elif success_rate >= 70:
            results["summary"] = f"⚠️ Một số tests failed ({results['passed_tests']}/{results['total_tests']} - {success_rate}%)"
        else:
            results["summary"] = f"❌ Nhiều tests failed ({results['passed_tests']}/{results['total_tests']} - {success_rate}%)"
        
        # Recommendations
        if not results["overall_success"]:
            if not any(r["api_key_available"] for r in results["test_results"]):
                results["recommendations"].append("🔑 Cần cấu hình Gemini API key")
            
            fallback_count = sum(1 for r in results["test_results"] if r.get("fallback_used"))
            if fallback_count > 0:
                results["recommendations"].append(f"⚠️ {fallback_count} tests sử dụng fallback - kiểm tra API key hoặc network")
            
            slow_tests = [r for r in results["test_results"] if r.get("response_time", 0) > 10]
            if slow_tests:
                results["recommendations"].append(f"🐌 {len(slow_tests)} tests chậm (>10s) - kiểm tra network")
        
        print(f"\n{results['summary']}")
        for rec in results["recommendations"]:
            print(f"  {rec}")
        
        return results
