# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
MangaTranslator - Fixed version with cache system
No emoji characters to avoid encoding issues
"""

# Core modules
from add_text import add_text
from detect_bubbles import detect_bubbles  
from process_bubble import process_bubble
from translator import MangaTranslator
from multi_ocr import MultiLanguageOCR
from api_key_manager import ApiKeyManager
from font_manager import FontManager

# External libraries
from PIL import Image
import gradio as gr
import numpy as np
import os
import tempfile
import time
import atexit
import zipfile
import shutil
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Configuration constants
MODEL = "model.pt"  
EXAMPLE_LIST = [["examples/0.png"], ["examples/ex0.png"]]
OUTPUT_DIR = "outputs"
CACHE_DIR = "cache"

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize global managers
print("🚀 Initializing MangaTranslator components...")

try:
    api_key_manager = ApiKeyManager()
    print("✅ API Key Manager initialized")
except Exception as e:
    print(f"⚠️ API Key Manager failed: {e}")
    api_key_manager = None

try:
    font_manager = FontManager()
    print("✅ Font Manager initialized")
except Exception as e:
    print(f"⚠️ Font Manager failed: {e}")
    font_manager = None

class ImageCache:
    """Cache manager for processed images"""
    
    def __init__(self):
        self.cache = {}
        self.session_data = {}
    
    def store_session_images(self, session_id, images_data):
        """Store processed images in session cache"""
        self.session_data[session_id] = {
            'images': images_data,
            'timestamp': datetime.now(),
            'total_count': len(images_data),
            'successful_count': len([img for img in images_data if img['status'] == 'success'])
        }
        print(f"Cached {len(images_data)} images for session {session_id}")
    
    def get_session_data(self, session_id):
        """Get session data from cache"""
        return self.session_data.get(session_id, None)
    
    def create_zip_from_cache(self, session_id):
        """Create ZIP file from cached images"""
        session_data = self.get_session_data(session_id)
        if not session_data:
            return None
        
        successful_images = [img for img in session_data['images'] if img['status'] == 'success']
        if not successful_images:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"manga_translated_{timestamp}_{session_id[:8]}.zip"
        zip_path = os.path.join(CACHE_DIR, zip_filename)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for img_data in successful_images:
                    # Convert PIL image to bytes
                    img_bytes = BytesIO()
                    img_data['image'].save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    # Add to ZIP
                    zipf.writestr(img_data['output_name'], img_bytes.getvalue())
            
            print(f"Created ZIP from cache: {zip_path}")
            return zip_path
        except Exception as e:
            print(f"Error creating ZIP from cache: {e}")
            return None
    
    def clear_old_sessions(self, max_age_hours=2):
        """Clear old session data to free memory"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            session_id for session_id, data in self.session_data.items()
            if data['timestamp'] < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.session_data[session_id]
            print(f"Cleared old session: {session_id}")

# Global cache instance
image_cache = ImageCache()

def cleanup_debug_files():
    """Clean up temporary debug files on exit"""
    debug_dir = os.path.join(tempfile.gettempdir(), "manga_translator_debug")
    if os.path.exists(debug_dir):
        try:
            shutil.rmtree(debug_dir)
            print(f"Cleaned up debug directory: {debug_dir}")
        except Exception as e:
            print(f"Could not clean debug directory: {e}")
    
    # Also cleanup cache directory
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Cleaned up cache directory: {CACHE_DIR}")
        except Exception as e:
            print(f"Could not clean cache directory: {e}")

# Register cleanup function to run on exit
atexit.register(cleanup_debug_files)

def process_single_image(img, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """Process a single image for translation with optimized batch processing"""
    
    # Set default values if None
    if translation_method is None:
        translation_method = "google"
        
    # Auto-detect font nếu font_path là None hoặc không hợp lệ
    if font_path is None or not os.path.exists(font_path):
        if font_manager:
            font_path = font_manager.get_font_path("manga")
            print(f"🎯 Auto-selected font: {font_path}")
        else:
            font_path = "fonts/animeace_i.ttf"  # fallback

    # Handle API key - use from input or environment variable
    if not gemini_api_key or gemini_api_key.strip() == "":
        gemini_api_key = os.getenv("GEMINI_API_KEY", None)
    
    # Handle custom prompt
    if custom_prompt and custom_prompt.strip():
        print(f"Using custom prompt: {custom_prompt[:50]}")
    else:
        custom_prompt = None
        print("Using automatic prompt based on source language")
    
    # Debug logging
    print(f"Using translation method: {translation_method}")
    print(f"Source language: {source_language}")
    print(f"Font: {os.path.basename(font_path) if font_path else 'Default'}")
    print(f"API key available: {'Yes' if gemini_api_key else 'No'}")

    # Step 1: Detect text bubbles using YOLO model with optimized filtering
    results = detect_bubbles(
        MODEL, 
        img, 
        conf_threshold=0.3,  # Slightly higher confidence threshold
        iou_threshold=0.4,   # Remove overlapping bubbles
        enable_nms=True      # Enable Non-Maximum Suppression
    )
    print(f"Detected {len(results)} filtered bubbles")
    
    # Results are already sorted by detect_bubbles function (top to bottom, then by confidence)
    # No need to sort again

    # Step 2: Initialize translator with optional Gemini API key
    manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
    
    # Step 3: Initialize multi-language OCR system
    multi_ocr = MultiLanguageOCR()
    
    # Show OCR recommendation for selected language
    ocr_method, ocr_desc = multi_ocr.get_best_ocr_for_language(source_language)
    print(f"OCR Engine: {ocr_desc}")

    # Convert PIL image to numpy array for processing
    original_image = np.array(img)
    image = original_image.copy()

    # ============================================================================
    # OPTIMIZED BATCH PROCESSING - OCR all bubbles first, then translate in batch
    # ============================================================================
    
    print("🔄 Starting optimized batch OCR + Translation...")
    
    # Phase 1: Extract all text from all bubbles (batch OCR)
    bubble_data = []
    
    for idx, result in enumerate(results):
        x1, y1, x2, y2, score, class_id = result
        print(f"OCR bubble {idx+1}/{len(results)}")

        # Extract the bubble region from ORIGINAL image
        detected_image = original_image[int(y1):int(y2), int(x1):int(x2)]

        # Convert to PIL Image for OCR processing
        im = Image.fromarray(np.uint8(detected_image))
        
        # Extract text using appropriate OCR engine
        text = multi_ocr.extract_text(im, source_language, method="auto")
        text = text.strip() if text else ""
        
        print(f"OCR Text {idx+1}: '{text}'")

        # Process the bubble for text replacement
        working_bubble = image[int(y1):int(y2), int(x1):int(x2)]
        processed_bubble, cont = process_bubble(working_bubble)

        # Store bubble data for batch translation
        bubble_data.append({
            'index': idx,
            'bbox': (x1, y1, x2, y2),
            'text': text,
            'processed_bubble': processed_bubble,
            'contour': cont
        })
    
    # Phase 2: Batch translation (translate all texts at once)
    print("🌐 Starting batch translation...")
    
    texts_to_translate = [bubble['text'] for bubble in bubble_data if bubble['text']]
    
    if texts_to_translate:
        if len(texts_to_translate) >= 3:  # Chỉ dùng batch nếu có từ 3 text trở lên
            print(f"📦 Batch translating {len(texts_to_translate)} texts...")
            translated_texts = manga_translator.batch_translate(
                texts_to_translate,
                method=translation_method,
                source_lang=source_language,
                custom_prompt=custom_prompt
            )
        else:
            print(f"📝 Individual translating {len(texts_to_translate)} texts...")
            translated_texts = []
            for text in texts_to_translate:
                translated = manga_translator.translate(
                    text,
                    method=translation_method,
                    source_lang=source_language,
                    custom_prompt=custom_prompt
                )
                translated_texts.append(translated)
    else:
        translated_texts = []
    
    # Phase 3: Apply translated text back to image
    print("🎨 Applying translated text to image...")
    
    translated_index = 0
    for bubble in bubble_data:
        if bubble['text']:
            if translated_index < len(translated_texts):
                text_translated = translated_texts[translated_index]
                translated_index += 1
            else:
                text_translated = bubble['text']  # fallback
        else:
            text_translated = ""
        
        print(f"Adding text to bubble {bubble['index']+1}: '{text_translated}'")
        
        # Add translated text back to the image
        x1, y1, x2, y2 = bubble['bbox']
        image[int(y1):int(y2), int(x1):int(x2)] = add_text(
            bubble['processed_bubble'], 
            text_translated, 
            font_path, 
            bubble['contour']
        )

    return Image.fromarray(image)

def process_mega_batch_cached(images, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """
    🚀 MEGA BATCH PROCESSING: Process multiple images with SINGLE API call for ALL texts
    Tối ưu: 10 ảnh → 1 API call thay vì 10 API calls
    """
    
    if not images:
        return None, [], "No images uploaded"
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    total_images = len(images)
    
    print(f"🚀 Starting MEGA BATCH processing: {total_images} images")
    print(f"📋 Session ID: {session_id}")
    print(f"🎯 Strategy: ALL images → 1 API call")
    
    # Clean old cache to free memory
    image_cache.clear_old_sessions()
    
    # Auto-detect font nếu font_path là None hoặc không hợp lệ
    if font_path is None or not os.path.exists(font_path):
        if font_manager:
            font_path = font_manager.get_font_path("manga")
            print(f"🎯 Auto-selected font: {font_path}")
        else:
            font_path = "fonts/animeace_i.ttf"  # fallback

    # Handle API key - use from input or environment variable
    if not gemini_api_key or gemini_api_key.strip() == "":
        gemini_api_key = os.getenv("GEMINI_API_KEY", None)
    
    # Handle custom prompt
    if custom_prompt and custom_prompt.strip():
        print(f"Using custom prompt: {custom_prompt[:50]}")
    else:
        custom_prompt = None
        print("Using automatic prompt based on source language")
    
    print(f"Using translation method: {translation_method}")
    print(f"Source language: {source_language}")
    print(f"Font: {os.path.basename(font_path) if font_path else 'Default'}")
    print(f"API key available: {'Yes' if gemini_api_key else 'No'}")

    # ============================================================================
    # MEGA BATCH PHASE 1: Collect ALL bubbles from ALL images
    # ============================================================================
    
    print("📊 PHASE 1: Collecting ALL bubbles from ALL images...")
    
    all_images_data = []  # Store all image processing data
    all_texts_to_translate = []  # All texts from all images
    text_to_image_mapping = []  # Track which text belongs to which image/bubble
    
    # Initialize components
    manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
    multi_ocr = MultiLanguageOCR()
    
    # Show OCR recommendation for selected language
    ocr_method, ocr_desc = multi_ocr.get_best_ocr_for_language(source_language)
    print(f"OCR Engine: {ocr_desc}")
    
    for img_idx, img_file in enumerate(images):
        try:
            print(f"🔍 Collecting bubbles from image {img_idx + 1}/{total_images}")
            
            # Open image
            if isinstance(img_file, str):
                img = Image.open(img_file)
                original_name = os.path.basename(img_file)
            else:
                img = Image.open(img_file.name)
                original_name = img_file.name if hasattr(img_file, 'name') else f"image_{img_idx+1}.png"
            
            # Convert PIL image to numpy array for processing
            original_image = np.array(img)
            image = original_image.copy()
            
            # Step 1: Detect text bubbles using YOLO model with optimized filtering
            results = detect_bubbles(
                MODEL, 
                img, 
                conf_threshold=0.3,  # Slightly higher confidence threshold
                iou_threshold=0.4,   # Remove overlapping bubbles
                enable_nms=True      # Enable Non-Maximum Suppression
            )
            print(f"Detected {len(results)} filtered bubbles in {original_name}")
            
            # Results are already sorted by detect_bubbles function
            # No need to sort again
            
            # Extract text from all bubbles in this image
            image_bubble_data = []
            
            for bubble_idx, result in enumerate(results):
                x1, y1, x2, y2, score, class_id = result
                
                # Extract the bubble region from ORIGINAL image
                detected_image = original_image[int(y1):int(y2), int(x1):int(x2)]
                
                # Convert to PIL Image for OCR processing
                im = Image.fromarray(np.uint8(detected_image))
                
                # Extract text using appropriate OCR engine
                text = multi_ocr.extract_text(im, source_language, method="auto")
                text = text.strip() if text else ""
                
                # Process the bubble for text replacement
                working_bubble = image[int(y1):int(y2), int(x1):int(x2)]
                processed_bubble, cont = process_bubble(working_bubble)
                
                # Store bubble data
                bubble_data = {
                    'bbox': (x1, y1, x2, y2),
                    'text': text,
                    'processed_bubble': processed_bubble,
                    'contour': cont,
                    'bubble_index': bubble_idx
                }
                image_bubble_data.append(bubble_data)
                
                # Add to global text collection if has text
                if text:
                    text_index = len(all_texts_to_translate)
                    all_texts_to_translate.append(text)
                    text_to_image_mapping.append({
                        'image_index': img_idx,
                        'bubble_index': bubble_idx,
                        'text_index': text_index,
                        'original_text': text
                    })
                    print(f"📝 Text {text_index + 1}: '{text}'")
            
            # Store image data
            image_data = {
                'image_index': img_idx,
                'original_name': original_name,
                'pil_image': img,
                'numpy_image': image,
                'original_image': original_image,
                'bubble_data': image_bubble_data
            }
            all_images_data.append(image_data)
            
        except Exception as e:
            error_msg = f"Error collecting bubbles from image {img_idx+1}: {str(e)}"
            print(error_msg)
            
            # Store error info
            error_data = {
                'image_index': img_idx,
                'original_name': img_file.name if hasattr(img_file, 'name') else f'image_{img_idx+1}',
                'error': str(e),
                'bubble_data': []
            }
            all_images_data.append(error_data)
    
    # ============================================================================
    # MEGA BATCH PHASE 2: Single API call for ALL texts
    # ============================================================================
    
    print(f"🌐 PHASE 2: MEGA BATCH translation - {len(all_texts_to_translate)} texts in 1 API call!")
    
    if all_texts_to_translate:
        if len(all_texts_to_translate) >= 1:  # Always use batch for mega batch
            print(f"🎯 MEGA BATCH: Translating {len(all_texts_to_translate)} texts with 1 API call")
            
            if translation_method == "gemini":
                # Use mega batch translation
                translated_texts = manga_translator.batch_translate(
                    all_texts_to_translate,
                    method=translation_method,
                    source_lang=source_language,
                    custom_prompt=custom_prompt
                )
            else:
                # For non-Gemini methods, still do individual calls
                translated_texts = []
                for text in all_texts_to_translate:
                    translated = manga_translator.translate(
                        text,
                        method=translation_method,
                        source_lang=source_language,
                        custom_prompt=custom_prompt
                    )
                    translated_texts.append(translated)
        else:
            translated_texts = []
    else:
        translated_texts = []
        print("⚠️ No texts found to translate")
    
    # ============================================================================
    # MEGA BATCH PHASE 3: Apply translated texts back to images
    # ============================================================================
    
    print(f"🎨 PHASE 3: Applying {len(translated_texts)} translations to images...")
    
    processed_images = []
    preview_images = []
    
    for image_data in all_images_data:
        try:
            if 'error' in image_data:
                # Handle error case
                error_image_data = {
                    "original_name": image_data['original_name'],
                    "output_name": "N/A",
                    "image": None,
                    "status": "error",
                    "error_message": image_data['error'][:100],
                    "index": image_data['image_index']
                }
                processed_images.append(error_image_data)
                continue
            
            img_idx = image_data['image_index']
            original_name = image_data['original_name']
            image = image_data['numpy_image'].copy()
            
            print(f"🖼️ Applying translations to image {img_idx + 1}: {original_name}")
            
            # Apply translations to each bubble
            for bubble_data in image_data['bubble_data']:
                bubble_idx = bubble_data['bubble_index']
                
                # Find translation for this bubble
                translated_text = ""
                for mapping in text_to_image_mapping:
                    if (mapping['image_index'] == img_idx and 
                        mapping['bubble_index'] == bubble_idx):
                        text_idx = mapping['text_index']
                        if text_idx < len(translated_texts):
                            translated_text = translated_texts[text_idx]
                        break
                
                if not translated_text:
                    translated_text = bubble_data['text']  # fallback to original
                
                print(f"  🎯 Bubble {bubble_idx + 1}: '{translated_text}'")
                
                # Add translated text back to the image
                x1, y1, x2, y2 = bubble_data['bbox']
                image[int(y1):int(y2), int(x1):int(x2)] = add_text(
                    bubble_data['processed_bubble'], 
                    translated_text, 
                    font_path, 
                    bubble_data['contour']
                )
            
            # Convert back to PIL Image
            processed_img = Image.fromarray(image)
            
            # Generate output filename
            base_name = os.path.splitext(original_name)[0]
            output_filename = f"{base_name}_translated.png"
            
            # Store in cache (in memory)
            success_image_data = {
                "original_name": original_name,
                "output_name": output_filename,
                "image": processed_img,
                "status": "success",
                "index": img_idx
            }
            processed_images.append(success_image_data)
            
            # Add to preview list (for Gradio Gallery)
            preview_images.append(processed_img)
            
            print(f"✅ Successfully processed: {original_name}")
            
        except Exception as e:
            error_msg = f"Error applying translations to image {img_idx+1}: {str(e)}"
            print(error_msg)
            
            # Store error info
            error_image_data = {
                "original_name": image_data.get('original_name', f'image_{img_idx+1}'),
                "output_name": "N/A",
                "image": None,
                "status": "error",
                "error_message": str(e)[:100],
                "index": img_idx
            }
            processed_images.append(error_image_data)
    
    # Store session data in cache
    image_cache.store_session_images(session_id, processed_images)
    
    # Generate status message
    successful_count = len([img for img in processed_images if img['status'] == 'success'])
    failed_count = total_images - successful_count
    
    if failed_count == 0:
        status_msg = f"🎉 MEGA BATCH Complete! {successful_count}/{total_images} images with 1 API call!"
    else:
        status_msg = f"MEGA BATCH Complete with errors! Success: {successful_count}, Failed: {failed_count}"
    
    print(f"🏁 {status_msg}")
    print(f"💰 API Efficiency: {len(all_texts_to_translate)} texts → 1 API call (vs {total_images} calls)")
    
    return session_id, preview_images, status_msg

def process_batch_cached(images, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """Process multiple images in batch and store in cache (LEGACY - use mega batch instead)"""
    
    if not images:
        return None, [], "No images uploaded"
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    total_images = len(images)
    
    print(f"Starting cached batch processing: {total_images} images")
    print(f"Session ID: {session_id}")
    
    # Clean old cache to free memory
    image_cache.clear_old_sessions()
    
    processed_images = []
    preview_images = []
    
    for idx, img_file in enumerate(images):
        try:
            print(f"Processing image {idx + 1}/{total_images}")
            
            # Open image
            if isinstance(img_file, str):
                img = Image.open(img_file)
                original_name = os.path.basename(img_file)
            else:
                img = Image.open(img_file.name)
                original_name = img_file.name if hasattr(img_file, 'name') else f"image_{idx+1}.png"
            
            # Process the image using existing function
            processed_img = process_single_image(
                img, translation_method, font_path, 
                source_language, gemini_api_key, custom_prompt
            )
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(original_name))[0]
            output_filename = f"{base_name}_translated.png"
            
            # Store in cache (in memory)
            image_data = {
                "original_name": original_name,
                "output_name": output_filename,
                "image": processed_img,
                "status": "success",
                "index": idx
            }
            processed_images.append(image_data)
            
            # Add to preview list (for Gradio Gallery)
            preview_images.append(processed_img)
            
            print(f"Successfully processed: {original_name}")
            
        except Exception as e:
            error_msg = f"Error processing image {idx+1}: {str(e)}"
            print(error_msg)
            
            # Store error info
            image_data = {
                "original_name": img_file.name if hasattr(img_file, 'name') else f'image_{idx+1}',
                "output_name": "N/A",
                "image": None,
                "status": "error",
                "error_message": str(e)[:100],
                "index": idx
            }
            processed_images.append(image_data)
    
    # Store session data in cache
    image_cache.store_session_images(session_id, processed_images)
    
    # Generate status message
    successful_count = len([img for img in processed_images if img['status'] == 'success'])
    failed_count = total_images - successful_count
    
    if failed_count == 0:
        status_msg = f"Complete! Successfully processed {successful_count}/{total_images} images"
    else:
        status_msg = f"Complete with errors! Success: {successful_count}, Failed: {failed_count}"
    
    return session_id, preview_images, status_msg

def create_file_list_display_cached(session_id):
    """Create HTML display for cached processed files list"""
    
    session_data = image_cache.get_session_data(session_id)
    if not session_data:
        return "<p>Session not found or expired</p>"
    
    images_data = session_data['images']
    total_count = session_data['total_count']
    successful_count = session_data['successful_count']
    
    html = f"""
    <div style="; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">Processed Files List</h3>
        <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <strong>Summary:</strong> {successful_count}/{total_count} images successful | Session: {session_id[:8]}
        </div>
    """
    
    for idx, img_data in enumerate(images_data, 1):
        if img_data['status'] == 'success':
            status_color = "#28a745"
            status_text = "Success"
        else:
            status_color = "#dc3545"
            status_text = f"Error: {img_data.get('error_message', 'Unknown error')}"
        
        html += f"""
        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid {status_color};">
            <div>
                <strong>#{idx}:</strong> {img_data['original_name']} -> {img_data['output_name']}
                <br>
                <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
            </div>
        </div>
        """
    
    if successful_count > 0:
        html += f"""
        <div style="background: #e8f5e8; padding: 15px; margin: 15px 0; border-radius: 8px; text-align: center;">
            <h4 style="color: #2e7d32; margin-bottom: 10px;">Ready to Download</h4>
            <p>Found <strong>{successful_count}</strong> successfully processed images</p>
            <p><em>Click "Create ZIP" button below to download all</em></p>
        </div>
        """
    
    html += "</div>"
    return html

def create_zip_download(session_id):
    """Create ZIP file from cached images when user requests download"""
    
    if not session_id:
        return None, "No session to create ZIP"
    
    zip_path = image_cache.create_zip_from_cache(session_id)
    if zip_path:
        return zip_path, "ZIP file created successfully! Ready to download."
    else:
        return None, "Cannot create ZIP file. Check session or no successful images."

def batch_predict(images, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """
    🚀 MEGA BATCH prediction function for multiple images 
    Tối ưu: N ảnh → 1 API call cho tất cả texts
    """
    
    session_id, preview_images, status_msg = process_mega_batch_cached(
        images, translation_method, font_path, 
        source_language, gemini_api_key, custom_prompt
    )
    
    if session_id:
        file_list_html = create_file_list_display_cached(session_id)
    else:
        file_list_html = "<p>Cannot process batch</p>"
    
    return session_id, preview_images, file_list_html, status_msg

def batch_predict_legacy(images, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """Legacy batch prediction function (1 API call per image)"""
    
    session_id, preview_images, status_msg = process_batch_cached(
        images, translation_method, font_path, 
        source_language, gemini_api_key, custom_prompt
    )
    
    if session_id:
        file_list_html = create_file_list_display_cached(session_id)
    else:
        file_list_html = "<p>Cannot process batch</p>"
    
    return session_id, preview_images, file_list_html, status_msg

# Legacy single image function
def predict(img, translation_method, font_path, source_language="auto", gemini_api_key=None, custom_prompt=None):
    """Main prediction function for manga translation (single image)"""
    return process_single_image(img, translation_method, font_path, source_language, gemini_api_key, custom_prompt)

# UI Configuration
TITLE = "Manga Translator - AI Comic Translation Tool"
DESCRIPTION = """
**🎯 Smart Comic Translation Tool**

**✨ Features:**
- 🤖 **AI Translation**: High-quality manga translation using Gemini AI
- 🚀 **Batch Processing**: Process multiple images efficiently
- 🎨 **Auto Font Selection**: Automatically selects appropriate fonts
- 🔑 **API Key Management**: Manage and test your translation API keys

**� How to Use:**
1. Upload manga pages (PNG, JPG, JPEG)
2. Select source language and translation method  
3. Click translate and download results
"""

def get_font_choices():
    """Get font choices for UI dropdown"""
    if font_manager and font_manager.available_fonts:
        choices = []
        
        # Add categorized fonts
        for font_name, info in font_manager.available_fonts.items():
            display_name = f"{font_name} ({info['recommended_for']})"
            choices.append((display_name, info['path']))
        
        # Add default recommendations first
        recommended = []
        if font_manager.default_fonts.get('manga'):
            manga_font = font_manager.default_fonts['manga']
            if manga_font in font_manager.available_fonts:
                info = font_manager.available_fonts[manga_font]
                recommended.append((f"🎯 {manga_font} (Recommended for Manga)", info['path']))
        
        return recommended + choices
    else:
        # Fallback to hardcoded fonts
        return [
            ("animeace_i (Manga Style)", "fonts/animeace_i.ttf"),
            ("animeace2_reg (Comic Style)", "fonts/animeace2_reg.ttf"),
            ("mangati (Manga Font)", "fonts/mangati.ttf"),
            ("ariali (General)", "fonts/ariali.ttf")
        ]

def get_api_key_status_info():
    """Get current API key status for display"""
    if not api_key_manager:
        return "⚠️ API Key Manager not available"
    
    try:
        status_info = []
        
        # Check Gemini keys
        gemini_keys = api_key_manager.config.get('gemini_keys', [])
        if gemini_keys:
            active_count = len([k for k in gemini_keys if k.get('active', True)])
            total_usage = sum(k.get('usage_count', 0) for k in gemini_keys)
            status_info.append(f"🔑 Gemini: {active_count}/{len(gemini_keys)} active keys, {total_usage} requests today")
        else:
            status_info.append("🔑 Gemini: No keys configured")
        
        return "<br>".join(status_info) if status_info else "No API keys configured"
        
    except Exception as e:
        return f"Error getting API status: {e}"

def test_api_key(api_key, provider="gemini"):
    """Test if an API key is working"""
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    
    try:
        if provider == "gemini":
            # Test Gemini API key
            from translator import MangaTranslator
            translator = MangaTranslator(gemini_api_key=api_key.strip())
            
            # Try a simple translation request
            test_result = translator.translate(
                "Hello", 
                method="gemini",
                source_lang="en"
            )
            
            if test_result and test_result.strip():
                return True, "✅ API key is working"
            else:
                return False, "❌ API key test failed - no response"
                
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "invalid" in error_msg:
            return False, "❌ Invalid API key"
        elif "quota" in error_msg or "limit" in error_msg:
            return False, "❌ API quota exceeded"
        elif "permission" in error_msg:
            return False, "❌ Permission denied"
        else:
            return False, f"❌ Error: {str(e)[:100]}"
    
    return False, "❌ Unknown error occurred"

def add_api_key_to_config(api_key, key_name, provider="gemini"):
    """Add new API key to configuration"""
    if not api_key_manager:
        return False, "API Key Manager not available"
    
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty"
    
    if not key_name or not key_name.strip():
        key_name = f"Key {len(api_key_manager.config.get(f'{provider}_keys', [])) + 1}"
    
    try:
        # Test the key first
        is_valid, test_message = test_api_key(api_key, provider)
        
        if not is_valid:
            return False, f"Key test failed: {test_message}"
        
        # Add to config
        provider_keys = f"{provider}_keys"
        if provider_keys not in api_key_manager.config:
            api_key_manager.config[provider_keys] = []
        
        # Check if key already exists
        existing_keys = [k.get('key', '') for k in api_key_manager.config[provider_keys]]
        if api_key.strip() in existing_keys:
            return False, "This API key already exists"
        
        # Add new key
        new_key = {
            "key": api_key.strip(),
            "name": key_name.strip(),
            "active": True,
            "daily_limit": 1000,
            "usage_count": 0,
            "last_reset": datetime.now().strftime("%Y-%m-%d")
        }
        
        api_key_manager.config[provider_keys].append(new_key)
        api_key_manager.save_config()
        
        return True, f"✅ Successfully added '{key_name}' - {test_message}"
        
    except Exception as e:
        return False, f"Error adding key: {str(e)}"

def remove_api_key_from_config(key_index, provider="gemini"):
    """Remove API key from configuration"""
    if not api_key_manager:
        return False, "API Key Manager not available"
    
    try:
        provider_keys = f"{provider}_keys"
        keys_list = api_key_manager.config.get(provider_keys, [])
        
        if 0 <= key_index < len(keys_list):
            removed_key = keys_list.pop(key_index)
            api_key_manager.save_config()
            return True, f"✅ Removed key '{removed_key.get('name', 'Unknown')}'"
        else:
            return False, "Invalid key index"
            
    except Exception as e:
        return False, f"Error removing key: {str(e)}"

def get_api_keys_list(provider="gemini"):
    """Get list of API keys for display"""
    if not api_key_manager:
        return "API Key Manager not available"
    
    try:
        provider_keys = f"{provider}_keys"
        keys_list = api_key_manager.config.get(provider_keys, [])
        
        if not keys_list:
            return "No API keys configured"
        
        html = "<div style='font-family: monospace;'>"
        for i, key_info in enumerate(keys_list):
            status = "🟢" if key_info.get('active', True) else "🔴"
            name = key_info.get('name', f'Key {i+1}')
            usage = key_info.get('usage_count', 0)
            limit = key_info.get('daily_limit', 1000)
            key_preview = key_info.get('key', '')[:10] + "..." if key_info.get('key', '') else "No key"
            
            html += f"""
            <div style='; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #007bff;'>
                <strong>{status} {name}</strong><br>
                <small>Key: {key_preview} | Usage: {usage}/{limit}</small>
            </div>
            """
        
        html += "</div>"
        return html
        
    except Exception as e:
        return f"Error loading keys: {str(e)}"

def get_api_key_status_info():
    """Get API key status for display"""
    if api_key_manager:
        status = api_key_manager.get_key_status('gemini')
        return f"📊 API Status: {status['available']}/{status['total']} keys available"
    else:
        return "⚠️ API Key Manager not initialized"

# Create Gradio interface with tabs for single and batch processing
with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #2c3e50; margin-bottom: 10px;">{TITLE}</h1>
    </div>
    """)
    
    gr.Markdown(DESCRIPTION)
    
    with gr.Tabs():
        # Tab 1: Single Image Processing
        with gr.TabItem("🖼️ Single Image"):
            with gr.Row():
                with gr.Column():
                    single_image_input = gr.Image(type="pil", label="Upload manga image")
                    
                    # Configuration inputs
                    translation_method = gr.Dropdown(
                        [("Google Translate", "google"),
                         ("Gemini AI (Recommended)", "gemini"),
                         ("Helsinki-NLP (JP->EN)", "hf"),
                         ("Bing", "bing")],
                        label="Translation Method",
                        value="gemini"
                    )
                    
                    # Dynamic font dropdown
                    font_choices = get_font_choices()
                    font_path = gr.Dropdown(
                        font_choices,
                        label="Font",
                        value=font_choices[0][1] if font_choices else "fonts/animeace_i.ttf"
                    )
                    
                    source_language = gr.Dropdown(
                        [("Auto Detect", "auto"),
                         ("Japanese", "ja"),
                         ("Chinese", "zh"),
                         ("Korean", "ko"),
                         ("English", "en")],
                        label="Source Language",
                        value="auto"
                    )
                    
                    gemini_api_key = gr.Textbox(
                        label="Gemini API Key (Optional)", 
                        type="password", 
                        placeholder="Enter API key for AI translation (leave blank if configured)",
                        value=""
                    )
                    
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt (Advanced)", 
                        lines=3,
                        placeholder="Leave blank for automatic prompt based on source language",
                        value=""
                    )
                    
                    single_submit_btn = gr.Button("Translate", variant="primary")
                
                with gr.Column():
                    single_output = gr.Image(label="Translation Result")
            
            # Examples for single image
            gr.Examples(
                examples=[[ex[0]] for ex in EXAMPLE_LIST],
                inputs=[single_image_input],
                label="Sample Images"
            )
        
        # Tab 2: Batch Processing
        with gr.TabItem("📚 Batch Processing"):
            # Mode selection
            with gr.Row():
                processing_mode = gr.Radio(
                    [("🚀 MEGA BATCH (1 API Call for ALL)", "mega"),
                     ("📝 Legacy Batch (1 API Call per Image)", "legacy")],
                    label="Processing Mode",
                    value="mega",
                    info="MEGA BATCH: More efficient, saves 90%+ API quota | Legacy: More stable, one call per image"
                )
            
            # Info boxes for each mode
            with gr.Row():
                mega_info = gr.HTML("""
                <div style="background: linear-gradient(135deg, #4CAF50, #45a049); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0;">🚀 MEGA BATCH Mode</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>Ultra Efficient:</strong> All images → 1 API call</li>
                        <li><strong>Cost Savings:</strong> 90%+ less API usage</li>
                        <li><strong>Best for:</strong> Large batches with Gemini AI</li>
                    </ul>
                </div>
                """)
                
                legacy_info = gr.HTML("""
                <div style="background: linear-gradient(135deg, #2196F3, #1976D2); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0 0 10px 0;">📝 Legacy Batch Mode</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>Stable:</strong> One API call per image</li>
                        <li><strong>Compatible:</strong> Works with all translation methods</li>
                        <li><strong>Best for:</strong> Small batches or debugging</li>
                    </ul>
                </div>
                """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    batch_images_input = gr.Files(
                        label="Upload multiple manga images (PNG, JPG, JPEG)",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    
                    # Shared configuration for batch
                    batch_translation_method = gr.Dropdown(
                        [("Google Translate", "google"),
                         ("Gemini AI (Recommended)", "gemini"),
                         ("Helsinki-NLP (JP->EN)", "hf"),
                         ("Bing", "bing")],
                        label="Translation Method",
                        value="gemini"
                    )
                    
                    # Dynamic font dropdown for batch
                    batch_font_choices = get_font_choices()
                    batch_font_path = gr.Dropdown(
                        batch_font_choices,
                        label="Font",
                        value=batch_font_choices[0][1] if batch_font_choices else "fonts/animeace_i.ttf"
                    )
                    
                    batch_source_language = gr.Dropdown(
                        [("Auto Detect", "auto"),
                         ("Japanese", "ja"),
                         ("Chinese", "zh"),
                         ("Korean", "ko"),
                         ("English", "en")],
                        label="Source Language",
                        value="auto"
                    )
                    
                    batch_gemini_api_key = gr.Textbox(
                        label="Gemini API Key (Optional)", 
                        type="password", 
                        placeholder="Enter API key or leave blank if configured",
                        value=""
                    )
                    
                    batch_custom_prompt = gr.Textbox(
                        label="Custom Prompt (Advanced)", 
                        lines=3,
                        placeholder="Leave blank for automatic prompt based on source language",
                        value=""
                    )
                    
                    batch_submit_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        value="Waiting for images..."
                    )
                    
                    # Preview gallery for processed images
                    batch_preview_gallery = gr.Gallery(
                        label="Preview Results",
                        show_label=True,
                        elem_id="preview_gallery",
                        columns=2,
                        rows=2,
                        height="400px",
                        show_share_button=False
                    )
            
            # File list and download section
            with gr.Row():
                with gr.Column():
                    batch_file_list = gr.HTML(
                        label="Processed files list",
                        value="<p>No files processed yet</p>"
                    )
                
                with gr.Column(scale=1):
                    # Hidden session ID storage
                    session_id_state = gr.Textbox(
                        value="",
                        visible=False,
                        interactive=False
                    )
                    
                    create_zip_btn = gr.Button(
                        "Create ZIP",
                        variant="secondary",
                        visible=False
                    )
                    
                    batch_download_zip = gr.File(
                        label="Download ZIP",
                        visible=False
                    )
                    
                    zip_status = gr.Textbox(
                        label="ZIP Status",
                        interactive=False,
                        visible=False
                    )
        
        # Tab 3: API Key Management
        with gr.TabItem("🔑 API Keys"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Add New API Key")
                    
                    new_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter your Gemini API key"
                    )
                    
                    new_key_name = gr.Textbox(
                        label="Key Name",
                        placeholder="Optional: Give this key a name (e.g., 'Personal Account')"
                    )
                    
                    with gr.Row():
                        test_key_btn = gr.Button("🧪 Test Key", variant="secondary")
                        add_key_btn = gr.Button("➕ Add Key", variant="primary")
                    
                    key_test_result = gr.HTML(
                        label="Test Result",
                        value=""
                    )
                    
                    key_add_result = gr.HTML(
                        label="Add Result", 
                        value=""
                    )
                
                with gr.Column():
                    gr.Markdown("### Current API Keys")
                    
                    # API Key status display
                    api_status_display = gr.HTML(
                        value=f"<div style='; padding: 10px; border-radius: 5px;'>{get_api_key_status_info()}</div>"
                    )
                    
                    # Current keys list
                    current_keys_display = gr.HTML(
                        label="API Keys List",
                        value=get_api_keys_list()
                    )
                    
                    refresh_keys_btn = gr.Button("🔄 Refresh", variant="secondary")
                    
                    # Remove key section
                    gr.Markdown("### Remove Key")
                    remove_key_index = gr.Number(
                        label="Key Index to Remove",
                        value=0,
                        precision=0
                    )
                    remove_key_btn = gr.Button("🗑️ Remove Key", variant="stop")
                    remove_result = gr.HTML(value="")
            
            # Examples for single image
            gr.Examples(
                examples=[[ex[0]] for ex in EXAMPLE_LIST],
                inputs=[single_image_input],
                label="Ảnh Mẫu"
            )
    
    # Event handlers
    single_submit_btn.click(
        fn=lambda img, tm, fp, sl, gak, cp: predict(img, tm, fp, sl, gak, cp),
        inputs=[single_image_input, translation_method, font_path, source_language, gemini_api_key, custom_prompt],
        outputs=[single_output]
    )
    
    # Batch processing event handler with mode selection
    def batch_process_with_mode(images, mode, translation_method, font_path, source_language, gemini_api_key, custom_prompt):
        if mode == "mega":
            # Use MEGA BATCH mode
            return batch_predict(images, translation_method, font_path, source_language, gemini_api_key, custom_prompt)
        else:
            # Use Legacy mode
            return batch_predict_legacy(images, translation_method, font_path, source_language, gemini_api_key, custom_prompt)
    
    batch_submit_btn.click(
        fn=batch_process_with_mode,
        inputs=[batch_images_input, processing_mode, batch_translation_method, batch_font_path, batch_source_language, batch_gemini_api_key, batch_custom_prompt],
        outputs=[session_id_state, batch_preview_gallery, batch_file_list, batch_status]
    ).then(
        # Show create ZIP button after processing
        lambda session_id: (
            gr.Button(visible=True if session_id else False),
            gr.Textbox(visible=True if session_id else False)
        ),
        inputs=[session_id_state],
        outputs=[create_zip_btn, zip_status]
    )
    
    # ZIP creation event handler  
    create_zip_btn.click(
        fn=create_zip_download,
        inputs=[session_id_state],
        outputs=[batch_download_zip, zip_status]
    ).then(
        # Show download file component
        lambda zip_path: gr.File(visible=True if zip_path else False),
        inputs=[batch_download_zip],
        outputs=[batch_download_zip]
    )
    
    # API Key Management event handlers
    def test_key_wrapper(api_key):
        is_valid, message = test_api_key(api_key)
        color = "#28a745" if is_valid else "#dc3545"
        return f"<div style='color: {color}; padding: 10px; ; border-radius: 5px;'>{message}</div>"
    
    def add_key_wrapper(api_key, key_name):
        success, message = add_api_key_to_config(api_key, key_name)
        color = "#28a745" if success else "#dc3545"
        result_html = f"<div style='color: {color}; padding: 10px; ; border-radius: 5px;'>{message}</div>"
        
        # Return result and updated displays
        return (
            result_html,
            get_api_keys_list(),
            f"<div style='; padding: 10px; border-radius: 5px;'>{get_api_key_status_info()}</div>",
            "" if success else api_key,  # Clear API key field if successful
            "" if success else key_name   # Clear name field if successful
        )
    
    def remove_key_wrapper(key_index):
        success, message = remove_api_key_from_config(int(key_index))
        color = "#28a745" if success else "#dc3545"
        result_html = f"<div style='color: {color}; padding: 10px; ; border-radius: 5px;'>{message}</div>"
        
        return (
            result_html,
            get_api_keys_list(),
            f"<div style='; padding: 10px; border-radius: 5px;'>{get_api_key_status_info()}</div>"
        )
    
    def refresh_displays():
        return (
            get_api_keys_list(),
            f"<div style='; padding: 10px; border-radius: 5px;'>{get_api_key_status_info()}</div>"
        )
    
    # Test API key
    test_key_btn.click(
        fn=test_key_wrapper,
        inputs=[new_api_key],
        outputs=[key_test_result]
    )
    
    # Add API key
    add_key_btn.click(
        fn=add_key_wrapper,
        inputs=[new_api_key, new_key_name],
        outputs=[key_add_result, current_keys_display, api_status_display, new_api_key, new_key_name]
    )
    
    # Remove API key
    remove_key_btn.click(
        fn=remove_key_wrapper,
        inputs=[remove_key_index],
        outputs=[remove_result, current_keys_display, api_status_display]
    )
    
    # Refresh displays
    refresh_keys_btn.click(
        fn=refresh_displays,
        outputs=[current_keys_display, api_status_display]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(debug=False, share=False)
