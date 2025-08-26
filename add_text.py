#!/usr/bin/env python3
"""
Text Addition Module
===================

Handles the insertion of translated text into processed speech bubbles.
This module provides intelligent text fitting, centering, and formatting
to ensure the translated text fits properly within bubble boundaries.

Features:
- Automatic text wrapping and sizing
- Smart font size adjustment
- Text centering (horizontal and vertical)
- Multi-line text support

Author: MangaTranslator Team
License: MIT
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2


def add_text(image, text, font_path, bubble_contour):
    """
    Add translated text inside a speech bubble with automatic sizing and centering
    
    This function intelligently places translated text within the bubble boundaries,
    automatically adjusting font size and text wrapping to ensure optimal fit.
    
    Args:
        image (numpy.ndarray): Processed bubble image in BGR format (OpenCV)
        text (str): Translated text to place inside the speech bubble
        font_path (str): Path to the font file (.ttf format)
        bubble_contour (numpy.ndarray): Contour defining the speech bubble boundary
        
    Returns:
        numpy.ndarray: Image with translated text properly placed inside the bubble
    """
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Get bounding rectangle of the bubble contour
    x, y, w, h = cv2.boundingRect(bubble_contour)
    bubble_area = w * h
    
    print(f"📏 Bubble dimensions: {w}x{h} (area: {bubble_area}px²)")

    # Calculate optimal font size based on bubble dimensions
    # Enhanced algorithm with better classification for vertical and rectangular bubbles
    aspect_ratio = w / h if h > 0 else 1
    min_dimension = min(w, h)
    max_dimension = max(w, h)
    
    # More detailed classification for AGGRESSIVE font sizing - maximizing text size
    if aspect_ratio < 0.3:  # Extremely vertical bubble
        base_font_size = max(24, min(80, int(min_dimension * 0.32)))
        bubble_type = "🔺 Extremely Vertical"
        boost_factor = 1.53  # Reduced by 10% (1.7 * 0.9)
    elif aspect_ratio < 0.5:  # Very vertical bubble  
        base_font_size = max(22, min(75, int(min_dimension * 0.28)))
        bubble_type = "🔺 Very Vertical"
        boost_factor = 1.395  # Reduced by 10% (1.55 * 0.9)
    elif aspect_ratio < 0.7:  # Vertical bubble
        base_font_size = max(20, min(72, int(min_dimension * 0.25)))
        bubble_type = "🔺 Vertical"
        boost_factor = 1.26  # Reduced by 10% (1.4 * 0.9)
    elif aspect_ratio > 3.0:  # Extremely wide bubble
        base_font_size = max(14, min(50, int(h * 0.60)))
        bubble_type = "🔸 Extremely Wide"
        boost_factor = 0.81  # Reduced by 10% (0.9 * 0.9)
    elif aspect_ratio > 2.2:  # Very wide bubble
        base_font_size = max(16, min(55, int(h * 0.65)))
        bubble_type = "🔸 Very Wide" 
        boost_factor = 0.9  # Reduced by 10% (1.0 * 0.9)
    elif aspect_ratio > 1.6:  # Wide bubble
        base_font_size = max(18, min(60, int(h * 0.70)))
        bubble_type = "🔸 Wide"
        boost_factor = 0.99  # Reduced by 10% (1.1 * 0.9)
    else:  # Square-ish bubble (0.7 <= ratio <= 1.6)
        base_font_size = max(22, min(72, int(np.sqrt(bubble_area) * 0.18)))
        bubble_type = "🔷 Balanced"
        boost_factor = 1.08  # Reduced by 10% (1.2 * 0.9)
    
    # Apply boost factor
    base_font_size = int(base_font_size * boost_factor)
    
    print(f"{bubble_type} bubble detected (ratio: {aspect_ratio:.2f})")
    print(f"🎨 Initial font size: {base_font_size}px (boost: {boost_factor}x)")
    
    # OPTIMAL DUAL-DIMENSION FITTING ALGORITHM
    # Use 80-90% of both width and height as requested
    target_width = int(w * 0.88)   # 88% of width - more aggressive
    target_height = int(h * 0.88)  # 88% of height - more aggressive
    line_spacing_ratio = 1.05      # Even tighter line spacing
    min_font_size = 16
    
    print(f"🎯 Target area: {target_width}x{target_height} ({target_width * target_height}px²)")
    
    # ITERATIVE FONT SIZE OPTIMIZATION - test from large to small
    best_font_size = min_font_size
    best_fit = None
    
    # Start from a reasonable maximum and work down
    max_test_font = min(120, max(target_width // 4, target_height // 2))
    
    for test_font in range(max_test_font, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size=test_font)
        line_height = int(test_font * line_spacing_ratio)
        
        # Calculate optimal characters per line for this font size
        # Improved character width estimation - varies by font size
        if test_font >= 40:
            char_width = test_font * 0.55  # Larger fonts are more compact
        elif test_font >= 20:
            char_width = test_font * 0.58  # Medium fonts
        else:
            char_width = test_font * 0.62  # Small fonts are wider proportionally
        
        chars_per_line = max(5, int(target_width / char_width))
        
        # Wrap text with current parameters
        wrapped_text = textwrap.fill(text, width=chars_per_line, 
                                   break_long_words=True, break_on_hyphens=True)
        lines = wrapped_text.split('\n')
        
        # Calculate actual text dimensions
        max_line_width = 0
        for line in lines:
            if line.strip():  # Skip empty lines
                line_width = draw.textlength(line, font=font)
                max_line_width = max(max_line_width, line_width)
        
        total_text_height = len(lines) * line_height
        
        # Check if this font size fits BOTH dimensions
        fits_width = max_line_width <= target_width
        fits_height = total_text_height <= target_height
        
        if fits_width and fits_height:
            # Found a good fit! Only save it if it's LARGER than current best
            if best_fit is None or test_font > best_font_size:
                best_font_size = test_font
                best_fit = {
                    'font': font,
                    'font_size': test_font,
                    'line_height': line_height,
                    'wrapped_text': wrapped_text,
                    'lines': lines,
                    'max_line_width': max_line_width,
                    'total_height': total_text_height,
                    'chars_per_line': chars_per_line
                }
                print(f"   ✅ Font {test_font}px: {max_line_width:.0f}x{total_text_height} fits - NEW BEST!")
            else:
                print(f"   ✅ Font {test_font}px: {max_line_width:.0f}x{total_text_height} fits")
            # Continue to find if there's an even LARGER font that works
        else:
            print(f"   ❌ Font {test_font}px: {max_line_width:.0f}x{total_text_height} too big for {target_width}x{target_height}")
            # Continue testing smaller sizes
    
    # Use the best fit we found
    if best_fit:
        font = best_fit['font']
        line_height = best_fit['line_height']
        wrapped_text = best_fit['wrapped_text']
        lines = best_fit['lines']
        total_text_height = best_fit['total_height']
        print(f"🎯 OPTIMAL: Using {best_font_size}px font (LARGEST that fits) with {len(lines)} lines")
    else:
        # Fallback: use minimum font size
        font_size = min_font_size
        font = ImageFont.truetype(font_path, size=font_size)
        line_height = int(font_size * line_spacing_ratio)
        char_width = font_size * 0.6
        chars_per_line = max(5, int(target_width / char_width))
        wrapped_text = textwrap.fill(text, width=chars_per_line, 
                                   break_long_words=True, break_on_hyphens=True)
        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height
        print(f"⚠️  FALLBACK: Using minimum {min_font_size}px font")

    # Calculate vertical centering position
    # Fix: Account for text baseline and proper centering
    # The text_y should position the text so its center aligns with bubble center
    bubble_center_y = y + h // 2
    text_center_offset = total_text_height // 2
    text_y = bubble_center_y - text_center_offset
    
    # Ensure text doesn't go above bubble bounds
    text_y = max(text_y, y + 5)  # Add small padding from top
    
    print(f"🎯 Final text placement: {len(lines)} lines, font={font.size}px, "
          f"bubble_center=({x + w//2}, {y + h//2}), text_start=({x + w//2}, {text_y})")

    # Draw each line of text with enhanced rendering
    current_y = text_y
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            current_y += line_height
            continue
            
        # Calculate text width for horizontal centering
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        # Calculate horizontal centering position
        text_x = x + (w - text_width) // 2

        # Draw text with slight outline for better readability
        outline_width = max(1, font.size // 16)  # Dynamic outline based on font size
        
        if outline_width > 0:
            # Draw outline
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, current_y + dy), line, 
                                 font=font, fill=(255, 255, 255))  # White outline
        
        # Draw main text (black)
        draw.text((text_x, current_y), line, font=font, fill=(0, 0, 0))

        # Move to next line position
        current_y += line_height

    # Convert PIL Image back to OpenCV format (BGR)
    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
