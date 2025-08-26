#!/usr/bin/env python3
"""
Font Manager for MangaTranslator
================================

Tự động detect và load fonts từ thư mục fonts/ thay vì hardcode.

Features:
- Auto scan fonts từ thư mục fonts/
- Detect font type (TTF, OTF, etc.)
- Hỗ trợ font fallback khi font không tồn tại
- Optimize font loading cho performance
- Font preview và recommendation

Author: MangaTranslator Team
License: MIT
Version: 1.0
"""

import os
import glob
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import platform

class FontManager:
    """
    Quản lý và auto-load fonts cho MangaTranslator
    """
    
    def __init__(self, fonts_dir: str = "fonts"):
        """
        Initialize Font Manager
        
        Args:
            fonts_dir (str): Thư mục chứa fonts
        """
        self.fonts_dir = fonts_dir
        self.available_fonts = {}
        self.default_fonts = {}
        self.font_cache = {}  # Cache loaded fonts để tăng performance
        
        # Supported font extensions
        self.font_extensions = ['.ttf', '.otf', '.woff', '.woff2']
        
        self.scan_fonts()
        self.setup_default_fonts()
    
    def scan_fonts(self):
        """
        Scan thư mục fonts và catalog tất cả fonts có sẵn
        """
        if not os.path.exists(self.fonts_dir):
            print(f"⚠️ Fonts directory not found: {self.fonts_dir}")
            os.makedirs(self.fonts_dir, exist_ok=True)
            return
        
        print(f"🔍 Scanning fonts in {self.fonts_dir}...")
        
        self.available_fonts = {}
        
        # Scan cho tất cả font files
        for ext in self.font_extensions:
            pattern = os.path.join(self.fonts_dir, f"*{ext}")
            font_files = glob.glob(pattern)
            
            for font_path in font_files:
                font_name = os.path.splitext(os.path.basename(font_path))[0]
                
                # Detect font info
                font_info = self._analyze_font(font_path)
                
                self.available_fonts[font_name] = {
                    'path': font_path,
                    'type': font_info['type'],
                    'recommended_for': font_info['recommended_for'],
                    'size': font_info['size'],
                    'supports_vietnamese': font_info['supports_vietnamese']
                }
        
        print(f"✅ Found {len(self.available_fonts)} fonts:")
        for font_name, info in self.available_fonts.items():
            print(f"  📝 {font_name} ({info['type']}) - {info['recommended_for']}")
    
    def _analyze_font(self, font_path: str) -> Dict:
        """
        Phân tích font để đưa ra recommendations
        """
        font_name = os.path.basename(font_path).lower()
        file_size = os.path.getsize(font_path)
        
        # Detect font type từ filename patterns
        if 'anime' in font_name or 'manga' in font_name:
            recommended_for = "Manga/Comic text"
        elif 'arial' in font_name or 'times' in font_name or 'roboto' in font_name:
            recommended_for = "General text"
        elif 'bold' in font_name or 'black' in font_name:
            recommended_for = "Headlines/Strong text"
        elif 'italic' in font_name or 'oblique' in font_name:
            recommended_for = "Emphasis/Thought bubbles"
        elif 'condensed' in font_name or 'narrow' in font_name:
            recommended_for = "Small spaces/Dense text"
        else:
            recommended_for = "General purpose"
        
        # Detect font format
        ext = os.path.splitext(font_path)[1].lower()
        font_type = ext[1:].upper() if ext else "Unknown"
        
        # Test Vietnamese support (basic check)
        supports_vietnamese = self._test_vietnamese_support(font_path)
        
        return {
            'type': font_type,
            'recommended_for': recommended_for,
            'size': file_size,
            'supports_vietnamese': supports_vietnamese
        }
    
    def _test_vietnamese_support(self, font_path: str) -> bool:
        """
        Test xem font có support Vietnamese characters không
        """
        try:
            # Load font với size nhỏ để test
            font = ImageFont.truetype(font_path, 12)
            
            # Test với một số Vietnamese characters
            vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ"
            
            # Tạo image nhỏ để test render
            test_img = Image.new('RGB', (100, 30), 'white')
            draw = ImageDraw.Draw(test_img)
            
            # Try render một số ký tự Vietnamese
            try:
                draw.text((5, 5), vietnamese_chars[:10], font=font, fill='black')
                return True
            except:
                return False
                
        except Exception:
            return False
    
    def setup_default_fonts(self):
        """
        Setup default fonts cho các use cases khác nhau
        """
        self.default_fonts = {
            'manga': self._find_best_font_for(['anime', 'manga', 'comic']),
            'general': self._find_best_font_for(['arial', 'roboto', 'noto']),
            'bold': self._find_best_font_for(['bold', 'black', 'heavy']),
            'italic': self._find_best_font_for(['italic', 'oblique']),
            'condensed': self._find_best_font_for(['condensed', 'narrow', 'compact'])
        }
        
        # Fallback system font nếu không có font nào
        if not any(self.default_fonts.values()):
            self.default_fonts['system'] = self._get_system_fallback_font()
        
        print("🎯 Default fonts setup:")
        for use_case, font_name in self.default_fonts.items():
            if font_name:
                print(f"  {use_case}: {font_name}")
    
    def _find_best_font_for(self, keywords: List[str]) -> Optional[str]:
        """
        Tìm font tốt nhất cho keywords cụ thể
        """
        scored_fonts = []
        
        for font_name, info in self.available_fonts.items():
            score = 0
            font_name_lower = font_name.lower()
            
            # Score based on keywords trong tên
            for keyword in keywords:
                if keyword in font_name_lower:
                    score += 3
            
            # Bonus cho Vietnamese support
            if info['supports_vietnamese']:
                score += 2
            
            # Bonus cho font size hợp lý (không quá nhỏ hoặc quá lớn)
            size_mb = info['size'] / (1024 * 1024)
            if 0.1 <= size_mb <= 5:  # 100KB - 5MB
                score += 1
            
            if score > 0:
                scored_fonts.append((font_name, score))
        
        if scored_fonts:
            # Sort by score descending và return font tốt nhất
            scored_fonts.sort(key=lambda x: x[1], reverse=True)
            return scored_fonts[0][0]
        
        return None
    
    def _get_system_fallback_font(self) -> str:
        """
        Lấy system fallback font nếu không có font custom nào
        """
        system = platform.system()
        
        if system == "Windows":
            # Windows fonts
            windows_fonts = ["arial.ttf", "times.ttf", "calibri.ttf"]
            for font in windows_fonts:
                font_path = os.path.join("C:\\Windows\\Fonts", font)
                if os.path.exists(font_path):
                    return font_path
        
        elif system == "Darwin":  # macOS
            macos_fonts = ["Arial.ttf", "Times New Roman.ttf", "Helvetica.ttc"]
            for font in macos_fonts:
                font_path = os.path.join("/System/Library/Fonts", font)
                if os.path.exists(font_path):
                    return font_path
        
        elif system == "Linux":
            linux_fonts = ["DejaVuSans.ttf", "liberation-sans.ttf", "ubuntu.ttf"]
            possible_dirs = ["/usr/share/fonts/truetype/dejavu/", 
                           "/usr/share/fonts/truetype/liberation/",
                           "/usr/share/fonts/truetype/ubuntu/"]
            
            for font_dir, font in zip(possible_dirs, linux_fonts):
                font_path = os.path.join(font_dir, font)
                if os.path.exists(font_path):
                    return font_path
        
        return None
    
    def get_font_path(self, font_preference: str = "manga") -> str:
        """
        Lấy đường dẫn font theo preference
        
        Args:
            font_preference (str): Loại font cần ('manga', 'general', 'bold', etc.) 
                                  hoặc tên font cụ thể
            
        Returns:
            str: Đường dẫn đến font file
        """
        # Nếu font_preference là tên font cụ thể
        if font_preference in self.available_fonts:
            return self.available_fonts[font_preference]['path']
        
        # Nếu là preference type
        if font_preference in self.default_fonts and self.default_fonts[font_preference]:
            font_name = self.default_fonts[font_preference]
            if font_name in self.available_fonts:
                return self.available_fonts[font_name]['path']
        
        # Fallback sequence
        fallback_order = ['manga', 'general', 'system']
        
        for fallback in fallback_order:
            if fallback in self.default_fonts and self.default_fonts[fallback]:
                font_name = self.default_fonts[fallback]
                if font_name in self.available_fonts:
                    return self.available_fonts[font_name]['path']
                elif fallback == 'system':
                    return font_name  # System font path
        
        # Cuối cùng return font đầu tiên có sẵn
        if self.available_fonts:
            first_font = list(self.available_fonts.values())[0]
            return first_font['path']
        
        # Hoàn toàn không có font
        print("⚠️ No fonts available! Please add fonts to fonts/ directory")
        return None
    
    def load_font(self, font_preference: str = "manga", size: int = 16) -> ImageFont.FreeTypeFont:
        """
        Load font với size cụ thể, có cache để tăng performance
        
        Args:
            font_preference (str): Font preference
            size (int): Font size
            
        Returns:
            ImageFont.FreeTypeFont: Loaded font object
        """
        cache_key = f"{font_preference}_{size}"
        
        # Check cache trước
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]
        
        font_path = self.get_font_path(font_preference)
        
        if not font_path or not os.path.exists(font_path):
            print(f"⚠️ Font not found: {font_preference}, using default")
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        else:
            try:
                font = ImageFont.truetype(font_path, size)
                print(f"✅ Loaded font: {os.path.basename(font_path)} (size {size})")
            except Exception as e:
                print(f"❌ Error loading font {font_path}: {e}")
                font = ImageFont.load_default()
        
        # Cache font
        self.font_cache[cache_key] = font
        return font
    
    def get_font_recommendations(self, text_type: str = "general") -> List[str]:
        """
        Lấy danh sách fonts được recommend cho loại text cụ thể
        
        Args:
            text_type (str): Loại text ('manga', 'thought', 'sfx', 'general')
            
        Returns:
            List[str]: Danh sách tên fonts được recommend
        """
        recommendations = []
        
        if text_type == "manga":
            keywords = ['anime', 'manga', 'comic']
        elif text_type == "thought":
            keywords = ['italic', 'light', 'thin']
        elif text_type == "sfx":
            keywords = ['bold', 'black', 'heavy', 'impact']
        else:
            keywords = ['arial', 'roboto', 'sans']
        
        # Score và sort fonts
        scored_fonts = []
        for font_name, info in self.available_fonts.items():
            score = 0
            font_name_lower = font_name.lower()
            
            for keyword in keywords:
                if keyword in font_name_lower:
                    score += 1
            
            if info['supports_vietnamese']:
                score += 2
            
            scored_fonts.append((font_name, score))
        
        # Sort và return top recommendations
        scored_fonts.sort(key=lambda x: x[1], reverse=True)
        return [font[0] for font in scored_fonts[:5]]
    
    def get_all_fonts(self) -> Dict[str, str]:
        """
        Lấy mapping tất cả fonts available cho UI dropdown
        
        Returns:
            Dict[str, str]: {display_name: font_path}
        """
        result = {}
        
        for font_name, info in self.available_fonts.items():
            display_name = f"{font_name} ({info['recommended_for']})"
            result[display_name] = info['path']
        
        return result
    
    def refresh_fonts(self):
        """
        Refresh font list (rescan thư mục fonts)
        """
        print("🔄 Refreshing font list...")
        self.font_cache.clear()  # Clear cache
        self.scan_fonts()
        self.setup_default_fonts()
        print("✅ Font list refreshed")

# Test và utility functions
def test_font_manager():
    """Test Font Manager functionality"""
    print("🧪 Testing Font Manager...")
    
    fm = FontManager()
    
    print(f"\nAvailable fonts: {len(fm.available_fonts)}")
    
    # Test get font path
    manga_font = fm.get_font_path("manga")
    print(f"Manga font: {manga_font}")
    
    # Test load font
    font = fm.load_font("manga", 20)
    print(f"Loaded font: {font}")
    
    # Test recommendations
    recommendations = fm.get_font_recommendations("manga")
    print(f"Manga recommendations: {recommendations}")
    
    return fm

if __name__ == "__main__":
    test_font_manager()