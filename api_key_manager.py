#!/usr/bin/env python3
"""
API Key Manager for MangaTranslator
===================================

Quản lý nhiều API key và tự động rotate để tối ưu usage.

Features:
- Hỗ trợ nhiều provider (Gemini, Google Translate, etc.)
- Round-robin rotation hoặc random selection
- Tracking usage và daily limits
- Auto fallback khi key hết quota
- Reset daily usage tự động

Author: MangaTranslator Team
License: MIT
Version: 1.0
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

class ApiKeyManager:
    """
    Quản lý và rotate API keys cho các dịch vụ translation
    """
    
    def __init__(self, config_path: str = "config/api_keys.json"):
        """
        Initialize API Key Manager
        
        Args:
            config_path (str): Đường dẫn đến file config JSON
        """
        self.config_path = config_path
        self.config = {}
        self.current_indices = {}  # Track current index for round-robin
        self.last_used_times = {}  # Track last used time for rate limiting
        
        self.load_config()
        
    def load_config(self):
        """Load configuration từ JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    
                # Reset daily usage nếu cần
                if self.config.get('auto_reset_usage', True):
                    self._reset_daily_usage_if_needed()
                    
                print(f"✅ Loaded API keys config from {self.config_path}")
                self._print_key_status()
            else:
                print(f"⚠️ Config file not found: {self.config_path}")
                self._create_default_config()
                
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            self.config = {}
    
    def save_config(self):
        """Lưu configuration vào JSON file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _create_default_config(self):
        """Tạo config mặc định nếu file không tồn tại"""
        default_config = {
            "gemini_keys": [],
            "google_translate_keys": [],
            "rotation_strategy": "round_robin",
            "fallback_to_free": True,
            "auto_reset_usage": True
        }
        
        self.config = default_config
        self.save_config()
        print(f"📝 Created default config at {self.config_path}")
    
    def _reset_daily_usage_if_needed(self):
        """Reset daily usage nếu đã sang ngày mới"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for provider in ['gemini_keys', 'google_translate_keys']:
            if provider in self.config:
                for key_info in self.config[provider]:
                    if key_info.get('last_reset') != today:
                        key_info['usage_count'] = 0
                        key_info['last_reset'] = today
        
        # Lưu config sau khi reset
        self.save_config()
    
    def _print_key_status(self):
        """In ra status của các API keys"""
        print("\n📊 API Key Status:")
        
        for provider in ['gemini_keys', 'google_translate_keys']:
            if provider in self.config and self.config[provider]:
                provider_name = provider.replace('_keys', '').title()
                print(f"\n{provider_name}:")
                
                for i, key_info in enumerate(self.config[provider]):
                    status = "🟢" if key_info.get('active', False) else "🔴"
                    usage = key_info.get('usage_count', 0)
                    limit = key_info.get('daily_limit', 0)
                    name = key_info.get('name', f'Key {i+1}')
                    
                    print(f"  {status} {name}: {usage}/{limit} requests")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Lấy API key cho provider theo rotation strategy
        
        Args:
            provider (str): Tên provider ('gemini', 'google_translate')
            
        Returns:
            str: API key hoặc None nếu không có key khả dụng
        """
        provider_key = f"{provider}_keys"
        
        if provider_key not in self.config or not self.config[provider_key]:
            print(f"⚠️ No {provider} keys configured")
            return None
        
        keys = self.config[provider_key]
        available_keys = [k for k in keys if k.get('active', False) and not self._is_key_exhausted(k)]
        
        if not available_keys:
            print(f"⚠️ No available {provider} keys (all exhausted or inactive)")
            return None
        
        # Chọn key theo strategy
        strategy = self.config.get('rotation_strategy', 'round_robin')
        
        if strategy == 'round_robin':
            selected_key = self._get_key_round_robin(provider, available_keys)
        elif strategy == 'random':
            selected_key = random.choice(available_keys)
        elif strategy == 'least_used':
            selected_key = min(available_keys, key=lambda k: k.get('usage_count', 0))
        else:
            selected_key = available_keys[0]  # Default to first
        
        if selected_key:
            # Increment usage count
            selected_key['usage_count'] = selected_key.get('usage_count', 0) + 1
            self.save_config()
            
            key_name = selected_key.get('name', 'Unknown')
            usage = selected_key.get('usage_count', 0)
            limit = selected_key.get('daily_limit', 0)
            
            print(f"🔑 Using {provider} key: {key_name} ({usage}/{limit})")
            
            return selected_key['key']
        
        return None
    
    def _get_key_round_robin(self, provider: str, available_keys: List[Dict]) -> Optional[Dict]:
        """Get key using round-robin strategy"""
        if provider not in self.current_indices:
            self.current_indices[provider] = 0
        
        # Find the original index in the config
        all_keys = self.config[f"{provider}_keys"]
        current_key = available_keys[self.current_indices[provider] % len(available_keys)]
        
        # Update index for next call
        self.current_indices[provider] = (self.current_indices[provider] + 1) % len(available_keys)
        
        return current_key
    
    def _is_key_exhausted(self, key_info: Dict) -> bool:
        """Kiểm tra xem key đã hết quota chưa"""
        usage = key_info.get('usage_count', 0)
        limit = key_info.get('daily_limit', float('inf'))
        
        return usage >= limit
    
    def add_api_key(self, provider: str, key: str, name: str = None, daily_limit: int = 1000):
        """
        Thêm API key mới
        
        Args:
            provider (str): Provider name ('gemini', 'google_translate')
            key (str): API key
            name (str): Tên mô tả cho key
            daily_limit (int): Giới hạn sử dụng hàng ngày
        """
        provider_key = f"{provider}_keys"
        
        if provider_key not in self.config:
            self.config[provider_key] = []
        
        new_key = {
            "key": key,
            "name": name or f"{provider.title()} Key {len(self.config[provider_key]) + 1}",
            "active": True,
            "daily_limit": daily_limit,
            "usage_count": 0,
            "last_reset": datetime.now().strftime("%Y-%m-%d")
        }
        
        self.config[provider_key].append(new_key)
        self.save_config()
        
        print(f"✅ Added new {provider} API key: {new_key['name']}")
    
    def deactivate_key(self, provider: str, key_index: int):
        """Vô hiệu hóa một API key"""
        provider_key = f"{provider}_keys"
        
        if provider_key in self.config and 0 <= key_index < len(self.config[provider_key]):
            self.config[provider_key][key_index]['active'] = False
            self.save_config()
            
            key_name = self.config[provider_key][key_index].get('name', f'Key {key_index}')
            print(f"🔴 Deactivated {provider} key: {key_name}")
    
    def activate_key(self, provider: str, key_index: int):
        """Kích hoạt lại một API key"""
        provider_key = f"{provider}_keys"
        
        if provider_key in self.config and 0 <= key_index < len(self.config[provider_key]):
            self.config[provider_key][key_index]['active'] = True
            self.save_config()
            
            key_name = self.config[provider_key][key_index].get('name', f'Key {key_index}')
            print(f"🟢 Activated {provider} key: {key_name}")
    
    def get_key_status(self, provider: str) -> Dict:
        """Lấy thống kê trạng thái keys của provider"""
        provider_key = f"{provider}_keys"
        
        if provider_key not in self.config:
            return {"total": 0, "active": 0, "exhausted": 0, "available": 0}
        
        keys = self.config[provider_key]
        total = len(keys)
        active = len([k for k in keys if k.get('active', False)])
        exhausted = len([k for k in keys if self._is_key_exhausted(k)])
        available = len([k for k in keys if k.get('active', False) and not self._is_key_exhausted(k)])
        
        return {
            "total": total,
            "active": active,
            "exhausted": exhausted,
            "available": available
        }
    
    def batch_translate_with_rotation(self, texts: List[str], translator_func, provider: str = 'gemini', 
                                     max_retries: int = 3) -> List[str]:
        """
        Dịch batch texts với rotation API keys khi cần
        
        Args:
            texts (List[str]): Danh sách texts cần dịch
            translator_func: Function dịch nhận (text, api_key) -> translated_text
            provider (str): Provider name
            max_retries (int): Số lần retry khi fail
            
        Returns:
            List[str]: Danh sách texts đã dịch
        """
        results = []
        current_api_key = self.get_api_key(provider)
        
        if not current_api_key:
            print(f"❌ No available {provider} API keys for batch translation")
            return texts  # Return original texts if no keys
        
        for i, text in enumerate(texts):
            retry_count = 0
            translated = None
            
            while retry_count < max_retries and not translated:
                try:
                    # Rate limiting - thêm delay nhỏ giữa các requests
                    if i > 0:
                        time.sleep(0.5)
                    
                    translated = translator_func(text, current_api_key)
                    
                    if not translated or translated == text:
                        # API có thể bị limit, thử key khác
                        print(f"⚠️ Translation failed for text {i+1}, trying next key...")
                        current_api_key = self.get_api_key(provider)
                        
                        if not current_api_key:
                            print(f"❌ No more available {provider} keys")
                            translated = text  # Fallback to original
                            break
                        
                        retry_count += 1
                        continue
                    
                    results.append(translated)
                    break
                    
                except Exception as e:
                    print(f"❌ Error translating text {i+1}: {e}")
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        # Try next API key
                        current_api_key = self.get_api_key(provider)
                        if not current_api_key:
                            break
                    
            # Nếu không dịch được sau max_retries, giữ nguyên text gốc
            if not translated:
                results.append(text)
                print(f"⚠️ Could not translate text {i+1}, keeping original")
        
        return results

# Convenience functions
def setup_api_keys():
    """Helper function để setup API keys lần đầu"""
    manager = ApiKeyManager()
    
    print("🔧 API Key Setup Helper")
    print("=" * 50)
    
    # Setup Gemini keys
    print("\n1. Gemini API Keys:")
    gemini_keys = input("Enter Gemini API keys (comma separated): ").strip()
    
    if gemini_keys:
        keys = [k.strip() for k in gemini_keys.split(',') if k.strip()]
        for i, key in enumerate(keys):
            manager.add_api_key('gemini', key, f"Gemini Account {i+1}", 1000)
    
    # Setup Google Translate keys (if any)
    print("\n2. Google Translate API Keys (optional):")
    google_keys = input("Enter Google Translate keys (comma separated, or press Enter to skip): ").strip()
    
    if google_keys:
        keys = [k.strip() for k in google_keys.split(',') if k.strip()]
        for i, key in enumerate(keys):
            manager.add_api_key('google_translate', key, f"Google Account {i+1}", 5000)
    
    print("\n✅ API Keys setup completed!")
    manager._print_key_status()
    
    return manager

if __name__ == "__main__":
    # Test the API Key Manager
    setup_api_keys()