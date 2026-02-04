"""Command whitelist manager for persistent storage."""

import json
from pathlib import Path
from typing import Set, Dict, Any
from loguru import logger


class WhitelistManager:
    """Manages persistent command whitelist."""
    
    def __init__(self, whitelist_file: str = "command_whitelist.json"):
        self.whitelist_file = Path(whitelist_file)
        self.whitelist: Dict[str, Set[str]] = self._load_whitelist()
    
    def _load_whitelist(self) -> Dict[str, Set[str]]:
        """Load whitelist from file."""
        if not self.whitelist_file.exists():
            return {
                "file_operations": set(),
                "applications": set(),
                "web_urls": set()
            }
        
        try:
            with open(self.whitelist_file, 'r') as f:
                data = json.load(f)
                # Convert lists back to sets
                return {
                    key: set(value) for key, value in data.items()
                }
        except Exception as e:
            logger.error(f"Error loading whitelist: {e}")
            return {
                "file_operations": set(),
                "applications": set(),
                "web_urls": set()
            }
    
    def _save_whitelist(self):
        """Save whitelist to file."""
        try:
            # Convert sets to lists for JSON serialization
            data = {
                key: list(value) for key, value in self.whitelist.items()
            }
            with open(self.whitelist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving whitelist: {e}")
    
    def is_whitelisted(self, category: str, item: str) -> bool:
        """Check if an item is whitelisted."""
        return item in self.whitelist.get(category, set())
    
    def add_to_whitelist(self, category: str, item: str):
        """Add an item to the whitelist."""
        if category not in self.whitelist:
            self.whitelist[category] = set()
        
        self.whitelist[category].add(item)
        self._save_whitelist()
        logger.info(f"Added to whitelist [{category}]: {item}")
    
    def remove_from_whitelist(self, category: str, item: str):
        """Remove an item from the whitelist."""
        if category in self.whitelist:
            self.whitelist[category].discard(item)
            self._save_whitelist()
            logger.info(f"Removed from whitelist [{category}]: {item}")
    
    def get_whitelist(self, category: str) -> Set[str]:
        """Get all items in a category."""
        return self.whitelist.get(category, set()).copy()
