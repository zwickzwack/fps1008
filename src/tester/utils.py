"""
Utility Functions for Pico Pi 2 W Tester
Helper functions for formatting, conversions, and common operations
"""

import time


def format_time(seconds):
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds (int): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def format_bytes(bytes_val):
    """
    Format bytes to human-readable format
    
    Args:
        bytes_val (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    units = ['B', 'KB', 'MB', 'GB']
    unit_idx = 0
    size = float(bytes_val)
    
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    
    return f"{size:.2f} {units[unit_idx]}"


def truncate_text(text, max_length):
    """
    Truncate text to fit within maximum length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_timestamp():
    """
    Get current timestamp as formatted string
    
    Returns:
        str: Formatted timestamp
    """
    try:
        # MicroPython time.localtime()
        t = time.localtime()
        return f"{t[0]}-{t[1]:02d}-{t[2]:02d} {t[3]:02d}:{t[4]:02d}:{t[5]:02d}"
    except:
        # Fallback for standard Python
        return time.strftime("%Y-%m-%d %H:%M:%S")


def calculate_percentage(current, total):
    """
    Calculate percentage
    
    Args:
        current (float): Current value
        total (float): Total value
        
    Returns:
        float: Percentage (0.0 to 100.0)
    """
    if total == 0:
        return 0.0
    return (current / total) * 100.0


def map_value(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another (Arduino-style map function)
    
    Args:
        value (float): Input value
        in_min (float): Input range minimum
        in_max (float): Input range maximum
        out_min (float): Output range minimum
        out_max (float): Output range maximum
        
    Returns:
        float: Mapped value
    """
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def clamp(value, min_val, max_val):
    """
    Clamp value between minimum and maximum
    
    Args:
        value (float): Value to clamp
        min_val (float): Minimum value
        max_val (float): Maximum value
        
    Returns:
        float: Clamped value
    """
    return max(min_val, min(value, max_val))


class Timer:
    """Simple timer class for measuring elapsed time"""
    
    def __init__(self):
        """Initialize timer"""
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.elapsed = 0
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def get_elapsed(self):
        """Get elapsed time without stopping"""
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed
    
    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.elapsed = 0


class RollingAverage:
    """Calculate rolling average of values"""
    
    def __init__(self, size=10):
        """
        Initialize rolling average calculator
        
        Args:
            size (int): Number of values to keep in the rolling window
        """
        self.size = size
        self.values = []
    
    def add(self, value):
        """
        Add a value to the rolling average
        
        Args:
            value (float): Value to add
        """
        self.values.append(value)
        if len(self.values) > self.size:
            self.values.pop(0)
    
    def get_average(self):
        """
        Get the current rolling average
        
        Returns:
            float: Average value, or 0 if no values
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the rolling average"""
        self.values = []
