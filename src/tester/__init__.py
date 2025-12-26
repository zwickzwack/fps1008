"""
Pico Pi 2 W Tester Module
A modular testing framework for Raspberry Pi Pico Pi 2 W with OLED display
"""

__version__ = "1.0.0"
__author__ = "fps1008"

# Import main components for easier access
from .hardware_config import (
    I2C_SDA_PIN,
    I2C_SCL_PIN,
    I2C_FREQ,
    OLED_WIDTH,
    OLED_HEIGHT,
    OLED_ADDRESS,
    get_i2c_config,
    get_oled_config
)

from .oled_display import OLEDDisplay
from .test_functions import TestRunner, test_display, test_memory, test_led_blink, test_i2c_scan
from .utils import (
    format_time,
    format_bytes,
    truncate_text,
    Timer,
    RollingAverage
)

__all__ = [
    # Constants
    'I2C_SDA_PIN',
    'I2C_SCL_PIN',
    'I2C_FREQ',
    'OLED_WIDTH',
    'OLED_HEIGHT',
    'OLED_ADDRESS',
    
    # Functions
    'get_i2c_config',
    'get_oled_config',
    'format_time',
    'format_bytes',
    'truncate_text',
    
    # Classes
    'OLEDDisplay',
    'TestRunner',
    'Timer',
    'RollingAverage',
    
    # Test functions
    'test_display',
    'test_memory',
    'test_led_blink',
    'test_i2c_scan',
]
