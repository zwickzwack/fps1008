"""
Hardware Configuration for Pico Pi 2 W Tester
Defines pin mappings, I2C configuration, and hardware constants
"""

# I2C Configuration for OLED Display
I2C_SDA_PIN = 0  # GPIO pin for I2C SDA
I2C_SCL_PIN = 1  # GPIO pin for I2C SCL
I2C_FREQ = 400000  # I2C frequency (400kHz)
I2C_ID = 0  # I2C bus ID

# OLED Display Configuration
OLED_WIDTH = 128  # Display width in pixels
OLED_HEIGHT = 64  # Display height in pixels
OLED_ADDRESS = 0x3C  # I2C address of OLED (common default)

# Test Configuration
TEST_INTERVAL = 1000  # Default test interval in milliseconds
LED_PIN = 25  # Built-in LED pin on Pico

# Display Settings
DISPLAY_REFRESH_RATE = 100  # Display refresh interval in ms
TEXT_SIZE = 1  # Default text size

def get_i2c_config():
    """
    Returns I2C configuration as a dictionary
    
    Returns:
        dict: I2C configuration parameters
    """
    return {
        'sda': I2C_SDA_PIN,
        'scl': I2C_SCL_PIN,
        'freq': I2C_FREQ,
        'id': I2C_ID
    }

def get_oled_config():
    """
    Returns OLED display configuration as a dictionary
    
    Returns:
        dict: OLED configuration parameters
    """
    return {
        'width': OLED_WIDTH,
        'height': OLED_HEIGHT,
        'address': OLED_ADDRESS
    }
