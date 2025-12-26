"""
OLED Display Management Module
Handles I2C communication and display operations for SSD1306 OLED
"""

try:
    from machine import I2C, Pin
    from ssd1306 import SSD1306_I2C
    MICROPYTHON_AVAILABLE = True
except ImportError:
    # Mock for development/testing on non-MicroPython environments
    MICROPYTHON_AVAILABLE = False
    print("Warning: MicroPython modules not available. Running in simulation mode.")

from hardware_config import get_i2c_config, get_oled_config


class OLEDDisplay:
    """
    Manages OLED display operations via I2C
    """
    
    def __init__(self):
        """Initialize OLED display with I2C communication"""
        self.i2c_config = get_i2c_config()
        self.oled_config = get_oled_config()
        self.display = None
        self.width = self.oled_config['width']
        self.height = self.oled_config['height']
        
        if MICROPYTHON_AVAILABLE:
            self._init_hardware()
        else:
            print("OLED Display initialized in simulation mode")
    
    def _init_hardware(self):
        """Initialize I2C and OLED hardware"""
        try:
            # Initialize I2C
            self.i2c = I2C(
                self.i2c_config['id'],
                sda=Pin(self.i2c_config['sda']),
                scl=Pin(self.i2c_config['scl']),
                freq=self.i2c_config['freq']
            )
            
            # Initialize OLED display
            self.display = SSD1306_I2C(
                self.width,
                self.height,
                self.i2c,
                addr=self.oled_config['address']
            )
            
            print("OLED Display initialized successfully")
        except Exception as e:
            print(f"Error initializing OLED: {e}")
            self.display = None
    
    def clear(self):
        """Clear the display"""
        if self.display:
            self.display.fill(0)
            self.display.show()
        else:
            print("[SIM] Display cleared")
    
    def show_text(self, text, x=0, y=0, clear=True):
        """
        Display text on OLED
        
        Args:
            text (str): Text to display
            x (int): X coordinate
            y (int): Y coordinate
            clear (bool): Clear display before showing text
        """
        if self.display:
            if clear:
                self.display.fill(0)
            self.display.text(str(text), x, y)
            self.display.show()
        else:
            print(f"[SIM] Display: '{text}' at ({x}, {y})")
    
    def show_multiline(self, lines, line_height=10, clear=True):
        """
        Display multiple lines of text
        
        Args:
            lines (list): List of text lines to display
            line_height (int): Height between lines in pixels
            clear (bool): Clear display before showing text
        """
        if self.display:
            if clear:
                self.display.fill(0)
            
            for i, line in enumerate(lines):
                y_pos = i * line_height
                if y_pos < self.height:
                    self.display.text(str(line), 0, y_pos)
            
            self.display.show()
        else:
            print("[SIM] Multiline display:")
            for i, line in enumerate(lines):
                print(f"  Line {i}: {line}")
    
    def show_centered(self, text, y=None, clear=True):
        """
        Display centered text
        
        Args:
            text (str): Text to display
            y (int): Y coordinate (defaults to center)
            clear (bool): Clear display before showing text
        """
        if y is None:
            y = self.height // 2 - 4
        
        # Approximate character width (8 pixels per character for default font)
        text_width = len(str(text)) * 8
        x = (self.width - text_width) // 2
        
        self.show_text(text, x, y, clear)
    
    def draw_progress_bar(self, progress, x=0, y=28, width=None, height=8, clear=True):
        """
        Draw a progress bar
        
        Args:
            progress (float): Progress value (0.0 to 1.0)
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width of progress bar (defaults to display width)
            height (int): Height of progress bar
            clear (bool): Clear display before drawing
        """
        if width is None:
            width = self.width - 2 * x
        
        progress = max(0.0, min(1.0, progress))
        
        if self.display:
            if clear:
                self.display.fill(0)
            
            # Draw border
            self.display.rect(x, y, width, height, 1)
            
            # Draw filled progress
            fill_width = int((width - 2) * progress)
            if fill_width > 0:
                self.display.fill_rect(x + 1, y + 1, fill_width, height - 2, 1)
            
            self.display.show()
        else:
            bar_length = 20
            filled = int(bar_length * progress)
            bar = '[' + '=' * filled + ' ' * (bar_length - filled) + ']'
            print(f"[SIM] Progress: {bar} {progress*100:.0f}%")
    
    def show_test_status(self, test_name, status, result=""):
        """
        Display test status information
        
        Args:
            test_name (str): Name of the test
            status (str): Status (e.g., "PASS", "FAIL", "RUNNING")
            result (str): Additional result information
        """
        lines = [
            "TEST STATUS",
            "-" * 16,
            f"{test_name}",
            f"Status: {status}",
        ]
        
        if result:
            lines.append(f"{result}")
        
        self.show_multiline(lines[:6])  # Limit to 6 lines for 64px display
    
    def is_available(self):
        """
        Check if display is available and initialized
        
        Returns:
            bool: True if display is available
        """
        return self.display is not None
