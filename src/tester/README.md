# Pico Pi 2 W Tester with OLED Display

A modular testing framework for Raspberry Pi Pico Pi 2 W with an SSD1306 OLED display connected via I2C.

## 📋 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **OLED Display Support**: Full support for SSD1306 128x64 OLED displays via I2C
- **Test Framework**: Built-in test runner with result tracking
- **Multiple Operation Modes**:
  - Quick Test: Run a comprehensive test suite
  - Continuous Monitoring: Real-time system monitoring
  - Display Demo: Showcase display capabilities
- **Utility Functions**: Helper functions for formatting, timing, and data processing

## 📁 Project Structure

```
src/tester/
├── main.py                 # Main script (lean orchestrator)
├── hardware_config.py      # Hardware configuration and pin definitions
├── oled_display.py         # OLED display management
├── test_functions.py       # Test routines and test runner
├── utils.py                # Utility functions and helpers
└── README.md               # This file
```

## 🔧 Hardware Requirements

- **Raspberry Pi Pico Pi 2 W**
- **SSD1306 OLED Display** (128x64 pixels, I2C interface)
- **Connections**:
  - OLED SDA → GPIO 0 (Pin 1)
  - OLED SCL → GPIO 1 (Pin 2)
  - OLED VCC → 3.3V
  - OLED GND → GND

## 📦 Dependencies

### MicroPython Libraries Required

1. **ssd1306.py** - OLED display driver
   ```bash
   # Download from: https://github.com/micropython/micropython/blob/master/drivers/display/ssd1306.py
   # Or use Thonny IDE's package manager
   ```

2. **machine** - Built-in MicroPython hardware interface (pre-installed)

## 🚀 Installation

### Step 1: Install MicroPython on Pico Pi 2 W

1. Download the latest MicroPython firmware for Pico W from [micropython.org](https://micropython.org/download/rp2-pico-w/)
2. Hold the BOOTSEL button and connect Pico to your computer
3. Copy the `.uf2` file to the RPI-RP2 drive

### Step 2: Install ssd1306 Library

Using Thonny IDE:
1. Open Thonny IDE
2. Go to Tools → Manage packages
3. Search for "ssd1306" and install

Or manually:
1. Download `ssd1306.py` from the MicroPython repository
2. Upload it to your Pico's root directory

### Step 3: Upload Tester Files

Upload all files from `src/tester/` to your Pico Pi 2 W:
- `main.py`
- `hardware_config.py`
- `oled_display.py`
- `test_functions.py`
- `utils.py`

## 💻 Usage

### Running the Tester

1. Connect to your Pico via Thonny IDE or any MicroPython REPL
2. Run the main script:
   ```python
   import main
   ```
   Or if main.py is set as the boot script, it will run automatically on power-up.

### Operation Modes

The tester supports multiple operation modes:

#### 1. Quick Test Mode (Default)
Runs a comprehensive test suite including:
- Display functionality test
- Memory availability test
- LED blink test
- I2C device scan

#### 2. Continuous Monitoring Mode
Displays real-time system information:
- Uptime counter
- Iteration count
- Free memory
- Updates every 5 seconds

#### 3. Display Demo Mode
Demonstrates display capabilities:
- Text positioning
- Centered text
- Multi-line text
- Progress bar animation

## 🔧 Configuration

### Modifying Hardware Configuration

Edit `hardware_config.py` to customize:

```python
# I2C Configuration
I2C_SDA_PIN = 0      # Change if using different pins
I2C_SCL_PIN = 1
I2C_FREQ = 400000    # I2C frequency

# OLED Display
OLED_WIDTH = 128     # Display width
OLED_HEIGHT = 64     # Display height
OLED_ADDRESS = 0x3C  # I2C address (0x3C or 0x3D)

# LED Configuration
LED_PIN = 25         # Built-in LED pin
```

## 📚 Module Documentation

### hardware_config.py
Centralizes all hardware-related configuration:
- Pin definitions
- I2C settings
- Display parameters
- Configuration getter functions

### oled_display.py
Manages OLED display operations:
- `OLEDDisplay` class: Main display controller
- `show_text()`: Display text at coordinates
- `show_multiline()`: Display multiple lines
- `show_centered()`: Display centered text
- `draw_progress_bar()`: Draw progress indicators
- `show_test_status()`: Display test results

### test_functions.py
Contains test routines and test runner:
- `TestRunner` class: Manages test execution
- `test_display()`: Test display functionality
- `test_memory()`: Check memory availability
- `test_led_blink()`: Test LED blinking
- `test_i2c_scan()`: Scan for I2C devices
- `test_comprehensive()`: Run all tests

### utils.py
Utility functions and helper classes:
- `format_time()`: Format time in seconds
- `format_bytes()`: Format byte sizes
- `truncate_text()`: Truncate strings
- `Timer` class: Measure elapsed time
- `RollingAverage` class: Calculate rolling averages

### main.py
Lean orchestrator script:
- Imports and initializes modules
- Manages operation modes
- Handles user interaction
- Error handling and cleanup

## 🧪 Development and Testing

### Testing on Non-MicroPython Environments

The code includes simulation mode for development on regular Python:
```bash
cd src/tester
python main.py
```

The modules will detect the absence of MicroPython and run in simulation mode, printing output to console instead of the OLED.

### Adding Custom Tests

Create new test functions in `test_functions.py`:

```python
def test_custom(display):
    """
    Your custom test
    
    Args:
        display: OLEDDisplay instance
        
    Returns:
        dict: Test result with 'success' and 'message' keys
    """
    try:
        # Your test logic here
        result = do_something()
        
        return {
            'success': True,
            'data': result,
            'message': 'Test passed'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Test failed: {e}'
        }
```

Then add it to the test sequence in `main.py`.

## 🐛 Troubleshooting

### Display Not Working

1. Check I2C connections (SDA, SCL, VCC, GND)
2. Verify I2C address (try 0x3C or 0x3D)
   ```python
   from machine import I2C, Pin
   i2c = I2C(0, sda=Pin(0), scl=Pin(1))
   print(i2c.scan())  # Should show [60] (0x3C) or [61] (0x3D)
   ```
3. Ensure ssd1306.py is installed
4. Check power supply (3.3V, sufficient current)

### Import Errors

- Ensure all module files are in the same directory
- Check file names match exactly (case-sensitive)
- Verify ssd1306.py is installed

### Memory Issues

- Run `gc.collect()` regularly
- Reduce buffer sizes if needed
- Simplify display operations

## 🔄 Future Enhancements

Potential improvements:
- [ ] WiFi connectivity tests
- [ ] Temperature sensor integration
- [ ] Data logging to file
- [ ] Web interface for remote monitoring
- [ ] Custom font support
- [ ] Graphics and image display
- [ ] Configuration via JSON file
- [ ] Test scheduling

## 📄 License

This project is part of the fps1008 repository. See repository LICENSE for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Keep modules focused and lean
2. Follow existing code style
3. Add documentation for new functions
4. Test on actual hardware when possible

## 📞 Support

For issues or questions, please create an issue in the repository.

---

**Happy Testing! 🚀**
