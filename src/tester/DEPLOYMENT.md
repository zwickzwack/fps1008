# Deployment Guide - Pico Pi 2 W Tester

## 📦 Package Contents

The tester module consists of these files:
```
src/tester/
├── main.py                         # Main orchestrator (lean)
├── hardware_config.py              # Pin and hardware configuration
├── oled_display.py                 # OLED display driver wrapper
├── test_functions.py               # Test routines and runner
├── utils.py                        # Utility functions
├── examples.py                     # Usage examples
├── __init__.py                     # Package initialization
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
└── requirements_micropython.txt    # Dependencies list
```

## 🔌 Hardware Setup

### Required Components
1. Raspberry Pi Pico Pi 2 W (or Pico W)
2. SSD1306 OLED Display (128x64, I2C)
3. 4 jumper wires
4. USB cable for programming

### Wiring Diagram
```
┌─────────────────────┐
│   OLED Display      │
│    SSD1306 I2C      │
│   (128x64 pixels)   │
└──────┬─┬─┬─┬────────┘
       │ │ │ │
       V S G 3
       C D N .
       C A D 3
       │ │ │ V
       │ │ │ │
┌──────┴─┴─┴─┴─────────────────────────┐
│      1 0 G 3                          │
│      │ │ N .                          │
│      │ │ D 3                          │
│      │ │   V                          │
│  ┌───┴─┴───┴───┐                     │
│  │ I2C Port    │                     │
│  │ GPIO 0,1    │   Raspberry Pi      │
│  └─────────────┘   Pico Pi 2 W      │
│                                       │
│              USB                      │
│          ┌────────┐                   │
└──────────┴────────┴───────────────────┘
           │        │
           └────────┘  To Computer
```

### Pin Connections
| OLED Pin | Pico Pin | GPIO | Description |
|----------|----------|------|-------------|
| VCC      | Pin 36   | 3.3V | Power supply |
| GND      | Pin 38   | GND  | Ground |
| SDA      | Pin 1    | GP0  | I2C Data |
| SCL      | Pin 2    | GP1  | I2C Clock |

## 💾 Software Installation

### Step 1: Install MicroPython Firmware

1. **Download firmware:**
   - Visit: https://micropython.org/download/rp2-pico-w/
   - Download latest `.uf2` file for Pico W/2W

2. **Flash firmware:**
   ```
   a) Hold BOOTSEL button on Pico
   b) Connect USB cable to computer
   c) Release BOOTSEL (Pico appears as USB drive)
   d) Drag .uf2 file to RPI-RP2 drive
   e) Pico will reboot automatically
   ```

### Step 2: Install Thonny IDE (Recommended)

**Windows/Mac/Linux:**
```bash
# Visit: https://thonny.org/
# Download and install for your OS
```

**Or use alternative:**
- Mu Editor: https://codewith.mu/
- PyCharm with MicroPython plugin
- VS Code with Pymakr extension

### Step 3: Install ssd1306 Driver

**Method A: Using Thonny (Easiest)**
```
1. Open Thonny IDE
2. Connect Pico Pi 2 W
3. Tools → Manage packages...
4. Search: "micropython-ssd1306"
5. Click "Install"
```

**Method B: Manual Installation**
```python
# 1. Download ssd1306.py from:
#    https://github.com/micropython/micropython/blob/master/drivers/display/ssd1306.py

# 2. In Thonny:
#    - Open ssd1306.py
#    - File → Save As
#    - Select "Raspberry Pi Pico"
#    - Save as "ssd1306.py" in root directory
```

**Method C: Using mpremote**
```bash
# Install mpremote
pip install mpremote

# Upload ssd1306.py
mpremote connect /dev/ttyACM0 fs cp ssd1306.py :ssd1306.py
```

### Step 4: Upload Tester Files

**Using Thonny IDE:**
```
For each file in src/tester/:
1. Open file in Thonny
2. File → Save As
3. Select "Raspberry Pi Pico"
4. Keep the same filename
5. Click "OK"

Upload in this order:
- hardware_config.py
- utils.py
- oled_display.py
- test_functions.py
- main.py
- (Optional) examples.py
```

**Using mpremote:**
```bash
cd src/tester/
mpremote connect /dev/ttyACM0 fs cp hardware_config.py :
mpremote connect /dev/ttyACM0 fs cp utils.py :
mpremote connect /dev/ttyACM0 fs cp oled_display.py :
mpremote connect /dev/ttyACM0 fs cp test_functions.py :
mpremote connect /dev/ttyACM0 fs cp main.py :
```

**Using ampy:**
```bash
pip install adafruit-ampy
cd src/tester/

# Upload files
ampy --port /dev/ttyACM0 put hardware_config.py
ampy --port /dev/ttyACM0 put utils.py
ampy --port /dev/ttyACM0 put oled_display.py
ampy --port /dev/ttyACM0 put test_functions.py
ampy --port /dev/ttyACM0 put main.py
```

## ▶️ Running the Tester

### First Run Test

1. **Connect via Thonny:**
   - Open Thonny
   - Select "MicroPython (Raspberry Pi Pico)" in bottom-right
   - Click "View → Files" to see Pico filesystem

2. **Verify files:**
   ```python
   import os
   print(os.listdir())
   # Should show: ssd1306.py, main.py, hardware_config.py, etc.
   ```

3. **Test display:**
   ```python
   from oled_display import OLEDDisplay
   display = OLEDDisplay()
   display.show_text("Hello!", 0, 0)
   ```

4. **Run main program:**
   ```python
   import main
   ```

### Auto-Start on Boot

To run automatically when Pico powers up:

**Method 1: Use boot.py**
```python
# Create boot.py with:
import main
```

**Method 2: Rename main.py**
```bash
# Rename main.py to boot.py
# It will auto-run on power-up
```

## 🧪 Verification Tests

### Test 1: I2C Connection
```python
from machine import I2C, Pin
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
devices = i2c.scan()
print("I2C devices found:", [hex(d) for d in devices])
# Expected: ['0x3c'] or ['0x3d']
```

### Test 2: Display Initialization
```python
from oled_display import OLEDDisplay
display = OLEDDisplay()
print("Display available:", display.is_available())
# Expected: True
```

### Test 3: Run Quick Test
```python
import main
# Should run test suite and show results on OLED
```

## 🔧 Customization

### Change I2C Pins
Edit `hardware_config.py`:
```python
I2C_SDA_PIN = 4  # Your custom SDA pin
I2C_SCL_PIN = 5  # Your custom SCL pin
```

### Change OLED Address
If display not detected, try alternate address:
```python
OLED_ADDRESS = 0x3D  # Try 0x3D instead of 0x3C
```

### Change Test Sequence
Edit `main.py`, modify the `run_quick_test()` function:
```python
def run_quick_test(display, test_runner):
    # Add/remove tests as needed
    test_runner.run_test("My Test", my_test_function)
```

## 📊 Operation Modes

Edit `main.py` to set default mode:
```python
mode = 1  # Quick Test (default)
# mode = 2  # Continuous Monitoring
# mode = 3  # Display Demo
```

## 🐛 Troubleshooting

### Problem: Display stays blank

**Solutions:**
1. Check wiring connections
2. Verify power supply (3.3V, not 5V!)
3. Check I2C address:
   ```python
   from machine import I2C, Pin
   i2c = I2C(0, sda=Pin(0), scl=Pin(1))
   print(i2c.scan())  # Try both 0x3C and 0x3D
   ```
4. Try different I2C frequency in `hardware_config.py`:
   ```python
   I2C_FREQ = 100000  # Slower speed for long wires
   ```

### Problem: ImportError: no module named 'ssd1306'

**Solutions:**
1. Install ssd1306.py (see Step 3 above)
2. Verify file is in root directory:
   ```python
   import os
   print('ssd1306.py' in os.listdir())
   ```

### Problem: OSError: [Errno 5] EIO

**Solutions:**
1. I2C communication error - check wiring
2. Wrong I2C address - try 0x3D instead of 0x3C
3. Add pull-up resistors (4.7kΩ) to SDA and SCL lines

### Problem: MemoryError

**Solutions:**
1. Run garbage collection:
   ```python
   import gc
   gc.collect()
   ```
2. Reduce buffer sizes in code
3. Limit concurrent operations

## 📱 Remote Access

### Via WiFi (Optional)

Add to `boot.py`:
```python
import network
import time

def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(ssid, password)
    
    max_wait = 10
    while max_wait > 0:
        if wlan.status() < 0 or wlan.status() >= 3:
            break
        max_wait -= 1
        time.sleep(1)
    
    if wlan.status() != 3:
        raise RuntimeError('WiFi connection failed')
    
    print('Connected! IP:', wlan.ifconfig()[0])

# Connect to your network
connect_wifi('YOUR_SSID', 'YOUR_PASSWORD')

# Then run main
import main
```

## 🔄 Updates

To update files:
1. Connect Pico via USB
2. Upload modified files via Thonny
3. Reset Pico or run `import main` again

## 📦 Backup

**Backup all files from Pico:**
```bash
# Using mpremote
mpremote connect /dev/ttyACM0 fs cp :main.py main.py.backup
mpremote connect /dev/ttyACM0 fs cp :hardware_config.py hardware_config.py.backup
# etc...
```

## 🎓 Next Steps

1. Read full documentation: `README.md`
2. Try examples: `examples.py`
3. Customize tests: `test_functions.py`
4. Build custom dashboards
5. Add sensors and expand functionality

## 📞 Support

- **Documentation**: See README.md and QUICKSTART.md
- **Examples**: See examples.py
- **Issues**: Create issue in repository
- **MicroPython docs**: https://docs.micropython.org/

---

**Deployment Complete! 🎉**
