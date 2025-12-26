# Quick Start Guide - Pico Pi 2 W Tester

## 🚀 5-Minute Setup

### Prerequisites
- Raspberry Pi Pico Pi 2 W with MicroPython installed
- SSD1306 OLED Display (128x64, I2C)
- Thonny IDE installed on your computer

### Hardware Connections
```
OLED Display    →    Pico Pi 2 W
─────────────────────────────────
VCC             →    3.3V (Pin 36)
GND             →    GND (Pin 38)
SDA             →    GPIO 0 (Pin 1)
SCL             →    GPIO 1 (Pin 2)
```

### Software Setup

#### Step 1: Install ssd1306 Driver
1. Open Thonny IDE
2. Connect your Pico Pi 2 W
3. Go to `Tools` → `Manage packages`
4. Search for "micropython-ssd1306"
5. Click "Install"

#### Step 2: Upload Files
Upload these files to your Pico (root directory):
- `main.py`
- `hardware_config.py`
- `oled_display.py`
- `test_functions.py`
- `utils.py`

#### Step 3: Run
In Thonny, open `main.py` and click Run (F5) or press the green play button.

## 📺 What You'll See

The OLED will display:
1. **Startup Screen** - "Pico Tester v1.0"
2. **Test Sequence** - Each test running with status
3. **Test Summary** - Final results with pass/fail counts

## 🎛️ Changing Operation Mode

Edit `main.py`, find this line:
```python
mode = 1  # Default mode
```

Change to:
- `mode = 1` - Quick Test (runs all tests once)
- `mode = 2` - Continuous Monitoring (real-time stats)
- `mode = 3` - Display Demo (showcase features)

## 🔍 Verification

### Check I2C Connection
Run in Thonny shell:
```python
from machine import I2C, Pin
i2c = I2C(0, sda=Pin(0), scl=Pin(1))
devices = i2c.scan()
print([hex(d) for d in devices])
# Should show: ['0x3c'] or ['0x3d']
```

### Manual Display Test
```python
from oled_display import OLEDDisplay
display = OLEDDisplay()
display.show_text("Hello!", 0, 0)
```

## ⚙️ Configuration

### Different I2C Pins
Edit `hardware_config.py`:
```python
I2C_SDA_PIN = 4  # Your SDA pin
I2C_SCL_PIN = 5  # Your SCL pin
```

### Different OLED Address
If display doesn't work, try:
```python
OLED_ADDRESS = 0x3D  # Alternative address
```

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Display blank | Check connections, verify I2C address |
| Import error | Ensure all files uploaded, install ssd1306 |
| Program stops | Check serial output for error messages |
| No I2C devices found | Verify VCC/GND connected, check wiring |

## 📖 Next Steps

- Read full `README.md` for detailed documentation
- Modify tests in `test_functions.py`
- Add custom functionality
- Explore continuous monitoring mode

## 💡 Tips

1. **Auto-run on boot**: Rename `main.py` to `boot.py`
2. **Save power**: Add sleep delays in monitoring mode
3. **Debugging**: Use `print()` statements - view in Thonny shell
4. **Backup**: Keep copies of working files before modifications

---

Need help? Check the main README.md or create an issue in the repository.
