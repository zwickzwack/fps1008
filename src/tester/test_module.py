#!/usr/bin/env python3
"""
Test Script - Verifies tester module functionality
Run this on your development machine before deploying to Pico
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import hardware_config
        print("  ✓ hardware_config")
        
        import utils
        print("  ✓ utils")
        
        import oled_display
        print("  ✓ oled_display")
        
        import test_functions
        print("  ✓ test_functions")
        
        import main
        print("  ✓ main")
        
        print("\n✅ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_configuration():
    """Test configuration functions"""
    print("Testing configuration...")
    try:
        from hardware_config import get_i2c_config, get_oled_config
        from hardware_config import I2C_SDA_PIN, I2C_SCL_PIN, OLED_WIDTH, OLED_HEIGHT
        
        i2c_cfg = get_i2c_config()
        assert 'sda' in i2c_cfg, "Missing SDA in config"
        assert 'scl' in i2c_cfg, "Missing SCL in config"
        print(f"  ✓ I2C Config: SDA={I2C_SDA_PIN}, SCL={I2C_SCL_PIN}")
        
        oled_cfg = get_oled_config()
        assert 'width' in oled_cfg, "Missing width in config"
        assert 'height' in oled_cfg, "Missing height in config"
        print(f"  ✓ OLED Config: {OLED_WIDTH}x{OLED_HEIGHT}")
        
        print("\n✅ Configuration tests passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}\n")
        return False


def test_display_class():
    """Test OLEDDisplay class"""
    print("Testing OLEDDisplay class...")
    try:
        from oled_display import OLEDDisplay
        
        display = OLEDDisplay()
        print("  ✓ Display initialized")
        
        # Test methods (in simulation mode)
        display.clear()
        print("  ✓ clear() method")
        
        display.show_text("Test", 0, 0)
        print("  ✓ show_text() method")
        
        display.show_multiline(["Line 1", "Line 2"])
        print("  ✓ show_multiline() method")
        
        display.show_centered("Center")
        print("  ✓ show_centered() method")
        
        display.draw_progress_bar(0.5)
        print("  ✓ draw_progress_bar() method")
        
        display.show_test_status("Test", "PASS")
        print("  ✓ show_test_status() method")
        
        print("\n✅ Display class tests passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Display test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions"""
    print("Testing utility functions...")
    try:
        from utils import (
            format_time, format_bytes, truncate_text,
            Timer, RollingAverage, map_value, clamp
        )
        
        # Test format_time
        assert format_time(30) == "30s"
        assert format_time(90) == "1m 30s"
        print("  ✓ format_time()")
        
        # Test format_bytes
        result = format_bytes(1024)
        assert "1.00 KB" in result
        print("  ✓ format_bytes()")
        
        # Test truncate_text
        result = truncate_text("Very long text", 10)
        assert len(result) <= 10
        print("  ✓ truncate_text()")
        
        # Test Timer
        timer = Timer()
        timer.start()
        elapsed = timer.stop()
        assert elapsed >= 0
        print("  ✓ Timer class")
        
        # Test RollingAverage
        avg = RollingAverage(size=3)
        avg.add(1)
        avg.add(2)
        avg.add(3)
        assert avg.get_average() == 2.0
        print("  ✓ RollingAverage class")
        
        # Test map_value
        result = map_value(5, 0, 10, 0, 100)
        assert result == 50
        print("  ✓ map_value()")
        
        # Test clamp
        assert clamp(15, 0, 10) == 10
        assert clamp(-5, 0, 10) == 0
        assert clamp(5, 0, 10) == 5
        print("  ✓ clamp()")
        
        print("\n✅ Utility tests passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Utility test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_test_functions():
    """Test test function module"""
    print("Testing test functions...")
    try:
        from test_functions import (
            TestRunner, test_display, test_memory,
            test_led_blink, test_i2c_scan
        )
        from oled_display import OLEDDisplay
        
        display = OLEDDisplay()
        runner = TestRunner(display)
        print("  ✓ TestRunner initialized")
        
        # Run tests (in simulation mode)
        result = test_memory()
        assert 'success' in result
        print("  ✓ test_memory()")
        
        result = test_led_blink(25, 1)
        assert 'success' in result
        print("  ✓ test_led_blink()")
        
        result = test_i2c_scan(display)
        assert 'success' in result
        print("  ✓ test_i2c_scan()")
        
        # Test runner execution
        runner.run_test("Memory", test_memory)
        summary = runner.get_results_summary()
        assert summary['total'] == 1
        print("  ✓ TestRunner.run_test()")
        
        print("\n✅ Test function tests passed!\n")
        return True
    except Exception as e:
        print(f"\n❌ Test function test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Verify all required files exist"""
    print("Checking file structure...")
    
    required_files = [
        'main.py',
        'hardware_config.py',
        'oled_display.py',
        'test_functions.py',
        'utils.py',
        '__init__.py',
        'README.md',
        'QUICKSTART.md',
        'DEPLOYMENT.md',
        'examples.py',
        'requirements_micropython.txt'
    ]
    
    current_dir = os.path.dirname(__file__)
    all_present = True
    
    for filename in required_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            all_present = False
    
    if all_present:
        print("\n✅ All required files present!\n")
    else:
        print("\n❌ Some files are missing!\n")
    
    return all_present


def main():
    """Run all tests"""
    print("=" * 60)
    print("Pico Pi 2 W Tester - Verification Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("File Structure", test_file_structure()))
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Display Class", test_display_class()))
    results.append(("Utilities", test_utilities()))
    results.append(("Test Functions", test_test_functions()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! Module is ready for deployment.")
        print("\nNext steps:")
        print("1. Read QUICKSTART.md for 5-minute setup")
        print("2. Read DEPLOYMENT.md for detailed deployment guide")
        print("3. Upload files to your Pico Pi 2 W")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
