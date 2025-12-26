"""
Main Script for Pico Pi 2 W Tester with OLED Display
This is a lean main script that imports and orchestrates modular components

Usage:
    - Upload all files in src/tester/ to your Pico Pi 2 W
    - Ensure ssd1306.py library is installed on the Pico
    - Run this script as main.py on the Pico
"""

import time
import sys

# Import custom modules
from oled_display import OLEDDisplay
from test_functions import TestRunner, test_display, test_memory, test_led_blink, test_i2c_scan
from utils import format_time, Timer
from hardware_config import LED_PIN

try:
    from machine import Pin
    MICROPYTHON_AVAILABLE = True
except ImportError:
    MICROPYTHON_AVAILABLE = False
    print("Running in simulation mode (not on MicroPython)")


def show_startup_screen(display):
    """Display startup screen"""
    display.clear()
    display.show_multiline([
        "Pico Tester",
        "v1.0",
        "",
        "Initializing...",
    ])
    time.sleep(2)


def run_quick_test(display, test_runner):
    """Run a quick test sequence"""
    print("\n=== Running Quick Test Sequence ===")
    
    # Test 1: Display
    test_runner.run_test("Display", test_display, display)
    
    # Test 2: Memory
    test_runner.run_test("Memory", test_memory)
    
    # Test 3: LED
    test_runner.run_test("LED Blink", test_led_blink, LED_PIN, 2)
    
    # Test 4: I2C Scan
    test_runner.run_test("I2C Scan", test_i2c_scan, display)
    
    # Show summary
    show_test_summary(display, test_runner)


def run_continuous_monitoring(display):
    """Run continuous monitoring mode"""
    print("\n=== Continuous Monitoring Mode ===")
    
    timer = Timer()
    timer.start()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            elapsed = timer.get_elapsed()
            
            # Display system status
            lines = [
                "MONITORING",
                f"Uptime: {format_time(int(elapsed))}",
                f"Iter: {iteration}",
            ]
            
            if MICROPYTHON_AVAILABLE:
                import gc
                gc.collect()
                free_mem = gc.mem_free()
                lines.append(f"Free: {free_mem//1024}KB")
            
            display.show_multiline(lines)
            
            # Update every 5 seconds
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        display.show_centered("Stopped")
        time.sleep(1)


def show_test_summary(display, test_runner):
    """Display test summary"""
    summary = test_runner.get_results_summary()
    
    display.clear()
    display.show_multiline([
        "TEST SUMMARY",
        "-" * 16,
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Errors: {summary['errors']}",
    ])
    
    print("\n=== Test Summary ===")
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    
    time.sleep(5)


def show_menu(display):
    """Display main menu"""
    display.show_multiline([
        "MAIN MENU",
        "-" * 16,
        "1: Quick Test",
        "2: Monitor",
        "3: Display Demo",
        "",
    ])


def run_display_demo(display):
    """Run display demonstration"""
    print("\n=== Display Demo ===")
    
    # Demo 1: Text positions
    display.show_text("Top Left", 0, 0)
    time.sleep(2)
    
    # Demo 2: Centered text
    display.show_centered("Centered Text")
    time.sleep(2)
    
    # Demo 3: Multiple lines
    display.show_multiline([
        "Multi-line",
        "Text Demo",
        "Line 3",
        "Line 4",
        "Line 5",
        "Line 6"
    ])
    time.sleep(2)
    
    # Demo 4: Progress bar animation
    for i in range(11):
        progress = i / 10.0
        display.draw_progress_bar(progress)
        time.sleep(0.3)
    
    display.show_centered("Demo Complete!")
    time.sleep(2)


def main():
    """Main entry point"""
    print("\n" + "=" * 40)
    print("Pico Pi 2 W Tester with OLED Display")
    print("=" * 40 + "\n")
    
    # Initialize display
    print("Initializing OLED display...")
    display = OLEDDisplay()
    
    if not display.is_available() and MICROPYTHON_AVAILABLE:
        print("ERROR: Could not initialize OLED display!")
        print("Please check I2C connections and configuration.")
        return
    
    # Show startup screen
    show_startup_screen(display)
    
    # Initialize test runner
    test_runner = TestRunner(display)
    
    # Main operation modes
    print("\nSelect operation mode:")
    print("1. Run Quick Test")
    print("2. Continuous Monitoring")
    print("3. Display Demo")
    print("4. Exit")
    
    # For automation, default to quick test
    # In interactive mode, you can add input handling here
    mode = 1  # Default mode
    
    try:
        if mode == 1:
            run_quick_test(display, test_runner)
        elif mode == 2:
            run_continuous_monitoring(display)
        elif mode == 3:
            run_display_demo(display)
        else:
            print("Invalid mode selected")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        display.clear()
        display.show_centered("Goodbye!")
        time.sleep(1)
        display.clear()
    
    except Exception as e:
        print(f"\n\nERROR: {e}")
        if display.is_available():
            display.show_multiline([
                "ERROR",
                "",
                str(e)[:16]
            ])
            time.sleep(3)
    
    finally:
        print("\nTester stopped.")
        if display.is_available():
            display.clear()


if __name__ == "__main__":
    main()
