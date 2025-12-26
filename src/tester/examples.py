"""
Example Usage - Pico Pi 2 W Tester
Demonstrates how to use the tester modules

This file shows various ways to use the tester framework
"""

# Example 1: Basic Display Usage
def example_basic_display():
    """Show basic display operations"""
    from oled_display import OLEDDisplay
    
    display = OLEDDisplay()
    
    # Simple text
    display.show_text("Hello World!", 0, 0)
    
    # Centered text
    display.show_centered("Centered Text")
    
    # Multiple lines
    display.show_multiline([
        "Line 1",
        "Line 2",
        "Line 3"
    ])


# Example 2: Using the Test Runner
def example_test_runner():
    """Demonstrate test runner usage"""
    from oled_display import OLEDDisplay
    from test_functions import TestRunner, test_display, test_memory
    
    display = OLEDDisplay()
    runner = TestRunner(display)
    
    # Run individual tests
    runner.run_test("Display", test_display, display)
    runner.run_test("Memory", test_memory)
    
    # Get summary
    summary = runner.get_results_summary()
    print(f"Passed: {summary['passed']}/{summary['total']}")


# Example 3: Custom Test Function
def example_custom_test():
    """Create and run a custom test"""
    from oled_display import OLEDDisplay
    from test_functions import TestRunner
    
    def my_custom_test():
        """Custom test that always passes"""
        # Your test logic here
        return {
            'success': True,
            'message': 'Custom test passed',
            'data': {'value': 42}
        }
    
    display = OLEDDisplay()
    runner = TestRunner(display)
    
    result = runner.run_test("Custom", my_custom_test)
    print(f"Test result: {result}")


# Example 4: Progress Bar
def example_progress_bar():
    """Show progress bar animation"""
    import time
    from oled_display import OLEDDisplay
    
    display = OLEDDisplay()
    
    for i in range(101):
        progress = i / 100.0
        display.draw_progress_bar(progress)
        time.sleep(0.05)
    
    display.show_centered("Complete!")


# Example 5: Using Utilities
def example_utilities():
    """Demonstrate utility functions"""
    from utils import Timer, format_time, RollingAverage, truncate_text
    import time
    
    # Timer usage
    timer = Timer()
    timer.start()
    time.sleep(1)
    elapsed = timer.stop()
    print(f"Elapsed: {format_time(int(elapsed))}")
    
    # Rolling average
    avg = RollingAverage(size=5)
    for i in range(10):
        avg.add(i)
        print(f"Average: {avg.get_average():.2f}")
    
    # Text truncation
    long_text = "This is a very long text that needs truncation"
    short = truncate_text(long_text, 20)
    print(f"Truncated: {short}")


# Example 6: System Monitoring
def example_monitoring():
    """Show continuous system monitoring"""
    import time
    from oled_display import OLEDDisplay
    from utils import format_time, Timer
    
    display = OLEDDisplay()
    timer = Timer()
    timer.start()
    
    for iteration in range(10):  # Run 10 iterations
        elapsed = timer.get_elapsed()
        
        display.show_multiline([
            "MONITORING",
            f"Uptime: {format_time(int(elapsed))}",
            f"Iteration: {iteration + 1}",
            "",
            "Press Ctrl+C to stop"
        ])
        
        time.sleep(2)


# Example 7: Configuration Access
def example_configuration():
    """Show how to access configuration"""
    from hardware_config import get_i2c_config, get_oled_config
    from hardware_config import OLED_WIDTH, OLED_HEIGHT
    
    i2c_cfg = get_i2c_config()
    oled_cfg = get_oled_config()
    
    print("I2C Configuration:")
    print(f"  SDA: GPIO {i2c_cfg['sda']}")
    print(f"  SCL: GPIO {i2c_cfg['scl']}")
    print(f"  Frequency: {i2c_cfg['freq']} Hz")
    
    print(f"\nOLED Display: {OLED_WIDTH}x{OLED_HEIGHT} pixels")
    print(f"I2C Address: 0x{oled_cfg['address']:02X}")


# Example 8: Error Handling
def example_error_handling():
    """Demonstrate error handling in tests"""
    from oled_display import OLEDDisplay
    from test_functions import TestRunner
    
    def test_that_fails():
        """A test that intentionally fails"""
        return {
            'success': False,
            'message': 'This test failed on purpose'
        }
    
    def test_that_errors():
        """A test that raises an exception"""
        raise ValueError("Something went wrong!")
    
    display = OLEDDisplay()
    runner = TestRunner(display)
    
    # Run tests and handle results
    result1 = runner.run_test("Fail Test", test_that_fails)
    print(f"Test 1 status: {result1['status']}")
    
    result2 = runner.run_test("Error Test", test_that_errors)
    print(f"Test 2 status: {result2['status']}")


# Example 9: Complete Workflow
def example_complete_workflow():
    """Complete example workflow"""
    import time
    from oled_display import OLEDDisplay
    from test_functions import (
        TestRunner, 
        test_display, 
        test_memory,
        test_led_blink,
        test_i2c_scan
    )
    
    # Initialize
    print("Initializing...")
    display = OLEDDisplay()
    runner = TestRunner(display)
    
    # Startup message
    display.show_centered("Starting Tests...")
    time.sleep(1)
    
    # Run test suite
    tests = [
        ("Display", test_display, display),
        ("Memory", test_memory),
        ("LED", test_led_blink, 25, 1),
        ("I2C Scan", test_i2c_scan, display),
    ]
    
    for test_name, test_func, *args in tests:
        runner.run_test(test_name, test_func, *args)
    
    # Show summary
    summary = runner.get_results_summary()
    display.show_multiline([
        "RESULTS",
        "-" * 16,
        f"Total: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
    ])
    
    print("\nTest suite completed!")


# Main execution
if __name__ == "__main__":
    print("Pico Pi 2 W Tester - Usage Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_basic_display()")
    print("2. example_test_runner()")
    print("3. example_custom_test()")
    print("4. example_progress_bar()")
    print("5. example_utilities()")
    print("6. example_monitoring()")
    print("7. example_configuration()")
    print("8. example_error_handling()")
    print("9. example_complete_workflow()")
    print("\nRun any example by calling its function.")
    print("\nExample:")
    print("  from examples import example_basic_display")
    print("  example_basic_display()")
