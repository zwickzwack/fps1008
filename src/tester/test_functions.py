"""
Test Functions Module
Contains various test routines and procedures for the Pico Pi 2 W tester
"""

import time
from utils import Timer, format_time, truncate_text

try:
    from machine import Pin
    import gc
    MICROPYTHON_AVAILABLE = True
except ImportError:
    MICROPYTHON_AVAILABLE = False
    print("Warning: MicroPython modules not available. Running in simulation mode.")


class TestRunner:
    """Manages and executes test routines"""
    
    def __init__(self, display):
        """
        Initialize test runner
        
        Args:
            display: OLEDDisplay instance
        """
        self.display = display
        self.test_results = {}
        self.timer = Timer()
    
    def run_test(self, test_name, test_func, *args, **kwargs):
        """
        Run a single test and display results
        
        Args:
            test_name (str): Name of the test
            test_func (callable): Test function to execute
            *args: Arguments to pass to test function
            **kwargs: Keyword arguments to pass to test function
            
        Returns:
            dict: Test result with status and data
        """
        print(f"Running test: {test_name}")
        self.display.show_test_status(test_name, "RUNNING", "")
        
        self.timer.start()
        
        try:
            result = test_func(*args, **kwargs)
            elapsed = self.timer.stop()
            
            status = "PASS" if result.get('success', False) else "FAIL"
            self.test_results[test_name] = {
                'status': status,
                'time': elapsed,
                'data': result
            }
            
            self.display.show_test_status(
                truncate_text(test_name, 16),
                status,
                f"Time: {elapsed:.2f}s"
            )
            
            time.sleep(2)  # Display result for 2 seconds
            
            return self.test_results[test_name]
            
        except Exception as e:
            elapsed = self.timer.stop()
            print(f"Test {test_name} failed with exception: {e}")
            
            self.test_results[test_name] = {
                'status': 'ERROR',
                'time': elapsed,
                'error': str(e)
            }
            
            self.display.show_test_status(
                truncate_text(test_name, 16),
                "ERROR",
                truncate_text(str(e), 16)
            )
            
            time.sleep(2)
            
            return self.test_results[test_name]
    
    def get_results_summary(self):
        """
        Get summary of all test results
        
        Returns:
            dict: Summary statistics
        """
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.test_results.values() if r['status'] == 'ERROR')
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors
        }


def test_display(display):
    """
    Test display functionality
    
    Args:
        display: OLEDDisplay instance
        
    Returns:
        dict: Test result
    """
    try:
        # Test clear
        display.clear()
        time.sleep(0.5)
        
        # Test text display
        display.show_text("Display Test", 0, 0)
        time.sleep(1)
        
        # Test multiline
        display.show_multiline([
            "Line 1",
            "Line 2",
            "Line 3"
        ])
        time.sleep(1)
        
        # Test centered text
        display.show_centered("Centered!")
        time.sleep(1)
        
        # Test progress bar
        for i in range(11):
            display.draw_progress_bar(i / 10.0)
            time.sleep(0.2)
        
        return {'success': True, 'message': 'Display test passed'}
    except Exception as e:
        return {'success': False, 'message': f'Display test failed: {e}'}


def test_memory():
    """
    Test memory availability
    
    Returns:
        dict: Test result with memory information
    """
    try:
        if MICROPYTHON_AVAILABLE:
            gc.collect()
            free_mem = gc.mem_free()
            allocated_mem = gc.mem_alloc()
            
            return {
                'success': True,
                'free_memory': free_mem,
                'allocated_memory': allocated_mem,
                'message': f'Free: {free_mem} bytes'
            }
        else:
            return {
                'success': True,
                'message': 'Memory test (simulated)'
            }
    except Exception as e:
        return {'success': False, 'message': f'Memory test failed: {e}'}


def test_led_blink(pin=25, duration=2):
    """
    Test LED blinking
    
    Args:
        pin (int): LED pin number
        duration (int): Test duration in seconds
        
    Returns:
        dict: Test result
    """
    try:
        if MICROPYTHON_AVAILABLE:
            led = Pin(pin, Pin.OUT)
            blinks = 0
            end_time = time.time() + duration
            
            while time.time() < end_time:
                led.toggle()
                time.sleep(0.2)
                blinks += 1
            
            led.value(0)  # Turn off LED
            
            return {
                'success': True,
                'blinks': blinks // 2,
                'message': f'LED blinked {blinks // 2} times'
            }
        else:
            return {
                'success': True,
                'message': 'LED test (simulated)'
            }
    except Exception as e:
        return {'success': False, 'message': f'LED test failed: {e}'}


def test_i2c_scan(display):
    """
    Scan I2C bus for devices
    
    Args:
        display: OLEDDisplay instance
        
    Returns:
        dict: Test result with list of I2C devices
    """
    try:
        if MICROPYTHON_AVAILABLE and hasattr(display, 'i2c'):
            devices = display.i2c.scan()
            device_addrs = [hex(dev) for dev in devices]
            
            return {
                'success': True,
                'devices': device_addrs,
                'count': len(devices),
                'message': f'Found {len(devices)} I2C devices'
            }
        else:
            return {
                'success': True,
                'message': 'I2C scan (simulated)',
                'devices': ['0x3c']
            }
    except Exception as e:
        return {'success': False, 'message': f'I2C scan failed: {e}'}


def test_comprehensive(display):
    """
    Run comprehensive system test
    
    Args:
        display: OLEDDisplay instance
        
    Returns:
        dict: Test result with all sub-test results
    """
    results = {}
    
    # Run individual tests
    results['display'] = test_display(display)
    results['memory'] = test_memory()
    results['led'] = test_led_blink()
    results['i2c'] = test_i2c_scan(display)
    
    # Determine overall success
    all_success = all(r.get('success', False) for r in results.values())
    
    return {
        'success': all_success,
        'sub_tests': results,
        'message': 'All tests passed' if all_success else 'Some tests failed'
    }
