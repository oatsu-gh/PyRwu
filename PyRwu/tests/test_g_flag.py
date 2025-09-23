import unittest
import numpy as np
import sys
import os

# Add the PyRwu directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from effects.g_flag import GFlag


class MockParams:
    """Mock params object for testing GFlag"""
    def __init__(self, sp, g_value=0):
        self.sp = sp
        self.framerate = 44100
        self.f0 = np.ones((sp.shape[0],))  # Mock f0 data
        
        # Mock flags
        class MockFlag:
            def __init__(self, value):
                self.value = value
        
        class MockFlags:
            def __init__(self, g_value):
                self.params = {"g": MockFlag(g_value)}
        
        self.flags = MockFlags(g_value)


class TestGFlag(unittest.TestCase):
    
    def test_zero_g_value(self):
        """Test that g=0 returns original sp without modification"""
        sp_data = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ])
        params = MockParams(sp_data, g_value=0)
        result = GFlag.apply(params)
        
        # Should return original sp unchanged
        np.testing.assert_array_equal(result, sp_data)
    
    def test_small_values_no_nan(self):
        """Test that very small sp values don't produce NaN"""
        sp_data = np.array([
            [7.732958562173011e-13, 0.9582792071350017, 1e-15, 0.1],
            [1e-20, 0.5, 0.0, 0.3]
        ])
        params = MockParams(sp_data, g_value=20)
        result = GFlag.apply(params)
        
        # Should not contain NaN or inf values
        self.assertFalse(np.isnan(result).any(), "Result contains NaN values")
        self.assertFalse(np.isinf(result).any(), "Result contains inf values")
        
        # Should preserve shape
        self.assertEqual(result.shape, sp_data.shape)
        
        # All values should be positive (after processing)
        self.assertTrue(np.all(result > 0), "Result contains non-positive values")
    
    def test_zero_values_handling(self):
        """Test that zero sp values are handled properly"""
        sp_data = np.array([
            [0.0, 0.1, 0.0, 0.2],
            [0.3, 0.0, 0.4, 0.0]
        ])
        params = MockParams(sp_data, g_value=50)
        result = GFlag.apply(params)
        
        # Should not contain NaN or inf values
        self.assertFalse(np.isnan(result).any(), "Result contains NaN values")
        self.assertFalse(np.isinf(result).any(), "Result contains inf values")
        
        # All values should be positive (zeros replaced with minimum)
        self.assertTrue(np.all(result > 0), "Result contains non-positive values")
    
    def test_negative_g_value(self):
        """Test negative g values (female/young voice effect)"""
        sp_data = np.array([
            [1e-10, 0.5, 1e-12, 0.3],
            [0.1, 1e-15, 0.2, 0.4]
        ])
        params = MockParams(sp_data, g_value=-30)
        result = GFlag.apply(params)
        
        # Should not contain NaN or inf values
        self.assertFalse(np.isnan(result).any(), "Result contains NaN values")
        self.assertFalse(np.isinf(result).any(), "Result contains inf values")
        
        # Should preserve shape
        self.assertEqual(result.shape, sp_data.shape)
    
    def test_positive_g_value(self):
        """Test positive g values (male/adult voice effect)"""
        sp_data = np.array([
            [1e-13, 0.8, 1e-14, 0.6],
            [0.2, 1e-16, 0.1, 0.9]
        ])
        params = MockParams(sp_data, g_value=80)
        result = GFlag.apply(params)
        
        # Should not contain NaN or inf values
        self.assertFalse(np.isnan(result).any(), "Result contains NaN values")
        self.assertFalse(np.isinf(result).any(), "Result contains inf values")
        
        # Should preserve shape
        self.assertEqual(result.shape, sp_data.shape)
    
    def test_interpolation_robustness(self):
        """Test that interpolation functions handle edge cases"""
        # Test _interp1 directly with problematic input
        x = np.array([0.0, 0.0, 1.0, 2.0])  # Duplicate values that cause h=0
        y = np.array([1.0, 1.0, 2.0, 3.0])
        xi = np.array([0.5, 1.5])
        yi = np.zeros_like(xi)
        
        result = GFlag._interp1(x, y, len(x), xi, len(xi), yi)
        
        # Should not contain NaN or inf values
        self.assertFalse(np.isnan(result).any(), "Interpolation result contains NaN")
        self.assertFalse(np.isinf(result).any(), "Interpolation result contains inf")
    
    def test_histc_robustness(self):
        """Test that _histc function handles edge cases"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        index = np.zeros(len(edges), dtype=np.int32)
        
        result = GFlag._histc(x, len(x), edges, len(edges), index)
        
        # Should not contain invalid indices
        self.assertTrue(np.all(result >= 0), "histc returned negative indices")
        self.assertTrue(np.all(result < len(x)), "histc returned out-of-bounds indices")


if __name__ == '__main__':
    unittest.main()