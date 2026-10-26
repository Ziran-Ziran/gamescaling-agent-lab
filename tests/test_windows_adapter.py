import unittest
from types import SimpleNamespace
from unittest.mock import patch

from controller.windows_adapter import validate_capture_rect, normalized_mouse
from controller.windows_adapter import emergency_key_pressed


class WindowGeometryTests(unittest.TestCase):
    def test_hotkey_only_reserves_f8_or_f24(self):
        from controller.windows_adapter import EmergencyHotkey
        for vk in [0x41, 0, None, True, 'F8']:
            with self.subTest(vk=vk), self.assertRaises(ValueError):
                EmergencyHotkey(vk=vk)

    def test_elevated_game_blocks_lower_privilege_input(self):
        from controller.windows_adapter import require_input_access
        with self.assertRaisesRegex(RuntimeError, 'administrator'):
            require_input_access({'controller_integrity': 8192, 'game_integrity': 12288})

    def test_equal_integrity_allows_input_preflight(self):
        from controller.windows_adapter import require_input_access
        require_input_access({'controller_integrity': 12288, 'game_integrity': 12288})

    def test_unknown_integrity_cannot_enable_input(self):
        from controller.windows_adapter import require_input_access
        with self.assertRaisesRegex(RuntimeError, 'Cannot verify'):
            require_input_access({'controller_integrity': 8192, 'game_integrity': None})

    def test_invalid_or_minimized_rect_is_rejected(self):
        for rect in [(0, 0, 0, 0), (100, 20, 50, 40), (0, 0, 40, 20)]:
            with self.assertRaises(ValueError):
                validate_capture_rect(rect)

    def test_negative_monitor_coordinates_are_supported(self):
        self.assertEqual(validate_capture_rect((-1280, 20, 0, 740)), (-1280, 20, 0, 740))

    def test_mouse_uses_actual_client_size_and_hard_pixel_cap(self):
        self.assertEqual(normalized_mouse((10, -10, 0), (1280, 720)), (13, -7))
        self.assertEqual(normalized_mouse((1000, -1000, 0), (3840, 2160)), (20, -20))

    def test_nonfinite_mouse_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_mouse((float('nan'), 0, 0), (1280, 720))

    def test_quick_emergency_tap_is_not_lost_between_polls(self):
        with patch('controller.windows_adapter.USER', SimpleNamespace(GetAsyncKeyState=lambda _: 1)):
            self.assertTrue(emergency_key_pressed())


if __name__ == '__main__':
    unittest.main()
