"""Mouse-wheel hook (ui/wheel.py): Win32 window subclassing, exercised on a plain STATIC window made with ctypes."""
import ctypes
import sys

import pytest

from particle_mesh.ui import wheel as WH

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the wheel hook is Windows-only")

TITLE = "pps wheel hook test"


@pytest.fixture
def hwnd():
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                       wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    h = user32.CreateWindowExW(0, "STATIC", TITLE, 0, 0, 0, 64, 64, None, None, None, None)
    assert h
    yield h
    user32.DestroyWindow(h)


def _send(hwnd, msg, wparam=0, lparam=0):
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    return user32.SendMessageW(hwnd, msg, wparam, lparam)


def _wheel(hwnd, delta):
    _send(hwnd, WH.WM_MOUSEWHEEL, (delta & 0xFFFF) << 16)


def test_find_window_by_title_in_this_process(hwnd):
    assert WH.find_window(TITLE) == hwnd
    assert WH.find_window(TITLE + " (missing)") is None
    assert WH.attach_wheel(TITLE + " (missing)") is None


def test_hook_counts_notches_both_ways(hwnd):
    hook = WH.attach_wheel(TITLE)
    assert hook is not None and hook.attached
    _wheel(hwnd, 120)
    _wheel(hwnd, 240)
    assert hook.take() == 3.0
    assert hook.take() == 0.0                       # take() clears
    _wheel(hwnd, -120)
    _wheel(hwnd, 60)                                # high-resolution wheels send fractions of a notch
    assert hook.take() == -0.5
    hook.detach()
    _wheel(hwnd, 120)
    assert hook.take() == 0.0 and not hook.attached


def test_hook_passes_every_message_on(hwnd):
    """The subclass must chain to the original window procedure (GLFW, and through it ImGui, in the app)."""
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    hook = WH.attach_wheel(TITLE)
    text = ctypes.create_unicode_buffer("renamed through the hook")
    WM_SETTEXT = 0x000C
    _send(hwnd, WM_SETTEXT, 0, ctypes.addressof(text))
    buf = ctypes.create_unicode_buffer(64)
    user32.GetWindowTextW(hwnd, buf, 64)
    assert buf.value == "renamed through the hook"
    hook.detach()


def test_hook_detaches_when_the_window_is_destroyed():
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    user32.CreateWindowExW.restype = wintypes.HWND
    h = user32.CreateWindowExW(0, "STATIC", TITLE + " 2", 0, 0, 0, 8, 8, None, None, None, None)
    hook = WH.attach_wheel(TITLE + " 2")
    assert hook.attached
    user32.DestroyWindow(wintypes.HWND(h))
    assert not hook.attached                        # WM_NCDESTROY removed the subclass
