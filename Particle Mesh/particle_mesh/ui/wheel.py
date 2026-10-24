"""
Mouse-wheel input for the GGUI window. Taichi 1.7 GGUI (ti.ui.Window) reports no wheel events, so on Windows the
window procedure of the GGUI (GLFW) window is subclassed (comctl32 SetWindowSubclass) to count WM_MOUSEWHEEL
notches; every message, the wheel included, is still passed on to GLFW (and so to ImGui). Elsewhere, or if the
window can't be found, attach_wheel() returns None and the wheel stays unused.

The hook runs on the thread that pumps the window's messages (the main thread, inside GGUI's event polling), so
it only adds to a counter; InputController reads it once per frame through WheelHook.take().
"""
import atexit
import ctypes
import os
import sys

WM_MOUSEWHEEL = 0x020A
WM_NCDESTROY = 0x0082
WHEEL_DELTA = 120          # one notch
_SUBCLASS_ID = 0x5050      # any id unique to this hook on the window


class WheelHook:
    """Counts wheel notches on one window (positive = away from the user). take() returns and clears them."""

    def __init__(self, hwnd):
        from ctypes import wintypes
        self.hwnd = hwnd
        self._notches = 0.0
        self._attached = False
        comctl = ctypes.WinDLL("comctl32")
        lresult = ctypes.c_ssize_t
        proc_type = ctypes.WINFUNCTYPE(lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                       ctypes.c_size_t, ctypes.c_size_t)
        self._set = comctl.SetWindowSubclass
        self._set.argtypes = [wintypes.HWND, proc_type, ctypes.c_size_t, ctypes.c_size_t]
        self._set.restype = wintypes.BOOL
        self._remove = comctl.RemoveWindowSubclass
        self._remove.argtypes = [wintypes.HWND, proc_type, ctypes.c_size_t]
        self._remove.restype = wintypes.BOOL
        self._default = comctl.DefSubclassProc
        self._default.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self._default.restype = lresult
        # Keep the ctypes callback alive as long as the window may call it
        self._proc = proc_type(self._window_proc)

    def _window_proc(self, hwnd, msg, wparam, lparam, subclass_id, ref_data):
        if msg == WM_MOUSEWHEEL:
            # The high word of wParam is the signed wheel delta, in multiples of WHEEL_DELTA per notch
            self._notches += ctypes.c_short((wparam >> 16) & 0xFFFF).value / WHEEL_DELTA
        elif msg == WM_NCDESTROY:
            self.detach()
        return self._default(hwnd, msg, wparam, lparam)

    def attach(self):
        self._attached = bool(self._set(self.hwnd, self._proc, _SUBCLASS_ID, 0))
        if self._attached:
            # Remove the subclass before interpreter teardown, in case the window outlives Python's callback
            atexit.register(self.detach)
        return self._attached

    def detach(self):
        if self._attached:
            self._attached = False
            self._remove(self.hwnd, self._proc, _SUBCLASS_ID)

    @property
    def attached(self):
        return self._attached

    def take(self):
        """Wheel notches since the last call (fractional for high-resolution wheels and touchpads)."""
        notches, self._notches = self._notches, 0.0
        return notches


def find_window(title):
    """HWND of this process's top-level window with the given title, or None."""
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32")
    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    pid = os.getpid()
    found = []

    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            if buf.value == title:
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(enum_proc_type(visit), 0)
    return found[0] if found else None


def attach_wheel(title):
    """A WheelHook on this process's window with that title, or None (not Windows, no such window, or the
    subclass failed)."""
    if sys.platform != "win32":
        return None
    try:
        hwnd = find_window(title)
        if hwnd is None:
            return None
        hook = WheelHook(hwnd)
        return hook if hook.attach() else None
    except (OSError, AttributeError):
        return None
