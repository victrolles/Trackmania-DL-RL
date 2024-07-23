import win32gui
import win32con

def set_window_pos(window_name, x, y, width, height):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        # print(win32gui.GetWindowRect(hwnd))
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, width, height, 0)

# Example usage
name = "TrackMania Nations Forever (TMInterface 1.4.3)"
set_window_pos(name, -6, 0, 256, 256)  # Replace 'Google Chrome' with the exact window title
