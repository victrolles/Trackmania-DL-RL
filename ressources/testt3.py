import cv2
from PIL import ImageGrab
import numpy as np
import win32gui
import win32con

def set_window_pos(window_name, x, y, width, height):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        print(win32gui.GetWindowRect(hwnd))
        # win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, width, height, 0)

 

while True:
    window_name = 'window'
    # Grab BGR image from the screen
    screen = np.array(ImageGrab.grab(bbox=(0, 60, 256, 316)))
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    cv2.imshow(window_name, gray_image)

    pixel_matrix = np.array(gray_image)
    print("Shape of pixel matrix:", pixel_matrix.shape)
    
    # Set the window position (x, y)
    # cv2.moveWindow(window_name, -17, 372)  # Move the window to position (100, 100)

    # Example usage
    # name = "TrackMania Nations Forever (TMInterface 1.4.3)"
    # set_window_pos(window_name, -6, 0, 775, 414)

    # Set the window size (width, height)
    # cv2.resizeWindow(window_name, 776, 411)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

