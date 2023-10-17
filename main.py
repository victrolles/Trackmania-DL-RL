import cv2
import numpy as np
from PIL import ImageGrab
import ctypes
import time

DIK_Z = 0x2C
DIK_Q = 0x10
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def screen_record():
    while(True):
        time1 = time.time()
        printscreen =  np.array(ImageGrab.grab(bbox=(1920,-300,2349,-58), include_layered_windows=False, all_screens=True))
        time2 = time.time()
        processed_img = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        time3 = time.time()
        processed_img = cv2.Canny(processed_img, threshold1=50, threshold2=150)
        time4 = time.time()
        cv2.imshow('caption',processed_img)
        time5 = time.time()
        cv2.moveWindow("caption", 2880,-165)
        time6 = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        time7 = time.time()
        print("Time to grab screen: {:.3f}".format(time7-time1), "grab : {:.3f}".format(time2-time1), "gray : {:.3f}".format(time3-time2), "canny : {:.3f}".format(time4-time3), "show : {:.3f}".format(time5-time4), "move : {:.3f}".format(time6-time5), "waitkey : {:.3f}".format(time7-time6), end="\r")
screen_record()