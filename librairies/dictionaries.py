from enum import Enum

class Rd(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"

INPUT = [
            {  # 0 Up
                "left": False,
                "right": False,
                "accelerate": True,
                "brake": False,
            },
            {  # 1 Left
                "left": True,
                "right": False,
                "accelerate": False,
                "brake": False,
            },
            {  # 2 Right
                "left": False,
                "right": True,
                "accelerate": False,
                "brake": False,
            },
            {  # 3 Up and Left
                "left": True,
                "right": False,
                "accelerate": True,
                "brake": False,
            },
            {  # 4 Up and Right
                "left": False,
                "right": True,
                "accelerate": True,
                "brake": False,
            }
        ]