from dataclasses import dataclass
import numpy as np

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Point2D:
    x: float
    y: float

    def to_array(self):
        return np.array([self.x, self.y])
    
@dataclass
class DataBus:
    car_pos: Point2D
    car_ahead_pos: Point2D
    car_sensors: list[Point2D]


@dataclass
class Experience:
    state: None
    action: int
    reward: float
    next_state: DataBus
    done: bool
