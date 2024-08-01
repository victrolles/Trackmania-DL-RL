from dataclasses import dataclass
import numpy as np

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Reward_limits:
    min: int
    max: int

@dataclass
class Point2D:
    x: float
    y: float

    def to_array(self):
        return np.array([self.x, self.y])
    
    def distance(self, point: 'Point2D'):
        # Euclidean distance
        norm = np.linalg.norm(self.to_array() - point.to_array())
        # normalize
        if norm > 100:
            return 1.0
        else:
            return norm / 100.0

@dataclass
class DetectedPoint:
    pos: Point2D
    dist: float

@dataclass
class RadarState:
    car_pos: Point2D
    car_ahead_pos: Point2D
    detected_points: list[DetectedPoint]
    distances: list[float]

@dataclass
class TMSimulationResult:
    reward: int
    done: bool
    score: float

@dataclass
class TrainingStats:
    epoch: int
    step: int
    loss: float
    training_time: float
    epsilon: float

@dataclass
class Exp:
    state: RadarState
    action: int
    reward: float
    next_state: RadarState
    done: bool

@dataclass
class DataBus:
    radar_state: RadarState
    training_stats: TrainingStats
    total_time: float
    distance_travelled: float
    fps_env: float
    exp_buffer_size: int