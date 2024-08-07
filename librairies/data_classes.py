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
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def to_list(self):
        return [self.x, self.y]
    
    def distance(self, point: 'Point2D', normalize: bool = False, max_dist: float = 100.0) -> float:

        # Euclidean distance
        norm = np.linalg.norm(self.to_array() - point.to_array())

        # normalize if needed
        if normalize:
            if norm > max_dist:
                norm = 1.0
            else:
                norm /= max_dist
            
        return norm

        

@dataclass
class DetectedPoint:
    pos: Point2D
    dist: float

@dataclass
class AgentConfig:
    name: str
    input_size: int

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
    dist_to_finish_line: float

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
    done: bool
    next_state: RadarState

@dataclass
class DataBus:
    radar_state: RadarState
    training_stats: TrainingStats
    total_time: float
    distance_travelled: float
    fps_env: float
    exp_buffer_size: int