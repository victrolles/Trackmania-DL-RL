from dataclasses import dataclass
import numpy as np
from rl_algorithms.dqn.dqn_model import DQNModel
import torch.optim as optim

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

# Configuration class

@dataclass
class EnvironmentConfig:
    name: str
    length: float
    game_speed: float = 1.0

@dataclass
class EpsilonConfig:
    start: float
    end: float
    decay: float

@dataclass
class RLConfig:
    lr: float
    gamma: float
    hidden_layer_size: int
    output_size: int
    batch_size: int
    sync_target_rate: int
    sync_save_rate: int
    epsilon: EpsilonConfig

@dataclass
class CooldownConfig:
    action: int
    display_state: int
    display_stats: int

@dataclass
class SpawnConfig:
    random: bool
    number: int

@dataclass
class AgentConfig:
    max_dist_radar: float

@dataclass
class ExpBufferConfig:
    buffer_size: int

# -----------------------------------------   

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
    dist_to_finish_line: float

@dataclass
class TrainingCheckpoint:
    model_network: DQNModel
    model_target_network: DQNModel
    optimizer: optim.Adam
    current_epsilon: float
    epoch: int

@dataclass
class TrainingStats:
    epoch: int
    step: int
    loss: float
    epsilon: float

@dataclass
class Exp:
    state: RadarState
    action: int
    reward: float
    done: bool
    next_state: RadarState

    def set_none():
        return Exp(None, None, None, None, None)

@dataclass
class DataBus:
    radar_state: RadarState
    training_stats: TrainingStats
    total_time: float
    distance_travelled: float
    fps_env: float
    exp_buffer_size: int