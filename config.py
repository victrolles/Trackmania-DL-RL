from dataclasses import dataclass
from librairies.data_classes import EnvironmentConfig, EpsilonConfig, RLConfig, CooldownConfig, SpawnConfig, AgentConfig, ExpBufferConfig

@dataclass
class Config:

    # Environment
    traininig: bool = True
    load_model: bool = True

    # RL Trainer
    rl_config: RLConfig = RLConfig(
        rl_algo='DQN',
        lr=1e-4,
        gamma=0.995,
        hidden_layer_size=128,
        output_size=5,
        batch_size=16,
        sync_target_rate=25,
        sync_save_rate=400,
        epsilon=EpsilonConfig(
            start=1.0,
            end=0.01,
            decay=0.0005
        )
    )
    
    # Agent
    agent_config: AgentConfig = AgentConfig(
        max_dist_radar=100.0
    )

    # Environment
    environment: EnvironmentConfig = EnvironmentConfig(
        name='snake_map_training',
        length=1103.422,
        game_speed=1.0)
    
    # Spawn
    spawn_config: SpawnConfig = SpawnConfig(   
        random=False,
        number=61
    )
    
    # Cooldown
    cooldown_config: CooldownConfig = CooldownConfig(
        action=8,
        display_state=14,
        display_stats=1
    )

    # Experience Buffer
    exp_buffer_config: ExpBufferConfig = ExpBufferConfig(
        buffer_size=8_192
    )
