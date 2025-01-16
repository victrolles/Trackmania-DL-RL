from librairies.data_classes import EnvironmentConfig, EpsilonConfig, RLConfig, CooldownConfig, SpawnConfig, AgentConfig, ExpBufferConfig

class Config:

    # RL Trainer
    rl_config: RLConfig = RLConfig(
        rl_algo='DQN',
        traininig=True,
        load_checkpoint=False,
        load_checkpoint_path='extras/maps/snake_map_training/saves/15-01-25-18-56_DQN_RadarAgent/model_977.pth',
        lr=1e-3, #1e-4
        gamma=0.995,
        hidden_layer_size=128,
        output_size=5,
        batch_size=16,
        sync_target_rate=25,
        sync_save_rate=400,
        epsilon=EpsilonConfig(
            start=1.0,
            end=0.01,
            decay=0.001
        )
    )
    
    # Agent
    agent_config: AgentConfig = AgentConfig(
        name='Radar',
        max_dist_radar=100.0
    )

    # Environment
    environment: EnvironmentConfig = EnvironmentConfig(
        name='snake_map_training',
        length=1257.32,
        game_speed=1.0)
    
    # Spawn
    spawn_config: SpawnConfig = SpawnConfig(   
        random=True,
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
