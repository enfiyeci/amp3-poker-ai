"""
Optimized 20-hour training configuration for AMP3
Reduces sample sizes by ~50% while maintaining good performance
"""

CONFIG_20HR = {
    # Save directory
    'save_dir': './checkpoints_20hr',

    # Device
    'device': 'cpu',

    # OSM Configuration (3 hours)
    'osm_num_games': 5000,         # 50% reduction
    'osm_epochs': 50,
    'osm_batch_size': 128,
    'osm_lr': 1e-3,
    'osm_lstm_hidden': 128,
    'osm_lstm_layers': 2,

    # Later-street models (12-15 hours parallel)
    'street_num_samples': 50000,   # 50% reduction
    'street_epochs': 50,
    'street_batch_size': 256,
    'street_lr': 1e-3,
    'street_hidden_dims': [256, 256, 128],

    # AMP3 RL (3-4 hours)
    'amp3_episodes': 20000,        # 80% reduction (still functional)
    'amp3_batch_size': 256,
    'amp3_buffer_size': 50000,
    'amp3_actor_lr': 5e-4,
    'amp3_critic_lr': 5e-4,
    'amp3_gamma': 0.99,
    'amp3_update_interval': 100,

    # Preflop imitation (already done)
    'preflop_num_samples': 100000,
    'preflop_epochs': 100,

    # CFR/NFSP (skipped)
    'cfr_iterations': 0,
    'nfsp_episodes': 0,
}
