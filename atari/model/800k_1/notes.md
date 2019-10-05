exploration_max = 1.0
exploration_min = 0.1
exploration_steps = 80000 # 800000
exploration_decay = (exploration_max-exploration_min)/exploration_steps

params = {"gamma":0.99, "memory_size": 900000,        "batch_size": 32,
            "training_frequency": 4, "target_network_update_frequency": 40000,
            "model_persistence_update_frequency": 10000,
            "replay_start_size": 5000 ,"exploration_test": 0.02,
            "exploration_max": exploration_max, 
            "exploration_min": exploration_min,
            "exploration_steps": exploration_steps, 
            "exploration_decay": exploration_decay}

model_save_freq = 10000 # 50000
total_step_limit = 1000000 # 10000000
total_run_limit = 1000 # 5000
render = False #True
clip = True
