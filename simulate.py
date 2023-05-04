from RLAgent import MultiAgentEnvironment

ma_env = MultiAgentEnvironment()

ma_env.train_agents(num_episodes=100000)

ma_env.save_models("fixer_model", "cutter_model")
