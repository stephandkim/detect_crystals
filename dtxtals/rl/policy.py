from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy
import dtxtals.rl.config as config


class CustomPPOPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=config.PPO_NET_ARCH)
