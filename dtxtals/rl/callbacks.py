from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.episode_count = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

        if self.n_calls % self.locals['log_interval'] == 0:
            rewards = self.locals['rewards']
            rewards_sum = 0

            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar('reward/env #{}'.format(i), rewards[i], self.n_calls)
                rewards_sum += rewards[i]
            self.tb_formatter.writer.add_scalar('reward/average', rewards_sum/self.locals['env'].num_envs, self.n_calls)

        for i in range(self.locals['env'].num_envs):
            if 'terminal_observation' in self.locals['infos'][i]:
                self.episode_count += 1
                episode_reward = self.locals['infos'][i]['episode']['r']
                episode_length = self.locals['infos'][i]['episode']['l']
                episode_time = self.locals['infos'][i]['episode']['t']

                self.tb_formatter.writer.add_scalar('episode/reward', episode_reward, self.episode_count)
                self.tb_formatter.writer.add_scalar('episode/length', episode_length, self.episode_count)
                self.tb_formatter.writer.add_scalar('episode/time', episode_time, self.episode_count)
        return True
