from stable_baselines3.common.vec_env import VecMonitor
import time


class CustomVecMonitor(VecMonitor):
    EXT = "data.csv"

    def __init__(self, venv, num_env=1, filename=None, info_keywords=()):
        self.num_env = num_env
        VecMonitor.__init__(self, venv=venv, filename=filename+'_episodes', info_keywords=info_keywords)
        from stable_baselines3.common.monitor import ResultsWriter
        self.rewards_writer = ResultsWriter(filename+'_rewards', header=None,
                                            extra_keys=('steps', 'rewards', 'done_env_idx'))
        self.steps = 0

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        self.done_env_idx = []
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
                self.done_env_idx.append(i)
        self.rewards_writer.write_row({'steps': self.steps, 'rewards': list(rewards), 'done_env_idx': self.done_env_idx})
        self.steps += 1
        return obs, rewards, dones, new_infos
