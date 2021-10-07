import numpy as np

from rlkit.envs.box2d.car_racing import CarRacingSimple
from . import register_env


@register_env('car_racing-acc')
class CarRacing(CarRacingSimple):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        # directions = [-1, 1]
        # self.tasks = [{'direction': direction} for direction in directions]
        
        # self._goal_dir = task.get('direction', 1)
        # self._goal = self._goal_dir
        super(CarRacing, self).__init__()

    def step(self, action):
        # xposbefore = self.sim.data.qpos[0]
        # self.do_simulation(action, self.frame_skip)
        # xposafter = self.sim.data.qpos[0]

        # forward_vel = (xposafter - xposbefore) / self.dt
        # forward_reward = self._goal_dir * forward_vel
        # ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        # observation = self._get_obs()
        # reward = forward_reward - ctrl_cost
        # done = False
        # infos = dict(reward_forward=forward_reward,
        #     reward_ctrl=-ctrl_cost, task=self._task)
        action = np.clip(action,-1.0,1.0)
        return self._step(action)
        # return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        frictions = np.random.uniform(0.1, 1.0, size=(num_tasks,))
        tasks = [{'friction': frictions} for frictions in frictions]
        return tasks


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_dir = self._task['friction']
        self._goal = self._goal_dir
        self.reset()
    
    def reset(self):
        return self._reset()

