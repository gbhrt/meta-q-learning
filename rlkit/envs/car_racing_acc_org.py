import numpy as np

# from rlkit.envs.box2d.car_racing import CarRacingSimple
from rlkit.envs.box2d.car_racing_simple import CarRacingSimple1 as CarRacingSimple
from . import register_env


@register_env('car_racing-acc-org')
class CarRacing(CarRacingSimple):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._acc_factor = self.tasks[0].get('acc_factor', 1.0)    
        self._max_acc = self.tasks[0].get('max_acc', 1.0)    
        self._target_distance_vec =  self.tasks[0].get('target_distance_vec', [20.0]) 
        self._target_velocity_vec = self.tasks[0].get('target_velocity_vec', [5.0]) 
        print('sampled a new acceleration factor: ', self._acc_factor)

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
        # action[0] *= self._acc_factor
        action = np.clip(action,-1.0,1.0)
        return self._step(action)
        # return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        acc_factors_values = np.random.uniform(0.5, 1.0, size=(num_tasks,))
        acc_factors = acc_factors_values
        # acc_factors = 2 * np.random.randint(2, size=num_tasks) - 1
        acc_factors = acc_factors*acc_factors_values
        tasks = [{'acc_factor': acc_factor} for acc_factor in acc_factors]

        max_accs = np.random.uniform(0.3, 1.0, size=(num_tasks,))
        for task,max_acc in zip(tasks,max_accs):
            task['max_acc'] = max_acc

        target_distance_vec_vec = np.random.uniform(10.0, 40.0, size=(num_tasks,10))
        for task,target_distance_vec in zip(tasks,target_distance_vec_vec):
            task['target_distance_vec'] = target_distance_vec

        target_velocity_vec_vec = np.random.uniform(5.0, 20, size=(num_tasks,10))
        for task,target_velocity_vec in zip(tasks,target_velocity_vec_vec):
            task['target_velocity_vec'] = target_velocity_vec
        return tasks


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._acc_factor = self._task['acc_factor']
        self._max_acc = self._task['max_acc']
        self._target_distance_vec = self._task['target_distance_vec']
        self._target_velocity_vec = self._task['target_velocity_vec']
        print('sampled a new acceleration factor: ', self._acc_factor, 'and a new target_distance_vec',self._target_distance_vec,'and a new target_velocity_vec',self._target_velocity_vec)

        # self._goal = self._goal_dir
        self.reset()
    
    def reset(self):
        return self._reset(self._acc_factor,self._target_distance_vec,self._target_velocity_vec,self._max_acc)

