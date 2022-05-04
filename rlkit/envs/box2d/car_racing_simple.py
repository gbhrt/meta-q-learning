"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
from curses.ascii import FS
import sys
import math
import numpy as np

# import Box2D
# from Box2D.b2 import fixtureDef
# from Box2D.b2 import polygonShape
# from Box2D.b2 import contactListener

import csv
import datetime
import gym
from gym import spaces
# from rlkit.envs.box2d.car_dynamics import Car #gym.envs.  box2d.
# from car_dynamics import Car 
from gym.utils import seeding, EzPickle

# import pyglet

# pyglet.options["debug_gl"] = False
# from pyglet import gl

import matplotlib.pyplot as plt
import json



class CarRacingSimple1(gym.Env, EzPickle):
    def __init__(self, verbose=1):
        print('init don - carRacingSimple')
        self.seed()
        self.vel_vec = []
        self.acc_vec = []
        self.pos_vec = []
        self.reward = 0.0
        self.prev_reward = 0.0
        self.last_action = [0.0]
        self.dt = 0.04 #0.2
        self.max_acceleration = 42.0
        self.max_velocity= 30.0 #50#
        self.jerk_constant= 10.0
        self.render_flag = False
        self.add_acc_factor_to_state = False


        plt.ion()
        self.vel_filename = 'trajectories/vel'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        self.acc_filename = 'trajectories/acc'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        self.pos_filename = 'trajectories/pos'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        self.targets_file_name = 'trajectories/targets'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        # self.target = None
        self.min_dis_to_target = 2.0

        self.action_space = spaces.Box(
            np.array([-1]).astype(np.float32),
            np.array([+1]).astype(np.float32),
        )  #  gas and brake in one command

     
        #state: linear velocity [0,30], distance to target [0,100]
        # self.observation_space = spaces.Box(
        #     np.array([0.0,0.0]).astype(np.float32),
        #     np.array([+30.0,100.0]).astype(np.float32),
        # )

        # self.observation_space = spaces.Box(
        #     np.array([0.0]).astype(np.float32),
        #     np.array([100.0]).astype(np.float32),
        # )
        # self.observation_space = spaces.Box(
        #     np.array([-30.0,0.0,0.0,0.0]).astype(np.float32),
        #     np.array([+30.0,100.0,40.0,1.0]).astype(np.float32),
        # )
        if self.add_acc_factor_to_state:
        #     self.observation_space = spaces.Box(
        #     np.array([-1.0,0.0,0.0,0.0]).astype(np.float32),
        #     np.array([+1.0,0.0,1.0,1.0]).astype(np.float32),
        # )
            self.observation_space = spaces.Box(
                np.array([-1.0,0.0,0.0]).astype(np.float32),
                np.array([+1.0,1.0,1.0]).astype(np.float32),
            )
        else:
            # self.observation_space = spaces.Box(
            #     np.array([-1.0,0.0,0.0]).astype(np.float32),
            #     np.array([+1.0,1.0,1.0]).astype(np.float32),
            # )
            self.observation_space = spaces.Box(
                np.array([-1.0,0.0]).astype(np.float32),
                np.array([+1.0,1.0]).astype(np.float32),
            )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def _reset(self, acc_factor = 1.0, target_distance_vec = [20.0], target_velocity_vec = [5.0], max_acc = 1.0): #

        #plot data from last episode:
        if self.render_flag:
            #plot graph using matplotlib:
            # plt.figure("velocity")
            # plt.xlim(0, 200)
            # plt.ylim(0, 20)
            # plt.cla()
            # plt.plot(self.vel_vec)

            # plt.figure("acc")
            # plt.xlim(0, 200)
            # plt.ylim(0, 20)
            # plt.cla()
            # plt.plot(self.acc_vec)

            # plt.draw()
            # plt.pause(0.0001)

            

            if len(self.acc_vec) > 1:
                with open(self.acc_filename, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.acc_factor] + [self.max_acc] + self.acc_vec)

            if len(self.vel_vec) > 1:
                with open(self.vel_filename, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.vel_vec)

            if len(self.pos_vec) > 1:
                with open(self.pos_filename, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.pos_vec)

            if len(self.pos_vec) > 1:
                with open(self.targets_file_name, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.target_distance_vec.tolist())
                    writer.writerow(self.target_velocity_vec.tolist())
                    writer.writerow(self.indexes)

            self.render_flag = False
            
            # if len(self.pos_vec) > 1:
            #     data ={'distances':self.target_distance_vec.tolist(),'velocities:':self.target_velocity_vec.tolist(),'indexes':self.indexes}
            #     with open(self.targets_file_name, 'a') as outfile:
            #         json.dump(data, outfile)

        self.acc_factor = acc_factor

        self.target_distance_vec = target_distance_vec
        self.target_velocity_vec = target_velocity_vec
        self.max_acc = max_acc
        self.indexes = []

        self.reward = 0.0
        self.prev_reward = 0.0
        self.t = 0.0


        
        self.vel_vec = []
        self.acc_vec = []
        self.pos_vec = []
        self.last_action = [0.0]

        self.x = 0
        self.vel = 0
        self.acc = 0
        self.target_index = 0
        self.target_pos = self.target_distance_vec[self.target_index]
        self.target_vel = self.target_velocity_vec[self.target_index]
        #np.array(self.target[2])

        # self.target = [0.0,self.target_distance_vec[self.target_index],self.target_velocity_vec[self.target_index]] #x,y,vel
        # self.target = [0.0,20.0,self.target_velocity_vec[0]] #x,y,vel
        # self.car = Car(self.world, *self.track[0][1:4])

        return self._step(None)[0]

    def _step(self, action):
        if action is not None:
            self.acc = self.acc_factor * action[0] * self.max_acceleration if action[0] < 0.0 else action[0] * self.max_acceleration
            # self.acc = self.acc_factor * action[0] * self.max_acceleration


            # jerk = self.acc_factor * action[0] * self.jerk_constant if action[0] < 0.0 else action[0] * self.jerk_constant
            # self.acc +=  jerk *self.dt ## * 
            # self.acc = np.clip(self.acc,-self.max_acceleration ,self.max_acceleration )

            # self.acc = action[0] * self.acceleration
            self.vel += self.acc*self.dt
            if self.vel < 0.0 or self.vel > self.max_velocity:
                self.acc = 0.0
            self.vel = np.clip(self.vel,0.0,self.max_velocity)
            self.x += self.vel*self.dt

            self.acc_vec.append(action[0])

        # target_pos = np.array(self.target[:2])
        
        # distance_to_target = np.linalg.norm(self.target_pos - self.x)
        distance_to_target = self.target_pos - self.x 
        # print('distance_to_target',distance_to_target,  'self.x:',self.x      )
        self.vel_vec.append(self.vel)
        self.pos_vec.append(self.x)
        
        if self.add_acc_factor_to_state:
            self.state =  np.concatenate([
                [
                self.vel/self.max_velocity,
                # self.acc/self.max_acceleration,
                distance_to_target/20,
                #
                # self.target_vel/30,
                self.acc_factor
                ]
            ]).astype(np.float32).flatten()
        else:
            self.state =  np.concatenate([
                [
                self.vel/self.max_velocity,
                # self.acc/self.max_acceleration,
                distance_to_target/20,
                #
                # self.target_vel/30,
                # self.acc_factor
                ]
            ]).astype(np.float32).flatten()       
        #np.array([vel,distance_to_target])
        

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            # step_reward = -0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0
            # step_reward = self.reward - self.prev_reward
            # self.prev_reward = self.reward

            # step_reward = self.vel/1000 - 0.01*(action[0] - self.last_action[0])**2# is working
            # normalized_acc = abs(self.acc_factor * action[0] if action[0] < 0.0 else action[0])
            # acceleration_cost =  0.0 if normalized_acc < self.max_acc  else (self.max_acc - normalized_acc)**2

            # action_cost = 0.01*(action[0] - self.last_action[0])**2 + 0.001*action[0]
            action_cost = 0.0
            # step_reward = np.sign(self.vel)* self.vel**2/1000 - action_cost #-0.2 * acceleration_cost# is working
            # step_reward = self.acc_factor * self.vel/30 # is working
            # step_reward = -distance_to_target/1000 - 0.01*(action[0] - self.last_action[0])**2# is working
            # print("distance_to_target:",distance_to_target, "min_dis_to_target:",self.min_dis_to_target )
            # step_reward = -0.01
            step_reward = -0.002
            if distance_to_target < self.min_dis_to_target:
                # step_reward = 0.0 if self.vel < self.target_vel else - 0.00001*((self.vel -self.target_vel))**2  
                # step_reward = 0.0 if self.vel < self.target_vel else - 0.000001*((self.vel -self.target_vel))**2  
                # step_reward = 0.0 if self.vel < self.target_vel else - 0.01*((self.vel -self.target_vel))**2  
                # step_reward = 0.0 if self.vel < self.target_vel else - 0.05*((self.vel -self.target_vel))**2  
                step_reward = 0.0 if self.vel < self.target_vel else - 0.01*((self.vel -self.target_vel))**2  
                step_reward -= action_cost
                self.indexes.append(len(self.vel_vec)-1)
                # print("reward:",step_reward, "vel:",self.vel,"target_vel:",self.target_vel )
                self.target_index += 1
                if self.target_index < len(self.target_distance_vec) - 1:
                    self.target_pos += self.target_distance_vec[self.target_index]#np.array(self.target[1])
                    self.target_vel = self.target_velocity_vec[self.target_index]#np.array(self.target[2])
                else:
                    done = True

            #     step_reward = 0.0 if self.vel < target_vel else - ((self.vel -target_vel)*0.5)**2# 
                # print("reward:",step_reward, "vel:",self.vel,"target_vel:",target_vel )

            self.last_action = action[:]
        return self.state, step_reward, done, {}

    def render(self, mode="human"):
        self.render_flag = True
    #    #plot graph using matplotlib:
    #     plt.figure("velocity")
    #     plt.xlim(0, 200)
    #     plt.ylim(0, 20)
    #     plt.cla()
    #     plt.plot(self.vel_vec)

    #     plt.figure("acc")
    #     plt.xlim(0, 200)
    #     plt.ylim(0, 20)
    #     plt.cla()
    #     plt.plot(self.acc_vec)

    #     plt.draw()
    #     plt.pause(0.0001)

if __name__ == "__main__":

    a = np.array([0.0])


    acc_command = 1.0
   
    env = CarRacingSimple1()
    env.render()

    while isopen:
        env._reset(acc_factor = 0.5)
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            v = s[0]
            d = s[1]
            a = []
            s, r, done, info = env._step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                # print(s.copy())
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
