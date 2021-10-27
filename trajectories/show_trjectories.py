import csv
import matplotlib.pyplot as plt
import numpy as np
import json

# date = '20211013-133636'# 0.5 - 1.0 good
# date = '20211013-141422'# 0.1 - 1.0 not good
# date = '20211013-152252'# 0.1 - 1.0 not good
# date = '20211014-144437'# -1.0 - 1.0 good, changing reward
# date = '20211014-145835'# -1.0 - 1.0 good, changing acc
# date = '20211014-155708'# -1.0 - 1.0 target 5 - good!!
# date = '20211014-191522'# -1.0 - -0.5 and 0.5 - 1.0 target 5 - good!!
# date = '20211017-092253'# ok, change target_distance
# date = '20211017-095225'# ok, change target_distance, higher cost on action change
# date = '20211020-141403'# multiple targets - ok
date = '20211022-170040' # multiple targets - ok




pos_filename = 'trajectories/pos'+date+'.csv'
vel_filename = 'trajectories/vel'+date+'.csv'
acc_filename = 'trajectories/acc'+date+'.csv'
targets_filename = 'trajectories/targets'+date+'.csv'

pos_trajectories = []
with open(pos_filename, newline='') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        pos_trajectories.append(row)
vel_trajectories = []
with open(vel_filename, newline='') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        vel_trajectories.append(row)
acc_trajectories = []
with open(acc_filename, newline='') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        acc_trajectories.append(row)

targets_vec = []
with open(targets_filename, newline='') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        targets_vec.append(row)

target_velocities = []
target_distances = []
target_indexes = []


for i in range(0,len(targets_vec),3):
    target_distances.append(targets_vec[i])
    target_velocities.append(targets_vec[i+1])
    target_indexes.append(targets_vec[i+2])

# with open(targets_filename) as json_file:
#     data = json.load(json_file)

# target_distance_vec = data['distances']
# target_velocity_vec = data['velocities']
# indexes = data['indexes']

plt.ion()

end = int(len(vel_trajectories)/2) -1

indexes= [end - i for i in range(0,20)] #[end,end -2]#

fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
# twin2 = ax.twinx()

# twin2.spines.right.set_position(("axes", 1.2))

# p1, = ax.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
# p2, = twin1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
# p3, = twin2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")

for i in indexes:
    # plt.figure("velocity")
    # plt.plot(pos_trajectories[i],vel_trajectories[i]) #

    # plt.figure("acc")
    # plt.plot(pos_trajectories[i][:-1],acc_trajectories[i]) 
    acc_factor = acc_trajectories[i].pop(0)
    max_acc = acc_trajectories[i].pop(0)
    # target_to_dist = acc_trajectories[i].pop(0)
    # taget_velocity = acc_trajectories[i].pop(0)
    # print("vel:",vel_trajectories[i][-1],"taget_velocity:",taget_velocity)
    # if acc_factor == 1:
    #     continue
    # print('acc_factor:',acc_factor)
    plt.cla()
    ax.cla()
    twin1.cla()
    dt = 0.02
    t = [dt*t for t in range(len(vel_trajectories[i]))]
    # p1, = ax.plot(pos_trajectories[i], "b-", label="Position")
    p2, = ax.plot(t,vel_trajectories[i], label="Velocity")
    p3, = twin1.plot(t[0:-1],acc_trajectories[i], p2.get_color(),linestyle='dashed', label="Acceleration")
    twin1.plot(t,[max_acc for _ in t],p2.get_color())
    twin1.plot(t,[-max_acc for _ in t],p2.get_color())
    target_indexes_time = np.asarray(target_indexes[i])*dt
    tar_vel = target_velocities[i]
    tar_vel = tar_vel[:target_indexes_time.size]

    ax.bar(target_indexes_time,tar_vel,0.05, color = p2.get_color())
  
    # p1, = ax.plot(pos_trajectories[i], label="Position")
    # p2, = twin1.plot(vel_trajectories[i], label="Velocity")
    # acc_vec = [acc*acc_factor if acc < 0.0 else acc for acc in acc_trajectories[i]]
    acc_vec = acc_trajectories[i]
    # p3, = twin2.plot(acc_vec, label="Acceleration")

    plt.draw()
    plt.pause(3.0)

ax.set_xlabel("Time")
# ax.set_ylabel("Position")
ax.set_ylabel("Velocity")
# twin1.set_ylabel("Velocity")
twin1.set_ylabel("Acceleration")

# ax.yaxis.label.set_color(p1.get_color())
# ax.yaxis.label.set_color(p2.get_color())
# twin1.yaxis.label.set_color(p2.get_color())
# twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', **tkw)
twin1.tick_params(axis='y', **tkw)
# twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

# ax.legend(handles=[p1, p2, p3])

plt.show()