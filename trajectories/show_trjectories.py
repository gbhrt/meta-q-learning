import csv
import matplotlib.pyplot as plt
import numpy as np


# date = '20211013-133636'# 0.5 - 1.0 good
# date = '20211013-141422'# 0.1 - 1.0 not good
# date = '20211013-152252'# 0.1 - 1.0 not good
# date = '20211014-144437'# -1.0 - 1.0 good, changing reward
# date = '20211014-145835'# -1.0 - 1.0 good, changing acc
# date = '20211014-155708'# -1.0 - 1.0 target 5 - good!!
# date = '20211014-191522'# -1.0 - -0.5 and 0.5 - 1.0 target 5 - good!!
# date = '20211017-092253'# ok, change target_distance
# date = '20211017-095225'# ok, change target_distance, higher cost on action change
date = '20211018-124201'# 



pos_filename = 'trajectories/pos'+date+'.csv'
vel_filename = 'trajectories/vel'+date+'.csv'
acc_filename = 'trajectories/acc'+date+'.csv'

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



end = len(vel_trajectories) -1

indexes= [end - i for i in range(10)] #[end,end -2]#

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
    target_to_dist = acc_trajectories[i].pop(0)
    taget_velocity = acc_trajectories[i].pop(0)
    # if acc_factor == 1:
    #     continue
    print('acc_factor:',acc_factor)
    dt = 0.02
    t = [dt*t for t in range(len(vel_trajectories[i]))]
    # p1, = ax.plot(pos_trajectories[i], "b-", label="Position")
    p2, = ax.plot(t,vel_trajectories[i], label="Velocity")
    p3, = twin1.plot(t[0:-1],acc_trajectories[i], p2.get_color(),linestyle='dashed', label="Acceleration")

    p4, = ax.bar([(len(vel_trajectories[i]) - 1)*dt],[taget_velocity],0.01, color = p2.get_color())
    # p1, = ax.plot(pos_trajectories[i], label="Position")
    # p2, = twin1.plot(vel_trajectories[i], label="Velocity")
    # acc_vec = [acc*acc_factor if acc < 0.0 else acc for acc in acc_trajectories[i]]
    acc_vec = acc_trajectories[i]
    # p3, = twin2.plot(acc_vec, label="Acceleration")

    

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