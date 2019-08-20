#!/usr/bin/python3.6

import numpy as np
import os
import sys


if len(sys.argv) == 2:
    if sys.argv[1][0] == '/':
        dir_path = sys.argv[1]
    else:
        dir_path = "/home/data/starcraft/replay_data/"+sys.argv[1]+"/"

else:
    dir_path = "/home/data/starcraft/replay_data/004bdeccc45c67d8f903c9166d1a176ef8c07beb9721b3582edffb5e420b831a/"

files = os.listdir(dir_path)

features = list()
features.append("screen_visibility_map.npz")
features.append("other_game_loop.npz")

#print(files)

z = np.array([])
files_counter = 0
idptr_len = -1


for file in files:
    if not file in features:
        pass
    full_path = dir_path+file

    try:
        data = np.load(full_path)
    except:
        #print("-E- failed: "+full_path)
        continue

    for x in data:
        if x == "indptr":
            #print(len(data[x]))
            if idptr_len == -1:
                idptr_len = len(data[x])
            else:
                if not idptr_len == len(data[x]):
                    print("-E- idptr_len not consistent")
                    exit()
               
            z = np.concatenate((z, data[x]), axis=0)
            files_counter += 1


print(z)
print(z.shape)

z = np.resize(z,(files_counter,idptr_len))

print(z)
print(z.shape)



