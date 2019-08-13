import numpy as np
import os
import sys


dir_path = "/home/data/starcraft/replay_data/004bdeccc45c67d8f903c9166d1a176ef8c07beb9721b3582edffb5e420b831a/"

files = os.listdir(dir_path)

#print(files)

for file in files:
    full_path = dir_path+file
    
    print("==================================================")
    print(full_path)
    print("")

    try:
        data = np.load(full_path)
    except:
        print("-E- failed: "+full_path)
        continue
    for x in data:
        print(x)
        try:
            print(len(data[x]))
        except:
            pass
        print(data[x])
        print("===============")
        print("")


