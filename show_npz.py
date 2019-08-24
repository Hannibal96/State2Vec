#!/usr/bin/python3.6
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


dir_name = sys.argv[1]
files = os.listdir(dir_name)

plt.ion()

for file in files:
    try:
        sparsed = scipy.sparse.load_npz(dir_name+file)
    except:
        print("-E- Could not open file named: "+file)
        continue
    densed = sparsed.todense()
    print("-I- "+file+" contains data of shape: "+str(densed.shape), end = '')
    photo_dim = (np.sqrt(densed.shape[1]))
    states_num = densed.shape[0]

    for s in range(states_num):
        if s == 0:
            print(", photo sizes: "+str(photo_dim)+"x"+str(photo_dim))
        try:
            image = np.reshape(densed[s], [int(photo_dim), int(photo_dim)])
            plt.imshow(image)
            plt.title(file+" - "+str(s))
            plt.show()
            plt.pause(0.001)
            #input("Press [enter] to continue.")
    
        except:
            print("\n-E- Could not show image from file named: "+file)
            break


