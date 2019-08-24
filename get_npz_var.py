#!/usr/bin/python3.6
import scipy.sparse
import numpy as np
import sys
import os


dir_name = sys.argv[1]
files = os.listdir(dir_name)


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

    var_time = np.var(densed, axis=0)
    tot_var_time = np.sum(var_time)

    for s in range(states_num):
        if s == 0:
            print(", photo sizes: "+str(photo_dim)+"x"+str(photo_dim))
        try:
            image = np.reshape(densed[s], [int(photo_dim), int(photo_dim)])
        except:
            print("-E- Could not calculate image variance from file named: "+file+" and state number: "+str(s))
            break

        var_time = np.var(densed, axis=0)
        print("-I- File name: "+file+", State: "+str(s)+" , photo variance is: "+str(np.var(image))+", time variance is: "+str(tot_var_time))




