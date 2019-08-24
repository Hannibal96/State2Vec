#!/usr/bin/python3.6
import scipy.sparse
import numpy as np
import sys
import os

dir_name = sys.argv[1]
files = (    'screen_unit_hit_point.npz',
             'screen_unit_type.npz',
             'screen_unit_hit_point_ratio.npz',
             'screen_height_map.npz',
             'screen_unit_density_ratio.npz'        )

channeled_state = np.array([]) 


for idx,file in enumerate(files):
    try:
        sparsed = scipy.sparse.load_npz(dir_name+file)
    except:
        print("-E- Could not open file named: "+file)
        continue
    densed = sparsed.todense()
    print("-I- "+file+" contains data of shape: "+str(densed.shape), end = '')
    print("")
    photo_dim = (np.sqrt(densed.shape[1]))
    states_num = densed.shape[0]

    for s in range(states_num):
        if s == 0:
            print(", photo sizes: "+str(photo_dim)+"x"+str(photo_dim))
    #    try:
        image = np.reshape(densed[s], [int(photo_dim), int(photo_dim)])
        channeled_state = np.array([][])  [idx][s] = 5
    #    except:
    #        print("-E- Could not calculate image variance from file named: "+file+" and state number: "+str(s))
    #        break

   


