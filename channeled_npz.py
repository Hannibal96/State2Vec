#!/usr/bin/python3.6
import scipy.sparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
dir_name = sys.argv[1]
output = (sys.argv[2])

files = (    'screen_unit_hit_point.npz',
             'screen_unit_type.npz',
             'screen_unit_hit_point_ratio.npz',
             'screen_height_map.npz',
             'screen_unit_density_ratio.npz'        )

channeled_state = np.array([]) 

for s in range(1):

    sparsed1 = scipy.sparse.load_npz(dir_name+"screen_unit_hit_point.npz")
    densed1 = sparsed1.todense()
    
    sparsed2 = scipy.sparse.load_npz(dir_name+"screen_unit_type.npz")
    densed2 = sparsed2.todense()
    
    sparsed3 = scipy.sparse.load_npz(dir_name+"screen_unit_hit_point_ratio.npz")
    densed3 = sparsed3.todense()
    
    sparsed4 = scipy.sparse.load_npz(dir_name+"screen_height_map.npz")
    densed4 = sparsed4.todense()    
    
    sparsed5 = scipy.sparse.load_npz(dir_name+"screen_unit_density_ratio.npz")
    densed5 = sparsed5.todense()
    

    channeled_state = np.array([densed1, densed2, densed3, densed4, densed5])
    channeled_state = np.swapaxes(channeled_state, 0, 1)

    print("-I- channeled state shape: "+str(channeled_state.shape))

#    plt.imshow( np.reshape(channeled_state[0,0,:], (84,84)) )
#    plt.show() 


    scipy.sparse.save_npz(output , (np.array)(channeled_state) )
    

    
    
    





