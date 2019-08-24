#!/usr/bin/python3.6
import scipy.sparse
import numpy as np
import sys
import os
import collections


dir_name = sys.argv[1]
top = int(sys.argv[2])
files = os.listdir(dir_name)

name_var_dic = dict()


for file in files:
    try:
        sparsed = scipy.sparse.load_npz(dir_name+file)
    except:
        print("-E- Could not open file named: "+file)
        continue
    densed = sparsed.todense()
#    print("-I- "+file+" contains data of shape: "+str(densed.shape), end = '')
#    print("")
    photo_dim = (np.sqrt(densed.shape[1]))
    states_num = densed.shape[0]

    var_time = np.var(densed, axis=0)
    tot_var_time = np.sum(var_time)

    name_var_dic[file] = tot_var_time
    """
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
    """

for i in range(top):
    print( max(name_var_dic, key=name_var_dic.get) )
    del name_var_dic[max(name_var_dic, key=name_var_dic.get)]



"""
print("==========================================================")
print(name_var_dic)
print("==========================================================")
print( max(name_var_dic, key=name_var_dic.get) )
del name_var_dic[max(name_var_dic, key=name_var_dic.get)]
print( max(name_var_dic, key=name_var_dic.get) )
print("==========================================================")
"""


