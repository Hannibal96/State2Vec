#!/usr/bin/python3.6
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.manifold import TSNE
import os
import scipy.sparse
from scipy.sparse import csr_matrix


def GetData(dir_name):
    #try:
    sparsed1 = scipy.sparse.load_npz("/home/data/starcraft/replay_data/"+dir_name+"/screen_unit_hit_point.npz")
    densed1 = sparsed1.todense()
    record_length = densed1.shape[0]
    sparsed2 = scipy.sparse.load_npz("/home/data/starcraft/replay_data/"+dir_name+"/screen_unit_type.npz")
    densed2 = sparsed2.todense()
    sparsed3 = scipy.sparse.load_npz("/home/data/starcraft/replay_data/"+dir_name+"/screen_unit_hit_point_ratio.npz")
    densed3 = sparsed3.todense()
    sparsed4 = scipy.sparse.load_npz("/home/data/starcraft/replay_data/"+dir_name+"/screen_height_map.npz")
    densed4 = sparsed4.todense()
    sparsed5 = scipy.sparse.load_npz("/home/data/starcraft/replay_data/"+dir_name+"/screen_unit_density_ratio.npz")
    densed5 = sparsed5.todense()
    if not (densed1.shape[0] == record_length or densed2.shape[0] == record_length or densed3.shape[0] == record_length or densed4.shape[0] == record_length or densed5.shape[0] == record_length):
        print("-E- Wrong states num")
        exit()
    channeled_state = np.array([densed1, densed2, densed3, densed4])
    channeled_state = np.swapaxes(channeled_state, 0, 1)
    channeled_state = channeled_state[:, :, :, np.newaxis]
    channeled_state = np.reshape(channeled_state, [record_length,4,84,84])
    #print("-I- channeled state shape: "+str(channeled_state.shape))
    """
    for idx, state in enumerate(channeled_state):
        plt.imshow(state[1])
        plt.show()
        print(state.shape)
    print("-I- channeled state shape: "+str(channeled_state.shape))
    """
    return record_length, channeled_state
    #except:
    #    return 0,[0]

directories = os.listdir("/home/data/starcraft/replay_data/")

state_length = 0
while state_length == 0 or state_length > 500:
    rand_dir_idx = np.random.randint(26592)
    state_length, data_test = GetData(directories[(rand_dir_idx)% 26592])
    print("-I- dir = "+ str(directories[(rand_dir_idx)% 26592]))
    print("-I- #states = "+str(state_length))

NAME = sys.argv[1]
#TEST_LENGTH = int(sys.argv[2])

model_conv_1 = torch.load("./"+NAME+"_conv_1")
model_conv_2 = torch.load("./"+NAME+"_conv_2")
model_conv_3 = torch.load("./"+NAME+"_conv_3")
model_lin_1 = torch.load("./"+NAME+"_lin_1")
model_embd = torch.load("./"+NAME+"_embed")


for idx,s in enumerate(data_test):
    break
    #print(s)
    plt.figure(idx)
    plt.imshow(s)
    plt.show(block=False)


model_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

for state in (data_test):

    curr_state = torch.tensor([state]).float().cuda()

    embed = model_conv_1(curr_state) # size-8 * size-8
    embed = model_pool(embed)     # size/2-4 * size/2-4
    embed = F.relu(model_conv_2(embed))
    embed = model_pool(embed)     #size/4-2 * size/4-2
    embed = F.relu(model_conv_3(embed))
    embed = model_pool(embed)
    embed = embed.reshape(1, -1)
    embed = model_lin_1(embed)
    embed = model_embd(embed)

    try:
        X = torch.cat((X,embed), dim=0)
    except:
        X = embed

X = X.detach().cpu().numpy()

PERPLEXITY = [30] # [1,5,10,30,100]
for perp in PERPLEXITY:
    print(X.shape)
    X_embed = TSNE(n_components=2, perplexity = perp, n_iter = 5000).fit_transform(X)
    print(X_embed.shape)

    fig, ax = plt.subplots()

    COLORS = ['black','blue','green','yellow','purple']
    TEXT = ['blue','black','black','black','black']
    for state in range(state_length):
        embd = X[state]
        ax.plot(float(embd[0]), float(embd[1]), linestyle='', ms=8, color = 'yellow',marker='o', label='Test')
        ax.annotate(state, xy = (float(embd[0]), float(embd[1])), color = 'black' )

    plt.grid()
    plt.savefig(NAME+str("_EmbdPlt2D_Perplexity=")+str(perp))
    plt.show()





