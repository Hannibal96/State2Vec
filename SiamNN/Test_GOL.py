#!/usr/bin/python3.6
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.manifold import TSNE
from GenData import *

TYPE = sys.argv[1]
NAME = sys.argv[2]
SIZE = int(sys.argv[3])
TEST_LENGTH = int(sys.argv[4])

if TYPE == 'Convolution':
    model_conv_1 = torch.load("./"+NAME+"_conv_1")
    model_conv_2 = torch.load("./"+NAME+"_conv_2")
    model_conv_3 = torch.load("./"+NAME+"_conv_3")
    model_lin_1 = torch.load("./"+NAME+"_lin_1")
    model_lin_2 = torch.load("./"+NAME+"_lin_2")
    model_embd = torch.load("./"+NAME+"_embed")

if TYPE == 'Linear':
    model_lin1 = torch.load("./"+NAME+"_lin_1")
    model_lin2 = torch.load("./"+NAME+"_lin_2")
    model_embd = torch.load("./"+NAME+"_embed")


data_test = game_of_life(SIZE, TEST_LENGTH)
#offset = np.random.randint(4)
#data_test = torch.load("./"+NAME+"_data")[offset*20:offset*20+TEST_LENGTH]

for idx,s in enumerate(data_test):
    break
    #print(s)
    plt.figure(idx)
    plt.imshow(s)
    plt.show(block=False)



for state in (data_test):

    curr_state = torch.tensor([[state]]).float().cuda()

    if TYPE == 'Convolution':
        conv = curr_state
        conv = F.relu(model_conv_1(conv))
        conv = F.relu(model_conv_2(conv))
        conv = model_conv_3(conv)

        conv = conv.reshape(1, -1)
        conv = F.relu(model_lin_1(conv))
        conv = model_lin_2(conv)
        embd = model_embd(conv)

    if TYPE == 'Linear':
        conv = curr_state.reshape(1, -1)
        lin1 = model_lin1(conv)
        lin2 = model_lin2(lin1)
        embd = model_embd(lin2)

    try:
        X = torch.cat((X,embd), dim=0)
    except:
        X = embd

X = X.detach().cpu().numpy()

PERPLEXITY = [30] # [1,5,10,30,100]
for perp in PERPLEXITY:
    print(X.shape)
    X_embed = TSNE(n_components=2, perplexity = perp, n_iter = 5000).fit_transform(X)
    print(X_embed.shape)

    fig, ax = plt.subplots()

    COLORS = ['black','blue','green','yellow','purple']
    TEXT = ['blue','black','black','black','black']
    for state in range(TEST_LENGTH-1):
        COLOR = COLORS[int(state/20)+1]
        TEXT_C = TEXT[int(state/20)+1]
        embd = X[state]
        next_embd = X[state+1]
        ax.plot(float(embd[0]), float(embd[1]), linestyle='', ms=8, color = COLOR,marker='o', label='Test')
#        ax.annotate(str(state%20+1), xy = (float(next_embd[0]),float(next_embd[1])) , xytext = (float(embd[0]), float(embd[1])), arrowprops=dict(facecolor='black', shrink=0.01, width=0.01), color = TEXT_C )
        ax.annotate(str(state%20+1), xy = (float(embd[0]), float(embd[1])), color = TEXT_C )

    plt.grid()
    plt.savefig(NAME+str("_EmbdPlt2D_Perplexity=")+str(perp))
    plt.show()





