#!/usr/bin/python3.6
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
from sklearn.manifold import TSNE

TYPE = sys.argv[1]
NAME = sys.argv[2]
SIZE = int(sys.argv[3])

if TYPE == 'Convolution':
    model_conv_1 = torch.load("./"+NAME+"_conv_1")
    model_conv_2 = torch.load("./"+NAME+"_conv_2")
    model_conv_3 = torch.load("./"+NAME+"_conv_3")
    model_lin1 = torch.load("./"+NAME+"_lin_1")
    model_lin2 = torch.load("./"+NAME+"_lin_2")
    model_embd = torch.load("./"+NAME+"_embed")

if TYPE == 'Linear':
    model_lin1 = torch.load("./"+NAME+"_lin_1")
    model_lin2 = torch.load("./"+NAME+"_lin_2")
    model_embd = torch.load("./"+NAME+"_embed")


for j in range(SIZE*SIZE):
    curr_vec = np.zeros([SIZE, SIZE])
    row = int(j / SIZE)
    col = int(j % SIZE)
    curr_vec[row][col] = 1

    curr_context_vector = torch.tensor([[curr_vec]]).float().cuda()
    
    if TYPE == 'Convolution':
        conv = curr_context_vector
        conv = model_conv_1(conv)
        conv = model_conv_2(conv)
        conv = model_conv_3(conv)
        conv = conv.reshape(1, -1)
        lin = model_lin1(conv)
        lin = model_lin2(lin)
        embd = model_embd(lin)

    if TYPE == 'Linear':
        conv = curr_context_vector.reshape(1, -1)
        lin1 = model_lin1(conv)
        lin2 = model_lin2(lin1)
        embd = model_embd(lin2)
    
    try:
        X = torch.cat((X,embd), dim=0)
    except:
        X = embd
    
print(X.shape)
X = X.detach().cpu().numpy()
X_embed = TSNE(n_components=2, perplexity = 10, n_iter = 5000).fit_transform(X)
print(X_embed.shape)

fig, ax = plt.subplots()

for state in range(SIZE*SIZE):
    embd = X[state]
    if 0 <= int(state/SIZE) <= SIZE/2 -1:
        if 0 <= int(state % SIZE) <= SIZE/2 -1 :
            group = 'I'
            COLOR = 'red'
            TextColor = 'black'
        else:
            group = 'X'
            COLOR = 'blue'
            TextColor = 'black'
    else:
        if 0 <= int(state % SIZE) <= SIZE/2 -1:
            group = 'O'
            COLOR = 'black'
            TextColor = 'green'
        else:
            group = 'E'
            COLOR = 'yellow'
            TextColor = 'black'

    ax.plot(float(embd[0]), float(embd[1]), linestyle='', ms=8, color = COLOR,marker='o', label='Test')
    ax.annotate(state, (float(embd[0]),float(embd[1])), color = TextColor )



plt.grid()
plt.savefig(NAME+str("_EmbdPlt2D"))
plt.show()





