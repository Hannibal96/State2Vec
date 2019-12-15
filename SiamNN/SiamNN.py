#!/usr/bin/python3.6

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import random
from GenData import *
from time import sleep
import signal


def sigterm_handler(_signo, _stack_frame):
    print("-I- catch kill exiting gracefully...")
    raise Exception()
signal.signal(signal.SIGTERM, sigterm_handler)

class SiameseNN_conv(nn.Module):
    def __init__(self, SIZE_nn, embedding_dim):
        super(SiameseNN_conv, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=7, padding=3)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.lin_1 = nn.Linear(128 * SIZE_nn * SIZE_nn, 16 * SIZE_nn * SIZE_nn)
        self.lin_2 = nn.Linear(16 * SIZE_nn * SIZE_nn, SIZE_nn * SIZE_nn)

        self.embeddings = nn.Linear(SIZE_nn * SIZE_nn , embedding_dim)
        self.fin_lin = nn.Linear(1, 2)

    def forward(self, sub_context, state):
        
        embed_ctx = self.conv_1(sub_context)
        embed_ctx = self.conv_2(embed_ctx)
        embed_ctx = self.conv_3(embed_ctx)

        embed_ctx = embed_ctx.reshape(1, -1)
        embed_ctx = self.lin_1(embed_ctx)
        embed_ctx = self.lin_2(embed_ctx)
        embed_ctx = self.embeddings(embed_ctx)


        embed_stt = self.conv_1(state)
        embed_stt = self.conv_2(embed_stt)
        embed_stt = self.conv_3(embed_stt)

        embed_stt = embed_stt.reshape(1, -1)
        embed_stt = self.lin_1(embed_stt)
        embed_stt = self.lin_2(embed_stt)
        embed_stt = self.embeddings(embed_stt)


        product = torch.mm(embed_ctx, torch.transpose(embed_stt, 0, 1) )
        
        outputs = torch.sigmoid(product)
        outputs = self.fin_lin(product)
        
        return outputs


class SiameseNN_linear(nn.Module):
    def __init__(self, SIZE_nn, embedding_dim):
        super(SiameseNN_linear, self).__init__()

        self.lin_1 = nn.Linear(SIZE_nn * SIZE_nn, SIZE_nn * SIZE_nn * 16)
        self.lin_2 = nn.Linear(16 * SIZE_nn * SIZE_nn, SIZE_nn * SIZE_nn)
        self.embeddings = nn.Linear(SIZE_nn * SIZE_nn, embedding_dim)
        self.fin_lin = nn.Linear(1, 2)

    def forward(self, sub_context, state):

        embed_ctx = sub_context.reshape(1, -1)
        embed_ctx = self.lin_1(embed_ctx)
        embed_ctx = self.lin_2(embed_ctx)
        embed_ctx = self.embeddings(embed_ctx)

        embed_stt = sub_context.reshape(1, -1)
        embed_stt = self.lin_1(embed_stt)
        embed_stt = self.lin_2(embed_stt)
        embed_stt = self.embeddings(embed_stt)

        product = torch.mm(embed_ctx, torch.transpose(embed_stt, 0, 1) )

        outputs = torch.sigmoid(product)
        outputs = self.fin_lin(product)

        return outputs



NN_TYPE = ["Convolution"]#, "Linear"]
ITER_LEN = [10000]
EMBEDDINGS = [4]
CONTEXTS = [2]
DATAS = ['Walls'] # 'Walls'
Sizes = [6]
Epochs = [500]
BATCHS = [1024]

for data in DATAS:
    for size in Sizes:
        for cntx in CONTEXTS:
            for iter in ITER_LEN:
                for embed in EMBEDDINGS:
                    for nn_type in NN_TYPE:
                        for epoch in Epochs:
                            for batch in BATCHS:
                            
                                    SIZE = int(size)
                                    CONTEXT_SIZE = int(cntx)
                                    ITERATIONS = int(iter)
                                    NN = nn_type
                                    EMBEDDING_DIM = int(embed)
                                    EPOCHS = int(epoch)
                                    DATA = data
                                    BATCH = int(batch)
                                    
                                    name="DATA="+str(data)+"_SIZE="+str(size)+"_CONTEXT="+str(cntx)+"_ITERATIONS="+str(iter)+"_EMBEDINNGS="+str(embed)+"_NN="+str(nn_type)+"_EPOCHS="+str(epoch)+"_BATCH="+str(batch) 
                                    print("\n====Running name:  \n\t" + name)
                
                                    if DATA == 'Walls':
                                        data_img = gen_random_data_divided(SIZE, ITERATIONS)
                                    elif DATA == 'GOL':
                                        data_img = game_of_life(SIZE, ITERATIONS)
                                    #show_data(data_img)
                
                                    if CONTEXT_SIZE == 2:
                                        Context = [([ [data_img[i], data_img[i+1], data_img[i+3], data_img[i+4]] , 
                                            [data_img[(np.random.randint(ITERATIONS))%ITERATIONS],data_img[(np.random.randint(ITERATIONS))%ITERATIONS], data_img[(np.random.randint(ITERATIONS))%ITERATIONS], data_img[(np.random.randint(ITERATIONS))%ITERATIONS]]], 
                                            data_img[i+2])
                                                    for i in range(len(data_img) - 4)]
                
                                    if CONTEXT_SIZE == 1:
                                        Context = [([ [data_img[i], data_img[i+2]] ,
                                            [data_img[(np.random.randint(ITERATIONS))%ITERATIONS], data_img[(np.random.randint(ITERATIONS))%ITERATIONS]]],
                                            data_img[i+1])
                                                    for i in range(len(data_img) - 2)]
                
                                    losses = []
                                    correctness = []
                                    loss_function = torch.nn.CrossEntropyLoss()
                
                                    if NN == 'Convolution':
                                        model = SiameseNN_conv(SIZE, EMBEDDING_DIM).cuda()
                                    elif NN == 'Linear':
                                        model = SiameseNN_linear(SIZE, EMBEDDING_DIM).cuda()
                                    else:
                                        print("-E- bad NN")
                                        exit()
                
                                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                                    print(model)
                                    counter = 0
                                    s_counter = 0
                                    p_counter = 0
                                    res_counter = 0
                                    percent = 0
                                    print_percentage(percent)
                
                                    loss = torch.tensor([0]).float().cuda()
                
                                    try:
                                        for epoch in range(EPOCHS):
                                            total_loss = 0
                
                                            for context, state in Context:
                                                rand_true = np.random.randint(CONTEXT_SIZE*2)
                                                rand_flse = np.random.randint(CONTEXT_SIZE*2)
                                                for i in range( 2 * CONTEXT_SIZE ):
                                                    true_idx = (i+rand_true) % (2*CONTEXT_SIZE)
                                                    flse_idx = (i+rand_flse) % (2*CONTEXT_SIZE)
                                                    model.zero_grad()
                
                                                    classification_pos = model( torch.tensor([[context[0][true_idx]]]).float().cuda(), torch.tensor([[state]]).float().cuda() )
                                                    labels_pos = torch.tensor([0]).cuda()
                                                    loss_pos = loss_function(classification_pos, labels_pos).cuda()
                
                                                    classification_neg = model( torch.tensor([[context[1][flse_idx]]]).float().cuda(), torch.tensor([[state]]).float().cuda() )
                                                    labels_neg = torch.tensor([1]).cuda()
                                                    loss_neg = loss_function(classification_neg, labels_neg).cuda()
                
                                                    if bool(classification_pos[0][0] > classification_pos[0][1]):
                                                        res_counter += 1
                                                    if bool(classification_neg[0][0] < classification_neg[0][1]):
                                                        res_counter += 1
                                                    s_counter += 1
                                                    p_counter += 1
                
                                                    if (int(np.random.randint(5000))) == 0 or s_counter == 1:
                                                        print("******Correctness******")
                                                        print(classification_pos)
                                                        print(labels_pos)
                                                        print(classification_neg)
                                                        print(labels_neg)
                                                        print("res_percentage = "+str(100*res_counter/(p_counter*2))+"%")
                                                        print("***********************")
                                                        correctness.append(100*res_counter/(p_counter*2))
                                                        p_counter = 0
                                                        res_counter = 0
                
                                                    loss += loss_pos + loss_neg
                                                    total_loss += loss.item()
                                                    
                                                    if s_counter % batch == 0 or s_counter == EPOCHS*len(Context)*2 * CONTEXT_SIZE:
                                                        loss.backward()
                                                        optimizer.step()
                                                        loss = torch.tensor([0]).float().cuda()
                
                                            print("***************Losses***************")
                                            print(total_loss)
                                            print("************************************")
                                            losses.append(total_loss)
                                            counter += 1
                                            print_percentage(int(100*counter/EPOCHS))
                                            random.shuffle(Context)
                
                                    except KeyboardInterrupt:
                                        print("-I- manually stopped by keyboard")
                
                                    except Exception as e:
                                        print(e)
                
                
                                    if NN == 'Convolution':
                                        torch.save(model.conv_1, name+str("_conv_1"))
                                        torch.save(model.conv_2, name+str("_conv_2"))
                                        torch.save(model.conv_3, name+str("_conv_3"))
                
                                        torch.save(model.lin_1 , name+str("_lin_1"))
                                        torch.save(model.lin_2 , name+str("_lin_2"))
                
                                        torch.save(model.embeddings, name+str("_embed") )
                
                
                                    elif NN == 'Linear':
                                        torch.save(model.lin_1, name+str("_lin_1"))
                                        torch.save(model.lin_2, name+str("_lin_2"))
                                        torch.save(model.embeddings, name+str("_embed") )
                
                
                                    print(losses)
                                    print("\nInital loss: "+str(losses[0])+", Final loss: "+str(losses[-1]))
                                    print("FINAL : res_percentage = "+str(100*res_counter/(p_counter*2))+"%")
                
                
                                    try:
                                        plt.plot(losses)
                                        plt.savefig(name+str("_lossPlt"))
                                        plt.show()
                
                                        plt.plot(correctness)
                                        plt.savefig(name+str("_correctnessPlt"))
                                        plt.show()
                
                
                                    except:
                                        torch.save(torch.tensor(losses),name+"_losses")
                                        torch.save(torch.tensor(correctness),name+"_correctness")
                
                
                
                
