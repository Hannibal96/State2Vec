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

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.lin_1 = nn.Linear(128 * SIZE_nn * SIZE_nn, 16 * SIZE_nn * SIZE_nn)
        self.lin_2 = nn.Linear(16 * SIZE_nn * SIZE_nn, SIZE_nn * SIZE_nn)

        self.embeddings = nn.Linear(SIZE_nn * SIZE_nn , embedding_dim)
        self.fin_lin = nn.Linear(1, 2)

    def forward(self, sub_context, state):
        
        embed_ctx = F.relu(self.conv_1(sub_context))
        embed_ctx = F.relu(self.conv_2(embed_ctx))
        embed_ctx = self.conv_3(embed_ctx)

        embed_ctx = embed_ctx.reshape(1, -1)
        embed_ctx = F.relu(self.lin_1(embed_ctx))
        embed_ctx = self.lin_2(embed_ctx)
        embed_ctx = self.embeddings(embed_ctx)


        embed_stt = F.relu(self.conv_1(state))
        embed_stt = F.relu(self.conv_2(embed_stt))
        embed_stt = self.conv_3(embed_stt)

        embed_stt = embed_stt.reshape(1, -1)
        embed_stt = F.relu(self.lin_1(embed_stt))
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
DATA_LEN = [1000]
EMBEDDINGS = [2]
CONTEXTS = [2]
DATAS = ['GOL'] # 'Walls'
Sizes = [5]
Epochs = [200]
BATCHS = [1,8,32,64]

SAMPLE_RATE = 10000

for data in DATAS:
    for size in Sizes:
        for cntx in CONTEXTS:
            for data_len in DATA_LEN:
                for embed in EMBEDDINGS:
                    for nn_type in NN_TYPE:
                        for epoch in Epochs:
                            for batch in BATCHS:                            
                                SIZE = int(size)
                                CONTEXT_SIZE = int(cntx)
                                length = int(data_len)
                                NN = nn_type
                                EMBEDDING_DIM = int(embed)
                                EPOCHS = int(epoch)
                                DATA = data
                                BATCH = int(batch)
                                
                                name="ForwardNN_DATA="+str(data)+"_SIZE="+str(size)+"_CONTEXT="+str(CONTEXT_SIZE)+"_LENGTH="+str(length)+"_EMBEDINNGS="+str(embed)+"_NN="+str(nn_type)+"_EPOCHS="+str(epoch)+"_BATCH="+str(batch) 
                                print("\n====Running name:  \n\t" + name)
            
                                if DATA == 'Walls':
                                    data_img = gen_random_data_divided(SIZE, length)
                                elif DATA == 'GOL':
                                    data_img = game_of_life(SIZE, length)
                                #show_data(data_img)                 
                                torch.save(data_img, name+"_data")
            
                                if NN == 'Convolution':
                                    model = SiameseNN_conv(SIZE, EMBEDDING_DIM).cuda()
                                elif NN == 'Linear':
                                    model = SiameseNN_linear(SIZE, EMBEDDING_DIM).cuda()
                                    
                                loss_function = torch.nn.CrossEntropyLoss()
                                optimizer = optim.Adam(model.parameters(), lr=0.001)
                                print(model)
                                
                                losses = []
                                correctness = []
                                
                                total_samples_counter = 0
                                temp_samples_counter = 0
                                temp_res_counter = 0
                                total_res_counter = 0
                                epochs_counter = 0
                                print_percentage(int(100*epochs_counter/EPOCHS))
            
                                loss = torch.tensor([0]).float().cuda()                                    
                                model.zero_grad()
                                
                                try:
                                    for ep in range(EPOCHS):
                                        total_loss = 0                                    
                                        for idx, state in enumerate(data_img):
                                            true_idx = (idx + 1)%len(data_img)
                                            flse_idx = np.random.randint(length)
                                            
                                            #model.zero_grad()
            
                                            classification_pos = model( torch.tensor([[data_img[true_idx]]]).float().cuda(), torch.tensor([[state]]).float().cuda() )
                                            labels_pos = torch.tensor([0]).cuda()
                                            loss_pos = loss_function(classification_pos, labels_pos).cuda()
        
                                            classification_neg = model( torch.tensor([[data_img[flse_idx]]]).float().cuda(), torch.tensor([[state]]).float().cuda() )
                                            labels_neg = torch.tensor([1]).cuda()
                                            loss_neg = loss_function(classification_neg, labels_neg).cuda()
        
                                            if bool(classification_pos[0][0] > classification_pos[0][1]):
                                                temp_res_counter += 1
                                                total_res_counter += 1
                                            if bool(classification_neg[0][0] < classification_neg[0][1]):
                                                temp_res_counter += 1
                                                total_res_counter += 1
                                            total_samples_counter += 1
                                            temp_samples_counter += 1
        
                                            if (int(np.random.randint(SAMPLE_RATE))) == 0 or total_samples_counter == 1:
                                                print("******Correctness******")
                                                print(classification_pos)
                                                print(labels_pos)
                                                print(classification_neg)
                                                print(labels_neg)
                                                print("res_percentage = "+str(100*temp_res_counter/(temp_samples_counter*2))+"%")
                                                print("***********************")
                                                correctness.append(100*temp_res_counter/(temp_samples_counter*2))
                                                temp_res_counter = 0
                                                temp_samples_counter = 0
        
                                            loss += loss_pos + loss_neg
                                            total_loss += loss.item()
                                            
                                            if total_samples_counter % batch == 0 or total_samples_counter == EPOCHS * length:
                                                loss.backward()
                                                optimizer.step()
                                                loss = torch.tensor([0]).float().cuda()
                                                model.zero_grad()
                                            
                                        if ep % (EPOCHS/100) == 0 or ep == EPOCHS-1:              
                                            print("***********Losses_Per_Epooch************")
                                            print(total_loss)
                                            print("****************************************")
                                            losses.append(total_loss)
                                            epochs_counter += 1
                                            print_percentage(int(100*epochs_counter/EPOCHS))
                                        #random.shuffle(Context)
                                            
            
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
                                print("FINAL : res_percentage = "+str(100*total_res_counter/(total_samples_counter*2))+"%")
            
            
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
            
            
            
            
