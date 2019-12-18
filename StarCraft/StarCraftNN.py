#!/usr/bin/python3.6

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import random
from time import sleep
import signal
import os
import scipy.sparse
from scipy.sparse import csr_matrix

def print_percentage(p_percent):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*p_percent, p_percent))
    sys.stdout.flush()

def GetData(dir_name):
    try:
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
    except:
        return 0,[0]


class SiameseNN_conv(nn.Module):
    def __init__(self, SIZE_nn, embedding_dim):
        super(SiameseNN_conv, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=9, padding=8)
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        #self.lin_1 = nn.Linear( (16*((SIZE_nn-8)/4) * ((SIZE_nn-8)/4) , SIZE_nn * SIZE_nn))

        self.lin_1 = nn.Linear(7744, SIZE_nn * SIZE_nn)

        self.embeddings = nn.Linear(SIZE_nn * SIZE_nn , embedding_dim)
        self.fin_lin = nn.Linear(1, 2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, sub_context, state):

        embed_ctx = self.conv_1(sub_context) # size-8 * size-8
        embed_ctx = self.pool(embed_ctx)     # size/2-4 * size/2-4
        embed_ctx = F.relu(self.conv_2(embed_ctx)) 
        embed_ctx = self.pool(embed_ctx)     #size/4-2 * size/4-2
        embed_ctx = F.relu(self.conv_3(embed_ctx)) 
        embed_ctx = self.pool(embed_ctx)  
        embed_ctx = embed_ctx.reshape(1, -1)
        embed_ctx = self.lin_1(embed_ctx)
        embed_ctx = self.embeddings(embed_ctx)

        embed_stt = self.conv_1(state) # size-8 * size-8
        embed_stt = self.pool(embed_stt)     # size/2-4 * size/2-4
        embed_stt = F.relu(self.conv_2(embed_stt))
        embed_stt = self.pool(embed_stt)     #size/4-2 * size/4-2
        embed_stt = F.relu(self.conv_3(embed_stt))
        embed_stt = self.pool(embed_stt)  
        embed_stt = embed_stt.reshape(1, -1)
        embed_stt = self.lin_1(embed_stt)
        embed_stt = self.embeddings(embed_stt)


        product = torch.mm(embed_ctx, torch.transpose(embed_stt, 0, 1) )
        outputs = torch.sigmoid(product)
        outputs = self.fin_lin(product)

        return outputs


DATA_LEN = [10]
EMBEDDINGS = [2]
CONTEXTS = [2]
Epochs = [20]
BATCHS = [32]

SAMPLE_RATE = 1000

directories = os.listdir("/home/data/starcraft/replay_data/")  

for cntx in CONTEXTS:
    print("-I- CONTEXTS: "+str(cntx))
    for data_len in DATA_LEN:
        print("-I- DATA_LEN: "+str(data_len))
        for embed in EMBEDDINGS:
            print("-I- EMBEDDINGS: "+str(embed))        
            for epoch in Epochs:
                print("-I- Epochs: "+str(epoch))        
                for batch in BATCHS:         
                    print("-I- BATCH: "+str(batch))                           
                    
                    CONTEXT_SIZE = int(cntx)
                    length = int(data_len)
                    EMBEDDING_DIM = int(embed)
                    EPOCHS = int(epoch)
                    BATCH = int(batch)
                    
                    name="StarCraftNN_CONTEXT="+str(CONTEXT_SIZE)+"_LENGTH="+str(length)+"_EMBEDINNGS="+str(embed)+"_EPOCHS="+str(epoch)+"_BATCH="+str(batch) 
                    print("\n-I- Running name:  \n\t" + name)
                    model = SiameseNN_conv(SIZE_nn = 84, embedding_dim = EMBEDDING_DIM).cuda()
                    print("-I- Finish building net")
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
                    
                    #try:
                    for ep in range(EPOCHS):
                        total_loss = 0
                        for dir_idx in range(data_len):                                             
                            rand_dir_idx = np.random.randint(26592)                           
                            state_length, data_states = GetData(directories[(dir_idx +rand_dir_idx)% 26592])
                            if state_length == 0:
                                continue
                            model.zero_grad()
                            
                            for idx, state in enumerate(data_states):
                            
                                while True:                            # -I- Generate index for the true and false samples
                                    true_idx = idx + np.random.randint(-CONTEXT_SIZE,CONTEXT_SIZE+1)
                                    false_idx = np.random.randint(0, state_length)
                                    if not (true_idx < 0 or true_idx >= state_length):
                                        break

                                #model.zero_grad()
                             
                                classification_pos = model( torch.tensor([data_states[true_idx]]).float().cuda(), torch.tensor([state]).float().cuda() )
                                labels_pos = torch.tensor([0]).cuda()
                                loss_pos = loss_function(classification_pos, labels_pos).cuda()
                            
                                classification_neg = model( torch.tensor([data_states[false_idx]]).float().cuda(), torch.tensor([state]).float().cuda() )
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
                                #loss.backward()
                                #optimizer.step()
                                total_loss += loss_pos.item() + loss_neg.item()

                                if total_samples_counter % batch == 0 or total_samples_counter == EPOCHS * length:
                                    loss.backward()
                                    optimizer.step()
                                    loss = torch.tensor([0]).float().cuda()
                                    model.zero_grad()
                            
                        #if ep % (EPOCHS/100) == 0 or ep == EPOCHS-1:              
                        print("***********Losses_Per_Epooch************")
                        print(total_loss)
                        print("****************************************")
                        losses.append(total_loss)
                        print_percentage(int(100*ep/EPOCHS))
                        #random.shuffle(Context)
                            
                    #except KeyboardInterrupt:
                    #    print("-I- manually stopped by keyboard")

                    #except Exception as e:
                    #    print(e)
                        
                    torch.save(model.conv_1, name+str("_conv_1"))
                    #torch.save(model.conv_2, name+str("_conv_2"))
                    #torch.save(model.conv_3, name+str("_conv_3"))
                    #torch.save(model.lin_1 , name+str("_lin_1"))
                    torch.save(model.lin_1 , name+str("_lin_1"))
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





    






