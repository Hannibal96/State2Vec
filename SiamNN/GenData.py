#!/usr/bin/python3.6

import numpy as np
import sys
import matplotlib.pyplot as plt

def print_board(board):
    size = len(board[0])
    for x in range(size):
        for y in range(size):
            print(int(board[x][y]), end = '')
    print("", end = " ")

def copy_board(board):
    next_board = [row[:] for row in board]
    return next_board

def get_next_state(board, x, y):
    DEAD = 0
    ALIVE = 1
    current_state = board[x][y]
    next_state = current_state
    number_of_negibors = 0
    for n in [-1, 0, 1]:
        for m in [-1, 0, 1]:
            if (x+n < 0 or x+n >= len(board)) or (y+m < 0 or y+m >= len(board[0])) or n == m == 0:
                continue
            if board[x + n][y + m] == ALIVE:
                number_of_negibors += 1
    if number_of_negibors < 2:
        next_state = DEAD
    elif number_of_negibors == 3:
        next_state = ALIVE
    elif number_of_negibors > 3:
        next_state = DEAD
    return next_state



def gen_random_data_divided(size=6, iterations=1000):
    time_states = list()
    curr_pos_x_in = np.random.randint(size/2)
    curr_pos_y_in = np.random.randint(size/2)
    out = np.random.randint(4)
    flag_just_switch = False
    for i in range(iterations):
        curr_pos_x_out = int(out / 2)
        curr_pos_y_out = out % 2
        curr_x = int(curr_pos_x_out * size / 2) + curr_pos_x_in
        curr_y = int(curr_pos_y_out * size / 2) + curr_pos_y_in
        state = np.zeros([size, size])
        state[curr_x][curr_y] = 1
        time_states.append(state)
        if curr_pos_x_in == (size/2-1)/2 and curr_pos_y_in == (size/2-1)/2 and flag_just_switch == False:
            flag_just_switch = True
            out = (out+1) % 4

        else:
            flag_just_switch = False
            if curr_pos_x_in == 0:
                plus_x = np.random.randint(0, 2)
            elif curr_pos_x_in == size/2-1:
                plus_x = np.random.randint(-1, 1)
            else:
                plus_x = np.random.randint(-1, 2)
            if curr_pos_y_in == 0:
                plus_y = np.random.randint(0, 2)
            elif curr_pos_y_in == size/2 - 1:
                plus_y = np.random.randint(-1, 1)
            else:
                plus_y = np.random.randint(-1, 2)
            curr_pos_x_in += plus_x
            curr_pos_y_in += plus_y
    return time_states


def game_of_life(SIZE_lg, steps):
    board = np.random.randint(2, size=(SIZE_lg, SIZE_lg))
    states_lg = list()
    x_axis = len(board)
    y_axis = len(board[0])
    states_lg.append(board)
    #print_board(board)
    counter_lg = 0

    #print("-I- Generating Data")

    for t in range(steps-1):
        next_board = np.random.randint(2, size=(SIZE_lg, SIZE_lg))
        for n in range(x_axis):
            for m in range(y_axis):
                next_board[n][m] = get_next_state(board, n, m)
        if counter_lg >= 20 :
            next_board = np.random.randint(2, size=(SIZE_lg, SIZE_lg))
            counter_lg = 0
            #print(" ")
        board = copy_board(next_board)
        #print_board(board)
        states_lg.append(board)
        counter_lg += 1
        #print_percentage(int(100*t/steps))
        
    #print("\n")
    return states_lg



def print_board(board):
    size = len(board[0])
    for x in range(size):
        for y in range(size):
            print(int(board[x][y]), end = '')
    print("", end = " ")

def print_percentage(p_percent):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*p_percent, p_percent))
    sys.stdout.flush()

def show_data(images):
    for s_img in images:
        plt.imshow(s_img)
        plt.show()



#img = game_of_life(5,10)
#show_data(img)



