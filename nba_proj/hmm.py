import numpy as np
import ast
import math 

class hmm: 
    def __init__(self,cap_count=501):
        self.dp = [[0]*cap_count,[0]*cap_count,[0]*cap_count]
        self.backpointers = [[-1]*cap_count,[-1]*cap_count,[-1]*cap_count]
        # was 0.97,0,0.03; 0,0.97,0.03; 0.1,0.1,0.8
        self.transition_matrix = [[0.985, 0, 0.015],[0, 0.985, 0.015],[0.15, 0.15, 0.7]]

        self.count = 1
        self.cap_count = cap_count
        self.decoded_sequence = []

    def add_first(self,first):
        self.dp[0][1] = math.log(1/3) + math.log(first['left']+0.000001)
        self.dp[1][1] = math.log(1/3) + math.log(first['right']+0.000001)
        self.dp[2][1] = math.log(1/3) + math.log(first['none']+0.000001)

    # def assign_max_label(self,col):
    #     max_ind = -1
    #     max_val = -99999
    #     for i in range(len(col)):
    #         if(col[i] > max_val):
    #             max_ind = i
    #             max_val = col[i]

    #     labels = ['left','right','none']
    #     return labels[max_ind]
    
    # what to ask about: 
    # so.. i am able to max out a window and traverse through (say the window is of length 500)
    # but, if i want to do this for like 10000 frames, it would be impractical to store a 10000x3 array
    # so how do I efficiently shift out some of the frames in the window (say the 100 oldest) to make room for 100 new ones
    # while keeping the decoded sequence intact

    # another thing, whats the best way to start embedding things for the vit and the cnn
    # as in, how do you embed entire clips (i assume you would have to take like the longest clip and pad all the other ones with more tokens) 
    # also how should we fine tune the encoder/decoder and the attention layers


    # he said it should go back to the past 1 or 2 possessions
    # as far as the actual models go, he wasnt sure about vits
    # said that video cnns are the same as normal cnns but theres a filter for each frame

    # another thing he said was that you could just use the last 3 to 5 seconds of each possession 
    # because you don't really need the entire possession (as of now) and it avoids having to manually pad
    def add_col_to_lattice(self,col):
        if(col['left'] == 0):
            col['left'] = 0.000001 # replacing the 0
        if(col['right'] == 0):
            col['right'] = 0.000001
        if(col['none'] == 0):
            col['none'] = 0.000001
        if(self.count >= self.cap_count-1): 
            print('window is maxed out')
            print('need to call the shift method first')
            print(self.decoded_sequence)
            print('-----------------')
            input('stop')
            # a = self.dp[0][2:]
            # self.dp[0].pop(1)
            # a.append(col[])
            return -1
        else:
            self.count += 1

            from_l_to_l = self.dp[0][self.count-1] + math.log(self.transition_matrix[0][0]) + math.log(col['left'])
            # from_r_to_l can't exist
            from_n_to_l = self.dp[2][self.count-1] + math.log(self.transition_matrix[2][0]) + math.log(col['left'])

            l_result = max(from_l_to_l,from_n_to_l)

            if(from_l_to_l > from_n_to_l):
                self.backpointers[0][self.count] = 0
            else:
                self.backpointers[0][self.count] = 2


            # from_l_to_r can't exist
            from_r_to_r = self.dp[1][self.count-1] + math.log(self.transition_matrix[1][1]) + math.log(col['right'])
            from_n_to_r = self.dp[2][self.count-1] + math.log(self.transition_matrix[2][1]) + math.log(col['right'])

            r_result = max(from_r_to_r,from_n_to_r)

            if(from_r_to_r > from_n_to_r):
                self.backpointers[1][self.count] = 1
            else:
                self.backpointers[1][self.count] = 2
            
            from_l_to_n = self.dp[0][self.count-1] + math.log(self.transition_matrix[0][2]) + math.log(col['none'])
            from_r_to_n = self.dp[1][self.count-1] + math.log(self.transition_matrix[1][2]) + math.log(col['none'])
            from_n_to_n = self.dp[2][self.count-1] + math.log(self.transition_matrix[2][2]) + math.log(col['none'])

            n_result = max(from_l_to_n,from_r_to_n,from_n_to_n)

            self.backpointers[2][self.count] = np.argmax([from_l_to_n,from_r_to_n,from_n_to_n])

            # if(from_l_to_n > from_r_to_n and from_l_to_n > from_n_to_n):
            #     backpointers[1][count] = 1
            # else:
            #     backpointers[0][count] = 2

            self.dp[0][self.count] = round(l_result,3)
            self.dp[1][self.count] = round(r_result,3)
            self.dp[2][self.count] = round(n_result,3)

    def decode_sequence(self):
        # the self.counts here used to be self.cap_counts
        # print(self.decoded_sequence)
        # print('-----------')
        # if(len(self.decoded_sequence) == self.cap_count):
        #     print('need to extend decoded sequence')
        #     input('stop decoding')
        decoded_sequence = [-1]*self.count
        cur_position = np.argmax([self.dp[0][self.count-1],
                            self.dp[1][self.count-1],
                            self.dp[2][self.count-1]])
        # input(cur_position)
        for i in range(self.count-1,1,-1):
        
            decoded_sequence[i] = cur_position
            cur_position = self.backpointers[cur_position][i-1]

        dirs = {-1:-1,0:'left',1:'right',2:'none'}
        # print(self.dp)
        # print(self.backpointers)
        for i in range(len(decoded_sequence)):
            decoded_sequence[i] = dirs[decoded_sequence[i]]
        
        self.decoded_sequence = decoded_sequence

        
        return decoded_sequence

    


# --- old version ----
# f = open('temp_file_for_hmm.txt')
# # [[l_l, l_r, l_n],[r_l, r_r, r_n],[n_l, n_r, n_n]]

# # transition matrix; arbitrarily defined weights (for now)
# transition_matrix = [[0.97, 0, 0.03],[0, 0.97, 0.03],[0.1, 0.1, 0.8]]

# def assign_max_label(col):
#     max_ind = -1
#     max_val = -99999
#     for i in range(len(col)):
#         if(col[i] > max_val):
#             max_ind = i
#             max_val = col[i]

#     labels = ['left','right','none']
#     return labels[max_ind]

    
# # this is the dictionary that acts as the lattice 
# # emission_weights = {'left':[],'right':[],'none':[]}
# # [lefts, rights, nones]
# cap_count = 339
# dp = [[0]*cap_count,[0]*cap_count,[0]*cap_count]

# # backpointers will store which direction went into the current frame; it will store an index
# # index 0 means it came from 'left'; 1 is from 'right'; 2 is from 'none'
# backpointers = [[-1]*cap_count,[-1]*cap_count,[-1]*cap_count]

# first = ast.literal_eval(f.readline())
# dp[0][1] = math.log(1/3) + math.log(first['left'])
# dp[1][1] = math.log(1/3) + math.log(first['right'])
# dp[2][1] = math.log(1/3) + math.log(first['none'])

# # print(max(dp[0][1],dp[1][1],dp[2][1]))
# # dp[3][1] = assign_max_label([dp[0][1],dp[1][1],dp[2][1]])

# # backpointers = 
# # what temp looks like
# # {'left': 0.10031155116925383, 'right': 0.0164824890325102, 'none': 0.883205959815034}
# # {'left': 0.17815278309886906, 'right': 0.06079148882055376, 'none': 0.761055728081249}
# # {'left': 0.24912461881894368, 'right': 0.08604854378006203, 'none': 0.6648268374016664}
# # {'left': 0.19950708184328486, 'right': 0.041628390896697856, 'none': 0.7588062474170326}

# count = 1
# # input(dp)
# for probs in f.readlines():
#     temp = probs[0:-1]
#     temp = ast.literal_eval(temp)

#     # dp =   [[0,-2.59],
#             # [0,-3.73],
#             # [0,-1.44]]

#     # transition matrix
#     # [0.97, 0, 0.03]
#     # [0, 0.97, 0.03]
#     # [0.1, 0.1, 0.8]

#     if(count < cap_count-1):
#         count += 1

#         from_l_to_l = dp[0][count-1] + math.log(transition_matrix[0][0]) + math.log(temp['left'])
#         # from_r_to_l can't exist
#         from_n_to_l = dp[2][count-1] + math.log(transition_matrix[2][0]) + math.log(temp['left'])

#         l_result = max(from_l_to_l,from_n_to_l)

#         if(from_l_to_l > from_n_to_l):
#             backpointers[0][count] = 0
#         else:
#             backpointers[0][count] = 2


#         # from_l_to_r can't exist
#         from_r_to_r = dp[1][count-1] + math.log(transition_matrix[1][1]) + math.log(temp['right'])
#         from_n_to_r = dp[2][count-1] + math.log(transition_matrix[2][1]) + math.log(temp['right'])

#         r_result = max(from_r_to_r,from_n_to_r)

#         if(from_r_to_r > from_n_to_r):
#             backpointers[1][count] = 1
#         else:
#             backpointers[1][count] = 2
        
#         from_l_to_n = dp[0][count-1] + math.log(transition_matrix[0][2]) + math.log(temp['none'])
#         from_r_to_n = dp[1][count-1] + math.log(transition_matrix[1][2]) + math.log(temp['none'])
#         from_n_to_n = dp[2][count-1] + math.log(transition_matrix[2][2]) + math.log(temp['none'])

#         n_result = max(from_l_to_n,from_r_to_n,from_n_to_n)

#         backpointers[2][count] = np.argmax([from_l_to_n,from_r_to_n,from_n_to_n])

#         # if(from_l_to_n > from_r_to_n and from_l_to_n > from_n_to_n):
#         #     backpointers[1][count] = 1
#         # else:
#         #     backpointers[0][count] = 2

#         dp[0][count] = round(l_result,3)
#         dp[1][count] = round(r_result,3)
#         dp[2][count] = round(n_result,3)


#         # from_l_to_l = dp[0][count-1] + math.log(transition_matrix[0][0]) + math.log(temp['left'])
#         # # from_r_to_r can't exist
#         # from_l_to_n = dp[0][count-1] + math.log(transition_matrix[0][2]) + math.log(temp['left'])

#         # l_result = max(from_l_to_l,from_l_to_n)
#         # l_bp = -1


#         # # from_r_to_l can't exist
#         # from_r_to_r = dp[1][count-1] + math.log(transition_matrix[1][1]) + math.log(temp['right'])
#         # from_r_to_n = dp[1][count-1] + math.log(transition_matrix[1][2]) + math.log(temp['right'])

#         # r_result = max(from_r_to_r,from_r_to_n)
        
#         # from_n_to_l = dp[2][count-1] + math.log(transition_matrix[2][0]) + math.log(temp['none'])
#         # from_n_to_r = dp[2][count-1] + math.log(transition_matrix[2][1]) + math.log(temp['none'])
#         # from_n_to_n = dp[2][count-1] + math.log(transition_matrix[2][2]) + math.log(temp['none'])

#         # n_result = max(from_n_to_l,from_n_to_r,from_n_to_n)

#         # dp[0][count] = round(l_result,3)
#         # dp[1][count] = round(r_result,3)
#         # dp[2][count] = round(n_result,3)

#         # backpointers[0][count] = 
#         # dp[3][count] = assign_max_label([l_result,r_result,n_result])
#         # print([l_result,r_result,n_result])
#         # for g in dp: 
#         #     print(g)
#         # print()
#         # print()
#         # print()
#         # for g in backpointers:
#         #     print(g)
#         # input()
#         # break

# decoded_sequence = [-1]*cap_count
# cur_position = np.argmax([dp[0][cap_count-1],
#                      dp[1][cap_count-1],
#                      dp[2][cap_count-1]])
# # input(cur_position)
# for i in range(cap_count-1,1,-1):
   
#     decoded_sequence[i] = cur_position
#     cur_position = backpointers[cur_position][i-1]

# dirs = {-1:-1,0:'left',1:'right',2:'none'}
# for i in range(len(decoded_sequence)):
#     decoded_sequence[i] = dirs[decoded_sequence[i]]
# print(decoded_sequence)


    # print(temp)
    # if(len(emission_weights['left'] < 200)):
    #     emission_weights['left'].append(temp['left'])
    #     emission_weights['right'].append(temp['right'])
    #     emission_weights['none'].append(temp['none'])
    # else: 
    #     emission_weights['left'] = emission_weights['left'][1::]
    #     emission_weights['right'] = emission_weights['right'][1::]
    #     emission_weights['none'] = emission_weights['none'][1::]

    #     emission_weights['left'].append(temp['left'])
    #     emission_weights['right'].append(temp['right'])
    #     emission_weights['none'].append(temp['none'])


    # print(temp)

# print(f.readlines())