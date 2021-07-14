import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import math


def get_dataset(dataset_no,p,d,iteration):
    

    df = pd.read_csv("static/E-MEXP-clean.csv")

    p = p    #The population of particle
    d = d     #the dimension of particle

    ## Fetching Data from the DataSet

    #particle is the initial sub-matrix
    #index is the initial indices of the sub matrix
    particle = []
    index = np.zeros((p, d), dtype = 'int')
    for i in range(0,p):
        sampleParticle = []
        for j in range(0,d):
            idx = random.randint(0,8793)
            index[i, j] = idx
            x = df.iloc[[idx]]
            x = np.array(x)
            sampleParticle.append(np.array(x[0]))

        particle.append(sampleParticle)
    particle = np.array(particle,dtype='float16')

    # Normalization
#     - Scaling the values of th sub matrix between 0 to 1

    for i in range(particle.shape[0]):
        mms = MinMaxScaler()
        particle[i] = mms.fit_transform(particle[i])

    score = np.zeros((2,p,d),dtype='int')

    #create a Sub Matrix and also update the indices in the original indexes
    def createSubMatrix(idx):
        newMatrix = []
        for j in range(0,d):
            x = df.iloc[[idx[j]]]
            x = np.array(x)
            newMatrix.append(np.array(x[0]))
        newMatrix = np.array(newMatrix,dtype='float16')
        mms = MinMaxScaler()
        newMatrix = mms.fit_transform(newMatrix)
        return newMatrix
    #test = np.ones((1,6), dtype = 'int')
    #new = createSubMatrix(test,1)

    # Score Calculation
#     - It firstly, initializes the score of the 30 particles
#     - Func scoreCal takes any sub matrix and return the score

    def _init_score(par):          
        score = np.zeros((2,p),dtype='int')
        pid = 1
        for j in range(0,p):#protein wise
            ck = 0
            sc = 0
            for k in range(0, 35):  # eapar value of the protein
                test = 0 if k<=17 else 1
                # 2d ndarray for dist for train set
                e_dist = np.zeros((2, 34), dtype='float16')  # for index and value of euclidean
                # euclidean dist calculation d = √[(x2 – x1)^2 + (y2 – y1)^2].
                pos = 0
                for x in range(0, 35):
                    if x != k:
                        sum = 0
                        for y in range(0, d):  # row wise traversal
                            sum = sum + math.pow((par[j, y, k] - par[j, y, x]), 2)
                        e_dist[0, pos] = x + 1
                        e_dist[1, pos] = math.sqrt(sum)
                        pos = pos + 1
                idx = np.argpartition(e_dist[1], 3)
                # print(e_dist)
                # print(e_dist[0, idx[:3]])
                type = np.zeros(3, dtype='int')
                type = e_dist[0, idx[:3]]
                for a in range(0, 3):
                    type[a] = 0 if type[a]<=17 else 1
                temp = np.unique(type[a], return_counts=True)
                index = temp[1].argmax()
                pred = temp[0][index]
                train = pred
                ck = 1 if train == test else 0
                sc = sc + ck
                #print(ck)
                # print("Min={}".format(min(e_dist[1])))
                # print("Max={}".format(max(e_dist[1])))
                # print(idx)
                #print()
            score[0, j] = j + 1
            pid = pid + 1
            score[1, j] = sc
        #print(score)
        return score
    init_score = _init_score(particle)

    def scoreCal(subMatrix):
        ck = 0
        sc = 0
        for k in range(0, 35):  # eapar value of the protein
            test = 0 if k <= 17 else 1
            # 2d ndarray for dist for train set
            e_dist = np.zeros((2, 34), dtype='float16')  # for index and value of euclidean
            # euclidean dist calculation d = √[(x2 – x1)^2 + (y2 – y1)^2].
            pos = 0
            for x in range(0, 35):
                if x != k:
                    sum = 0
                    for y in range(0, d):  # row wise traversal
                        sum = sum + math.pow((subMatrix[y, k] - subMatrix[y, x]), 2)
                    e_dist[0, pos] = x + 1
                    e_dist[1, pos] = math.sqrt(sum)
                    pos = pos + 1
            idx = np.argpartition(e_dist[1], 3)
            # print(e_dist)
            # print(e_dist[0, idx[:3]])
            type = np.zeros(3, dtype='int')
            type = e_dist[0, idx[:3]]
            for a in range(0, 3):
                type[a] = 0 if type[a] <= 17 else 1
            temp = np.unique(type[a], return_counts=True)
            index = temp[1].argmax()
            pred = temp[0][index]
            train = pred
            ck = 1 if train == test else 0
            sc = sc + ck
            # print(ck)
            # print("Min={}".format(min(e_dist[1])))
            # print("Max={}".format(max(e_dist[1])))
            # print(idx)
            # print()
        return sc
    #scoreCal(particle[3])  #(index= 0-29)

    #main PSO INITIALIZATION
    velocity = np.zeros((p,d), dtype = 'float16')
    c1 = 2   #Cognitive acceleration coefficient (C1) 
    c2 = 2   #social acceleration coefficients (C2) 
    w = .5   #initial weight
    maxIt = 2 #Max iteration

    # gBestVal = np.max(init_score[1])
    # t = np.where(init_score[1] == np.max(init_score[1]))
    # gBest = index[t[0]]
    #test = np.ones((1,6), dtype = 'int')
    #new = createSubMatrix(test,1)
    position = np.copy(index)
    pBest = np.copy(position)

    def main_pso(position,pBest,velocity,init_score):
        gBestVal = np.max(init_score[1])
        t = np.where(init_score[1] == np.max(init_score[1]))
        gBest = np.copy(position[t[0]])
        for i in range(0, p):
            for j in range(0, d):
                velocity[i, j] = (w * velocity[i, j]) + (
                        c1 * round((random.uniform(0, 1)), 4) * (pBest[i, j] - position[i, j])) + (
                                         c2 * round((random.uniform(0, 1)), 4) * (gBest[0, j]) - position[i, j])
                velocity[i, j] = math.ceil(velocity[i, j])  # rounding off and absoluting the value
                position[i, j] = position[i, j] + velocity[i, j]
                position[i, j] = abs(position[i, j])
                if position[i, j].item() > 8793 or position[i, j].item() < 0:
                    position[i, j] = random.randint(1, 8790)
            tempMatrix = createSubMatrix(position[i])
            newScore = scoreCal(tempMatrix)
            if newScore > init_score[1, i].item():
                init_score[1, i] = np.copy(newScore)
                pBest[i] = np.copy(position[i])
            if newScore > gBestVal.item():
                gBestVal = np.copy(newScore)
                gBest[0] = np.copy(position[i])
    #         print()
#         print("Final Score:")
#         print(init_score[1])
        Init_Score =init_score[1]
#         print("The global best: {}".format(gBestVal))
        Global_Best = "{}".format(gBestVal)
#         print("The optimal Solution: {}".format(gBest[0]))
        Optimal_solution = "{}".format(gBest[0])
#         print()
        data_ = {"Init_Score":Init_Score,"Global_Best":Global_Best,"Optimal_solution":Optimal_solution}
        return data_
    data_list=[]
    for i in range(0,iteration):
    #         if i in d:
#             pass
#         else:
#             d[i]['final score':]
        data=main_pso(position,pBest,velocity,init_score)
        data['Iteration']=i
        data_list.append(data)
    return data_list