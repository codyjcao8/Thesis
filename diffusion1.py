#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:59:17 2017

@author: codycao
"""

import numpy as np
import matplotlib.pyplot as plt


size = 100                              #size of the playing field
players = 100
np.random.seed(25)

g = 1                                   #which game to play
PERTURBATION = False                    #implement perturbation of game matrix
HALFinit = True                         #create two halves of pure s1 and s2
w = np.power(10.0,-1)                   #epsilon

            

if g == 1:
    A = [[2,0],[1,1]]                    #Stag hunt
elif g == 2:
    A = [[2,1],[3,0]]                    #Snowdrift
elif g == 3:
    A = [[2,0],[3,1]]                    #Prisoner's Dilemma
else:
    A = [[2,0],[0,1]]                    #Generic co-op game


A = np.array(A,dtype = int)


G = [[1 for i in range(2)] for i in range(2)]
G = np.array(G)
u = 0.1


if PERTURBATION:
    G = u*G + A
else:
    G = A


T = np.random.rand(size)               #keeps track of the relative frequencies

                  
if HALFinit:
    for i in range(int(len(T)/2)):
        T[i] = 0
    for i in range(int(len(T)/2),len(T)):
        T[i] = 1

#for i in range(len(T)):
#    T[i] = 1/2.0
#          
#T[0:30] = 1
#T[70:99] = 0

  
S = np.zeros((2,size))                 #counts the actual population
for i in range(size):
    S[0,i] = T[i]*players
    S[1,i] = players - S[0,i]


T = np.expand_dims(T,axis = 0)
initDisplay = T.copy()
initDisplay = np.tile(initDisplay,(50,1))
p = plt.imshow(initDisplay, interpolation='nearest')
plt.set_cmap('Greys')
plt.pause(2)


loops = 10000



#count = np.zeros(loops)
e1 = np.array([1,0])
e2 = np.array([0,1])



"""
direction of travelling wave determined by:
    beta/alpha >< F(mu1/mu2) where F(u) = u**-.61
"""

mu1 = 1
mu2 = 1
mu = np.array([mu1,mu2])


for loop in range(loops):
    Sc = np.zeros(size)
    delta = np.zeros((2,size))

    for i in range(size):
        N = S[0,i] + S[1,i]
        n = np.array([S[0,i],S[1,i]])
        s1fit = np.dot(np.dot(e1,G),n)/N
        s2fit = np.dot(np.dot(e2,G),n)/N
        avgfit = np.dot(np.dot(n,G),n)/N**2

        delta[0,i] = S[0,i]*(s1fit - avgfit)
        delta[1,i] = S[1,i]*(s2fit - avgfit)


    intermediate = S + w*delta

    print(S[0,48:52])
    print("_______________________")
    print("_______________________")
    newS = np.zeros((2,size))

    for i in range(size):
        newS[:,i] = (1-w*mu)*intermediate[:,i] + 0.5*w*mu*intermediate[:,(i-1)%size] + 0.5*w*mu*intermediate[:,(i+1)%size]

    S = newS
    newT = newS[0,:]/(newS[0,:]+newS[1,:])
#    count[loop] = sum(newT/n)
    T = newT.copy()
    T = np.expand_dims(T,axis = 0)
    display = np.tile(T,(50,1))
    p.set_data(display)
    plt.pause(0.0001)


#x = range(loops)
#plt.plot(x,count)
#plt.show()
#raw_input()
