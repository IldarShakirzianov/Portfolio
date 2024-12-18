import numpy as np
import matplotlib.pyplot as plt

mu = 0
N = 1000
T = 1
i = 1000
S0 = 35
sigma = 0.4
X = 35
r = 0.01
q = 0.03
S = np.zeros([i,N]) #created an array of zeros
P = np.zeros([i-1,1]) #created an array of zeros for payoffs

deltat = (T/N)
for y in range(0,i-1): #loop for i simulations
    S[y,0]=S0
    dropped_below_30 = False #condition
    for x in range(0,N-1): #simulating a geom brownian motion for N steps
        S[y,x+1]= S[y,x]*np.exp((r-q-0.5*sigma**2)*deltat + sigma* np.sqrt(deltat)*np.random.normal(0,1))
        if S[y,x+1] <30:
            dropped_below_30 = True
            break
    if dropped_below_30 == True:
        P[y] = 0
    else:
        P[y]=np.maximum(X-S[y,N-1],0)
    

payoff = np.average(P)
result = np.exp(-r*T)*payoff
print(result)
