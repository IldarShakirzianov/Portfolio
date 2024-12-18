import numpy as np
import matplotlib.pyplot as plt

mu_s = 0  # drift for the stock
theta_r = 0.0297  # mean for the risk-free rate
kappa_r = 0.1  # speed of mean reversion for the risk-free rate

rho = 0.03 # correlation between stock and rf rate
N = 1000 #number of steps
T = 1 
i = 1000 #number of sims
S0 = 35
sigma_s = 0.4
sigma_r = 0.03
X = 35
r0 = 0.01
q = 0.03
S = np.zeros([i, N]) #array for the stock prices
R = np.zeros([i, N]) #array for the rf rate


deltat = T / N

for y in range(i - 1): #loop for i simulations
    S[y, 0] = S0
    R[y, 0] = r0

    for x in range(N - 1): 
        # Created correlated random variables
        z_s = np.random.normal(0, 1)
        z_r = rho * z_s + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        # Simulating a geom brownian motion for N steps
        S[y, x + 1] = S[y, x] * np.exp((mu_s - q - 0.5 * sigma_s**2) * deltat + sigma_s * np.sqrt(deltat) * z_s)

        #Simulating a mean-reverting process for the rf rate
        dR = kappa_r * (theta_r - R[y, x]) * deltat + sigma_r * np.sqrt(deltat) * z_r #Ornstein-Uhlenbeck process
        R[y, x + 1] = R[y, x] + dR
P = np.zeros([i - 1, 1])  #created an array of zeros for payoffs

for y in range(i - 1):
    P[y] = np.maximum(X - S[y, N - 1], 0)

payoff = np.average(P)
result = np.exp(-r0*T)*payoff
print(result) 
