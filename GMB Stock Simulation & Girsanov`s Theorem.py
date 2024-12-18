import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = 487.16
alpha = 0.000476237
sigma = 0.03848375542
T = 252
N = 252
dt = T/N
M = 3000
r = 0.02
St = np.zeros(N+1)

def stock_price(S, alpha, sigma, T, N, dt, M):
    simulations = np.zeros((N + 1, M))
    all_St = np.zeros((N + 1, M))  # Separate array for all St
    all_Zt = np.zeros((N + 1, M))  # Separate array for all Zt

    for j in range(M):   
        t = np.linspace(0, T, N+1)
   
        St[0] = S
        e = np.random.normal(0, 1, N+1)

        for i in range(1, N+1):
            St[i] = St[i-1]*np.exp((alpha-0.5*sigma**2)*dt + sigma*e[i-1]*np.sqrt(dt))
            all_St[i, j] = St[i]  # Save St for each simulation in the separate array
            all_Zt[i, j] = np.exp(-((alpha-r)/sigma)*e[i-1]*dt - 0.5*((alpha-r)/sigma)**2*dt)  # Save Zt for each simulation in the separate array

        simulations[:, j] = St

    return t, simulations, all_St, all_Zt

t, stock_prices, all_St, all_Zt = stock_price(S, alpha, sigma, T, N, dt, M)

# Flatten the all_St and all_Zt arrays
all_St_flat = all_St.flatten()
all_Zt_flat = all_Zt.flatten()

# Create a DataFrame with all St and Zt values
df_all_data = pd.DataFrame({'All St': all_St_flat, 'All Zt': all_Zt_flat})

# Export the DataFrame to Excel
df_all_data.to_excel('simulations_data6.xlsx', index=False)
print("done")

