import numpy as np

N = 10
eps = np.ones(N) * 1.8
r0 = np.linspace(1.1, 2, N)
sigma = np.ones(N) * np.sqrt(0.02)
params = np.c_[eps, r0, sigma]

np.savetxt('params.txt', params, delimiter='\t')
