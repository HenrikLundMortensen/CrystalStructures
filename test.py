import numpy as np


def calcE(x):
    return x


# Best to take a number of atoms that has (N / Ncores) % 2 = 0
N = 16
Ncores = 4
Nwork = int(np.ceil(N / Ncores))

# Distribute workload
Ncores_use = min(N, Ncores)
work_list = []
for i in range(Ncores_use):
    work_list.append([i])
for i in range(Ncores, N):
    work_list[3 - i % Ncores_use].append(i)

