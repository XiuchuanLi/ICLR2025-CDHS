import numpy as np
from algorithm.utils import independence
from itertools import combinations


def SelectPdf(Num,data_type="exponential"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 5

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":
        noise = np.random.exponential(scale=1.0, size=Num)

    else: #gauss
        noise = np.random.normal(0, 1, size=Num)

    return noise


def normalize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


for Num in [5000,]:
    noises = []
    for i in range(20):
        print(i)
        while True:
            new_noise = normalize(SelectPdf(Num))
            if np.all(np.array([independence(new_noise, noise, 0.25)[0] for noise in noises])):
                noises.append(new_noise)
                break
    noises = np.stack(noises, axis=0)
    np.save(f'noise_{Num}.npy', noises)
