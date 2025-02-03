import SimulationData as SD
from itertools import combinations
import numpy as np
import pandas as pd
from utils import correlation, independence, cum31, cum22, pr


class Partial_Identification():
    
    def __init__(self, data):
        self.O = data.values.T
        self.tildeO = self.O.copy()
        self.indices = list(range(len(self.O)))
        self.M = np.eye(len(self.O))
        self.latent2homologous = {}

    def FindObservedRoot(self):
        # Identifying Observed Root Variables
        root_indices = []
        for i in self.indices:
            flag = 1
            for j in self.indices:
                if i == j or not correlation(self.tildeO[i], self.O[j])[0]:
                    continue
                else:
                    R = pr(self.O[j], self.O[i], self.tildeO[i])
                    if not independence(R, self.tildeO[i])[0]:
                        flag = 0
                        break
            if flag:
                root_indices.append(i)
        return root_indices

    def RemoveObservedRoot(self, root_indices):
        # Estimating the Effects of Observed Root Variables & Removing Observed Root Variables
        for i in root_indices:
            self.indices.remove(i)
        for i in root_indices:
            for j in self.indices:
                if correlation(self.tildeO[i], self.O[j])[0]:
                    self.M[j,i] = np.cov(self.tildeO[i], self.O[j])[0,1] / np.cov(self.tildeO[i], self.O[i])[0,1]
                    self.tildeO[j] = self.tildeO[j] - self.M[j,i] * self.tildeO[i]

    def FindLatentRoot(self):
        # Theorem 2
        root_indices = []
        for i in self.indices:
            flag = 1
            for (j, k) in combinations(self.indices,2):
                if i == j or i == k or not correlation(self.tildeO[i], self.O[j])[0] or not correlation(self.tildeO[i], self.O[k])[0]:
                    continue
                else:
                    R = pr(self.O[j], self.O[k], self.tildeO[i])
                    if not independence(R, self.tildeO[i])[0]:
                        flag = 0
                        break
            if flag:
                root_indices.append(i)
        return root_indices # all candidate HSu of all latent roots
        
    def MergeOverlap(self, root_indices):
        # Proposition 1
        root_dict = {}
        for j in root_indices:
            flag = 0
            for i in root_dict:
                if correlation(self.tildeO[i], self.O[j])[0]:
                    root_dict[i].append(j)
                    flag = 1
                    break
            if not flag:
                root_dict[j] = [j,]
        return [root_dict[k] for k in root_dict] # each element is a list, comprising all candidate HSu of a same latent root


    def FindTrueHSu(self, root_indices):
        # Proposition 2
        if self.M.shape[1] == len(self.O):
            return root_indices
        true_root_indices = []
        for root in root_indices:
            num_ancestors = [np.sum(np.abs(self.M[i, len(self.O):]) > 1e-6) for i in root]
            min_value = min(num_ancestors)
            true_root_indices.append([i for i, num in zip(root, num_ancestors) if num == min_value])
        return true_root_indices # each element is a list, comprising all HSu of a same latent root


    def RemoveLatentRoot(self, root_indices):
        # Estimating the Effects of Latent Root Variables & Removing Latent Root Variables
        for root in root_indices:
            for i in root:
                self.indices.remove(i)

        for root in root_indices:
            i, mi = root[0], [] # i: latent root's first Hsu; mi: multiple estimations of effects from latent root to its first HSu
            if len(root) > 1: # if latent root has multiple HSu, we need not use high-order statistics
                j = root[1]
                for k in root + self.indices:
                    if i == k or j == k or not correlation(self.tildeO[i], self.O[k])[0] or not correlation(self.tildeO[j], self.O[k])[0]:
                        continue
                    else:
                        product = np.cov(self.tildeO[i], self.O[j])[0, 1]
                        quotient = np.cov(self.tildeO[i], self.O[k])[0, 1] / np.cov(self.tildeO[j], self.O[k])[0, 1]
                        if product * quotient > 0:
                            mi.append((product * quotient) ** 0.5)
            if len(mi) == 0: # we have to use high-order statistics
                for j in root + self.indices:
                    if i == j or not correlation(self.tildeO[i], self.O[j])[0]:
                        continue
                    else:
                        product = np.cov(self.tildeO[i], self.O[j])[0, 1]
                        quotient1 = cum22(self.tildeO[i], self.O[j]) / cum31(self.O[j], self.tildeO[i])
                        if product * quotient1 > 0:
                            mi.append((product * quotient1) ** 0.5)
                        quotient2 = cum31(self.tildeO[i], self.O[j]) / cum22(self.tildeO[i], self.O[j])
                        if product * quotient2 > 0:
                            mi.append((product * quotient2) ** 0.5)
                        quotient_squred = cum31(self.tildeO[i], self.O[j]) / cum31(self.O[j], self.tildeO[i])
                        if quotient_squred > 0:
                            quotient3 = np.sign(product) * (quotient_squred ** 0.5)
                            mi.append((product * quotient3) ** 0.5)
            
            if len(mi) == 0:
                return 1
            self.M = np.concatenate([self.M, np.zeros([len(self.M), 1])], axis=1)
            self.latent2homologous[self.M.shape[1] - 1] = root
            self.M[i, -1] = np.median(np.array(mi)) # to reduce estimation error, we use median

            for j in root + self.indices:
                if i == j or not correlation(self.tildeO[i], self.O[j])[0]:
                    continue
                else:
                    product = np.cov(self.tildeO[i], self.O[j])[0, 1]
                    self.M[j, -1] = product / self.M[i, -1]
                    self.tildeO[j] = self.tildeO[j] - (self.M[j,-1] / self.M[i, -1]) * self.tildeO[i]
        return 0
    
    def run(self):
        while len(self.indices) > 0:
            root_indices = self.FindObservedRoot()
            while len(root_indices) > 0:
                self.RemoveObservedRoot(root_indices)
                root_indices = self.FindObservedRoot()
            root_indices = self.FindLatentRoot()
            if len(root_indices) == 0:
                break
            root_indices = self.MergeOverlap(root_indices)
            root_indices = self.FindTrueHSu(root_indices)
            flag = self.RemoveLatentRoot(root_indices)
            if flag:
                break
        num_observed, num_latent = len(self.O), self.M.shape[1] - len(self.O)
        self.M = np.concatenate([self.M, np.zeros([num_latent, self.M.shape[1]])], axis=0)
        for i in range(num_observed, self.M.shape[1]-1):
            for j in range(i+1, self.M.shape[1]):
                if np.all(np.abs(self.M[self.latent2homologous[j], i]) > 1e-6):
                    self.M[j, i] = 1.0
        for i in range(num_observed, self.M.shape[1]):
            self.M[i, i] = 1.0
        return self.M, num_observed, num_latent
    
