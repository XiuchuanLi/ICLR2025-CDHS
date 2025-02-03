import SimulationData as SD
from itertools import permutations
import numpy as np
import copy

def support(input, threshold=1e-6):
    return np.abs(input) > threshold

class Full_Identification():

    def __init__(self, M, num_observed, num_latent):
        self.M_OO_initial = M[:num_observed, :num_observed]
        self.M_LL = M[num_observed:, num_observed:]
        self.M_OL = np.linalg.inv(M[:num_observed, :num_observed]) @ M[:num_observed, num_observed:]
        self.num_observed = num_observed
        self.num_latent = num_latent

        self.DePairs_full = self.get_DePairs() # dict, key is hop, value is a tuple (ancestor, descendant)
        self.GHSu_full = self.get_GHSu() # dict, key is latent variable, value is the list of its all GHSu
        self.DePairs, self.GHSu = self.discard_pathological()

        self.A_OL = np.zeros([self.num_observed, self.num_latent])
        self.no_edge = set()
    
    def longest_path_length(self, start, end):
        if not support(self.M_LL[end, start]):
            return 0
        A_LL = np.eye(len(self.M_LL)) - np.linalg.inv(self.M_LL)
        path_matrices = {1:A_LL}
        for i in range(2,len(A_LL)):
            path_matrices[i] = path_matrices[i - 1] @ A_LL
        for i in range(len(A_LL) - 1, 0, -1):
            if support(path_matrices[i][end, start]):
                return i

    def get_DePairs(self):
        # get k-hop descendants
        DePairs = {i:[] for i in range(1, self.num_latent)}
        for (i,j) in permutations(range(self.num_latent), 2):
            hop = self.longest_path_length(i, j)
            if hop > 0:
                DePairs[hop].append((i, j))
        return DePairs
    
    def get_GHSu(self):
        # get GHSu
        GHSu = {}
        for i in range(self.num_latent):
            GHSu[i] = []
            for j in range(self.num_observed):
                if np.all(support(self.M_LL[i, :]) == support(self.M_OL[j, :])):
                    GHSu[i].append(j)
        return GHSu
    
    def discard_pathological(self):
        # discard variables that have insufficient GHSu
        discard = []
        for i in range(self.num_latent):
            if np.sum(support(self.M_LL[i,:])) == 1: # for root, 1 GHSu is enough
                continue
            if len(self.GHSu_full[i]) < 2:
                discard.append(i)
        if len(discard) == 0:
            return copy.deepcopy(self.DePairs_full), copy.deepcopy(self.GHSu_full)
        
        discard = np.where(support(np.sum(self.M_LL[:,discard], axis=1)))[0].tolist()
        for i in discard:
            self.M_LL[i,:] = np.eye(self.num_latent)[i,:]
            self.M_LL[:,i] = np.eye(self.num_latent)[:,i]
            self.M_OL[:,i] = 0
            for j in self.GHSu_full[i]:
                self.M_OL[j,:] = 0
        GHSu, DePairs = {}, {}
        for i in range(self.num_latent):
            if i not in discard:
                GHSu[i] = copy.deepcopy(self.GHSu_full[i])
        for hop in range(1, self.num_latent):
            DePairs[hop] = []
            for (i, j) in self.DePairs_full[hop]:
                if i not in discard and j not in discard:
                    DePairs[hop].append((i,j))
        return DePairs, GHSu


    def mu(self, start, end, index):
        # get mu^{L_{start}}_{O^*_{end}}
        result = self.M_OL[self.GHSu[end][index], start]
        if start == end:
            return result
        for i in range(self.num_latent):
            if support(self.M_LL[end, i]) and support(self.M_LL[i, start]) and i != start and i != end:
                result -= self.M_LL[i, start] * self.A_OL[self.GHSu[end][index], i]
        return result

    def find_two_nonchild(self, candidate_m):
        min_diff = float('inf')
        closest_pair_indices = (None, None)
        for i in range(len(candidate_m)):
            for j in range(i + 1, len(candidate_m)):
                if candidate_m[i] * candidate_m[j] < 0:
                    continue
                maximum, minimum = max(np.abs(candidate_m[i]), np.abs(candidate_m[j])), min(np.abs(candidate_m[i]), np.abs(candidate_m[j]))
                ratio = maximum - minimum
                diff = ratio - 1
                if diff < min_diff:
                    min_diff = diff
                    closest_pair_indices = (i, j)
        return closest_pair_indices
        
    def update(self, hop):
        for (i, j) in self.DePairs[hop]:
            assert len(self.GHSu[j]) > 1
            if len(self.GHSu[j]) == 2:
                self.no_edge.add((i, self.GHSu[j][0]))
                self.no_edge.add((i, self.GHSu[j][1]))
                m_ji = (self.mu(i, j, 0) / self.mu(j, j, 0) + self.mu(i, j, 1) / self.mu(j, j, 1)) / 2
                self.M_LL[j, i] = m_ji
            else:
                candidate_m_ji = [self.mu(i, j, k) / self.mu(j, j, k) for k in range(len(self.GHSu[j]))]
                (i_, j_) = self.find_two_nonchild(candidate_m_ji)
                self.no_edge.add((i, self.GHSu[j][i_]))
                self.no_edge.add((i, self.GHSu[j][j_]))
                m_ji = (candidate_m_ji[i_] + candidate_m_ji[j_]) / 2
                self.M_LL[j, i] = m_ji
                for k in range(len(self.GHSu[j])):
                    self.A_OL[self.GHSu[j][k], i] = self.mu(i, j, k) - m_ji * self.mu(j, j, k)

    def result(self):
        M_top = np.hstack((np.eye(self.num_observed), self.M_OL))
        M_bottom = np.hstack((np.zeros([self.num_latent, self.num_observed]), self.M_LL))
        M = np.vstack((M_top, M_bottom))
        A = np.eye(self.num_latent + self.num_observed) - np.linalg.inv(M)
        A[:self.num_observed, :self.num_observed] = np.eye(self.num_observed) - np.linalg.inv(self.M_OO_initial)
        A[np.abs(A) < 0.25] = 0 # threshold
        A[np.abs(A) > 0.25] = 1 # threshold
        # post-process
        for i in range(self.num_latent):
            for j in self.GHSu_full[i]:
                A[j, i+self.num_observed] = 1
        if 1 in self.DePairs_full:
            for (i, j) in self.DePairs_full[1]:
                A[j+self.num_observed, i+self.num_observed] = 1
        for (i, j) in self.no_edge:
            A[j, i+self.num_observed] = 0
        return A
    
    def run(self):
        for i in range(1, self.num_latent):
            self.update(i)
        return self.result()
    
