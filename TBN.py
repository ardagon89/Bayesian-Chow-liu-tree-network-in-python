#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import copy

def loadfile(filename1, filename2=None):
    ds1 = np.loadtxt(filename1, delimiter=",", dtype=int)
    if filename2:
        ds2 = np.loadtxt(filename1, delimiter=",", dtype=int)
        ds = np.vstack((ds1, ds2))
    else:
        ds = ds1
    return ds, ds.shape[0], ds.shape[1]

def prob_matrix(ds, m, n):
    prob_xy = np.zeros((n, n, 4))
    for i in range(n):
        subds = ds[ds[:, i] == 0]
        for j in range(n):
            if prob_xy[i, j, 0] == 0:
                prob_xy[i, j, 0] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 0] == 0:
                prob_xy[j, i, 0] = prob_xy[i, j, 0]
            if prob_xy[i, j, 1] == 0:
                prob_xy[i, j, 1] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 2] == 0:
                prob_xy[j, i, 2] = prob_xy[i, j, 1]
            
        subds = ds[ds[:, i] == 1]
        for j in range(n):
            if prob_xy[i, j, 2] == 0:
                prob_xy[i, j, 2] = (subds[subds[:, j] == 0].shape[0]+1)/(m+4)
            if prob_xy[j, i, 1] == 0:
                prob_xy[j, i, 1] = prob_xy[i, j, 2]
            if prob_xy[i, j, 3] == 0:
                prob_xy[i, j, 3] = (subds[subds[:, j] == 1].shape[0]+1)/(m+4)
            if prob_xy[j, i, 3] == 0:
                prob_xy[j, i, 3] = prob_xy[i, j, 3]
    return prob_xy

def mutual_info(prob_xy, n):
    I_xy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                I_xy[i, j] = prob_xy[i, j, 0]*np.log(prob_xy[i, j, 0]/(prob_xy[i, i, 0]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 1]*np.log(prob_xy[i, j, 1]/(prob_xy[i, i, 0]*prob_xy[j, j, 3]))                 + prob_xy[i, j, 2]*np.log(prob_xy[i, j, 2]/(prob_xy[i, i, 3]*prob_xy[j, j, 0]))                 + prob_xy[i, j, 3]*np.log(prob_xy[i, j, 3]/(prob_xy[i, i, 3]*prob_xy[j, j, 3]))
    return I_xy

def draw_tree(edge_wts, prnt = False):
    edge_wts_cp = copy.deepcopy(edge_wts)
    edges = [np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)]
    visited = [[edges[-1][0],edges[-1][1]]]
    edge_wts_cp[edges[-1]] = 0
    while(len(edges) < edge_wts.shape[0]-1):
        i = j = -1
        edge = np.unravel_index(np.argmax(edge_wts_cp), edge_wts_cp.shape)
        for bag in range(len(visited)):
            if edge[0] in visited[bag]:
                i = bag
            if edge[1] in visited[bag]:
                j = bag
        if i == -1 and j != -1:
            edges.append(edge)
            visited[j].append(edge[0])
        elif i != -1 and j == -1:
            edges.append(edge)
            visited[i].append(edge[1])
        elif i == -1 and j == -1:
            edges.append(edge)
            visited.append([edge[0], edge[1]])
        elif i != -1 and j != -1 and i != j:
            edges.append(edge)
            visited[i] += visited[j]
            visited.remove(visited[j])
        elif i == j != -1:
            pass
        else:
            print("Discarded in else", edge)
        edge_wts_cp[edge] = 0
    
    new_tree = []
    make_tree(edges, new_tree, edges[0][0])
    
    return new_tree

def count_matrix(ds, tree, cols):
    count_xy = np.zeros((len(tree), cols))
    for idx, node in enumerate(tree):
        i, j = node
        count_xy[idx] = [ds[(ds[:, i]==0) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==0) & (ds[:, j]==1)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==0)].shape[0], ds[(ds[:, i]==1) & (ds[:, j]==1)].shape[0]]
    return count_xy

def make_tree(ls, new_tree, parent):
    for node in [item for item in ls if parent in item]:
        if node[0] == parent:
            new_tree.append(node)
            ls.remove(node)
            make_tree(ls, new_tree, node[1])
        else:
            new_tree.append((node[1],node[0]))
            ls.remove(node)
            make_tree(ls, new_tree, node[0])
            
if __name__ == "__main__":
    import sys

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    if len(sys.argv) != 4:
        print("Usage:python TBN.py <training-dataset> <validation-dataset> <testing-dataset>")
    else:
        ds, m, n = loadfile(sys.argv[1], sys.argv[2])   
        prob_xy = prob_matrix(ds, m, n)
        I_xy = mutual_info(prob_xy, n)
        tree = draw_tree(I_xy, False)
        tree = [(tree[0][0], tree[0][0])] + tree
        cond_prob = np.zeros((len(tree), prob_xy.shape[2]))
        for idx, node in enumerate(tree):
            if node[0] == node[1]:
                cond_prob[idx] = np.log(prob_xy[node[0], node[1],:])
            else:
                cond_prob[idx] = np.log(np.hstack(((prob_xy[node[0], node[1],:2]/prob_xy[node[0], node[0], 0]),(prob_xy[node[0], node[1],2:]/prob_xy[node[0], node[0], 3]))))
        ts = np.loadtxt(sys.argv[3], delimiter=",", dtype=int)
        count_xy = count_matrix(ts, tree, prob_xy.shape[2])
        print("Avg. Log Likelihood:", np.sum(count_xy*cond_prob)/ts.shape[0])

