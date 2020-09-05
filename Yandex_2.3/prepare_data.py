#!/usr/bin/env python3
# coding: utf-8

# In[52]:


import os
import pickle
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


# In[53]:


structures = pd.read_csv('structures.csv')
structures.head()


# In[54]:


train_data = pd.read_csv('train.csv', index_col='id')
train_data.head()


# In[55]:


test_data = pd.read_csv('test.csv', index_col='id')
test_data.head()


# In[56]:


# Create dictionary for all elements in the dataset
def create_atom_dictionary(structures):
    atom_dictionary = defaultdict(int)
    itera = 0
    for row in structures.itertuples():
        atom_dictionary[row.atom] += 1
        itera += 1
    elements = (atom_dictionary).keys()
    elements_dictionary = {e:i for (i, e) in zip(range(len(elements)), elements)}
    return elements_dictionary 
atom_dict = create_atom_dictionary(structures)
atom_dict


# In[57]:


def relation_compose(index, maindata, structures):
    relation = maindata.iloc[index]
    struc = structures[structures['molecule_name'] == relation.molecule_name]
    print(struc)
    
relation_compose(5, train_data, structures)


# In[58]:


# Functions from https://github.com/masashitsubaki/QuantumGNN_molecules/blob/master/code/preprocess_data.py

def create_atoms(atoms):
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_distances(coordinates):
    distances = [[np.linalg.norm(r_i - r_j) if i != j else 1e6
                 for j, r_j in enumerate(coordinates)]
                 for i, r_i in enumerate(coordinates)]
    return np.array(distances)


# In[81]:


##%%time
Molecules = []
Distances = []
Bounds = []
Types = []
Names = []
Coupling_constants = []

counter = 0

for relation in train_data.itertuples():
    atoms = list(structures[structures['molecule_name'] == relation.molecule_name]['atom'].values)
    atoms = create_atoms(atoms)
    coordinates = structures[structures['molecule_name'] == relation.molecule_name][['x', 'y', 'z']].values
    distances = create_distances(coordinates)
    Molecules.append(atoms)
    Distances.append(distances)
    Molecules.append(atoms)      # повторяем добавление молекул и расстояний дважды, так как можно моделировать 
    Distances.append(distances)  # взаимодействие атомов срава-налево и слева направо, они идентичные
    atom_idexes = [relation.atom_index_0, relation.atom_index_1]
    atom_idexes_inv = [relation.atom_index_1, relation.atom_index_0]
    Bounds.append(atom_idexes)
    Bounds.append(atom_idexes_inv)
    Types.append(relation.type)
    Names.append(relation.molecule_name)
    Coupling_constants.append(relation.scalar_coupling_constant)
    
    #atom_indexes = list(structures[structures['molecule_name'] == relation.molecule_name]['atom_index'].values)   
    #print(atoms, atom_idexes, Coupling_constants, sep='\n')
    
    counter += 1
    if counter > 1000:
        break
        
    if counter % 1000 == 0:
        print(index, train_data.shape[0], sep = ' from ')

# scalar_coupling_constant normalization
Coupling_constants = np.array(Coupling_constants)
mean = np.mean(Coupling_constants)
stdev = np.std(Coupling_constants)
Coupling_constants = (Coupling_constants - mean) / stdev

dir_input = 'dataset/input/'
os.makedirs(dir_input, exist_ok=True)
np.save(dir_input + 'molecules', Molecules)
np.save(dir_input + 'distances', Distances)
np.save(dir_input + 'bounds', Bounds)
np.save(dir_input + 'types', Types)
np.save(dir_input + 'names', Names)
np.save(dir_input + 'coupling_constants', Coupling_constants)
with open(dir_input + 'atom_dict.pickle', 'wb') as f:
    pickle.dump(atom_dict, f)


# In[84]:


def molecule_from_atom(molecule_name):
    atoms = list(structures[structures['molecule_name'] == molecule_name]['atom'].values)
    atoms = create_atoms(atoms)
    return atoms


# In[ ]:




