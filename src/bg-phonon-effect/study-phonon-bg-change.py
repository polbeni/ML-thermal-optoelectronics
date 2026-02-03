# Pol Benítez Colominas, Desember 2025
# The University of Tokyo and Universitat Politècnica de Catalunya

# Computes the maximum displacement at T for a given gamma phonon mode, and computes
# the band gap change

################################# LIBRARIES ###############################
import os
import csv
import shutil
from datetime import datetime
import json
import math
import glob
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

from pymatgen.io.vasp import Poscar

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data

from models_cgcnn import *
###########################################################################



################################ PARAMETERS ###############################
hidden = 64                                                             # Number of hidden channels in the convolutional layers
dropout = 0.4                                                           # Fraction of elements to dropout
seed_model_torch = 12345                                                # Seed for the model

model_path = 'trained_model'                                            # Path or name of the final trained model
inputs_path = '../../../anharmonic/'                                    #'data/'
outputs_dir = 'outputs_file/'                                           # Path to dir where outputs are saved

compounds = ['Ag3SI', 'Ag24S8Br1I7', 'Ag24S8Br2I6', 'Ag24S8Br3I5', 'Ag24S8Br4I4',
             'Ag24S8Br5I3', 'Ag24S8Br6I2', 'Ag24S8Br7I1','Ag3SBr']
temperature_list = [300, 600]

### Physical constants
reduced_planck_ct = 6.5821e-16 # eV·s
boltzmann_ct = 8.6173e-5 # eV·K^-1
###########################################################################



################################ FUNCTIONS ################################
def bose_einstein(temperature, phonon_energy):
    """
    Computes the Bose-Einstein distribution of a phonon population with a given energy

    Inputs:
        temperature -> temperature (in K)
        phonon_energy -> energy of the given phonon (in s^-1 (Hz))
    """

    distribution = 1 / (np.exp((reduced_planck_ct * 2 * np.pi * phonon_energy) / (boltzmann_ct * temperature)) - 1) 

    return distribution


def phonon_amplitude(temperature, phonon_energy, atom_mass):
    """
    Computes the phonon amplitude for a given temperature and phonon mode

    Inputs:
        temperature -> temperature (in K)
        phonon_energy -> energy of the given phonon (in s^-1 (Hz))
        atom_mass -> atomic mass of the given atom
    """

    dist = bose_einstein(temperature, phonon_energy)

    h_si = reduced_planck_ct * 6.24e-18 # in J·s
    mass = atom_mass * 1.660539e-27 # in Kg

    amplitude = np.sqrt((h_si / (2 * mass * 2 * np.pi * phonon_energy)) * (1 + 2*dist))

    amplitude = amplitude * 1e10 # in angstrom

    return amplitude


def get_phonopy(path_file):
    """
    It extracts the number of atoms and the eigenvalues and eigenvectors at gamma-point,
    from band.yaml phonopy output file. It also extracts the atomic masses of the atoms
    IMPORTANT: it assumes that gamma point is the first q-pont in the band.yaml file
    change the k-path to start at gamma otherwise

    Inputs:
        path_file -> path to the band.yaml file
    """

    with open(path_file, "r") as file:
        data = yaml.safe_load(file)

    number_phonons = data['natom'] * 3

    number_atoms = data['natom']
    number_q = 1
    eigenvalues = []
    eigenvectors = []
    mass_list = []

    for q_val in range(number_q):
        eigenvalue = []
        for mode in range(number_phonons):
            eigenvalue.append(data['phonon'][q_val]['band'][mode]['frequency'] * 1e12) # to express it in Hz (not in THz)

        eigenvalues.append(eigenvalue)
    
    for q_val in range(number_q):
        eigenvector_mu = []
        for mode in range(number_phonons):
            eigenvector_atom = []
            for atom in range(number_phonons // 3):
                vec_x = data['phonon'][q_val]['band'][mode]['eigenvector'][atom][0][0]
                vec_y = data['phonon'][q_val]['band'][mode]['eigenvector'][atom][1][0]
                vec_z = data['phonon'][q_val]['band'][mode]['eigenvector'][atom][2][0]

                eigenvector_atom.append([vec_x, vec_y, vec_z])

            eigenvector_mu.append(eigenvector_atom)
        
        eigenvectors.append(eigenvector_mu)

    for atom in range(number_phonons // 3):
        mass_list.append(data['points'][atom]['mass'])

    return number_atoms, eigenvalues, eigenvectors, mass_list


def compute_norm(eigenvectors, num_atoms, acoustic):
    """
    It computes the norm of a given set of eigenvectors for all the phonon modes
    neglecting the acoustics since we are at gamma

    Inputs:
        eigenvectors -> phonon eigenvectors at the gamma-point
        num_atoms -> number of atoms in the structure
        acoustic -> boolean indicating if the list of eigenvectors contain acoustics (True) or not (False)
    """

    phonon_modes = (num_atoms * 3) - 3

    norm = 0

    for mode in range(phonon_modes):
        for atom in range(num_atoms):
            for coord in range(3):
                if acoustic == True:
                    norm = norm + ((eigenvectors[mode + 3][atom][coord])**2)
                elif acoustic == False:
                    norm = norm + ((eigenvectors[mode][atom][coord])**2)

    norm = np.sqrt(norm)

    return norm


def normalize_eigenvectors(eigenvectors, num_atoms):
    """
    Computes the normalization constant and returns renormalized eigenvectors

    Inputs:
        eigenvectors -> phonon eigenvectors of a given phonon mode at the gamma-point
        num_atoms -> number of atoms in the structure
    """

    normalization = 1 / compute_norm(eigenvectors, num_atoms, True)

    phonon_modes = (num_atoms * 3) - 3

    new_vectors = []

    for mode in range(phonon_modes):
        mode_vectors = []
        for atom in range(num_atoms):
            atom_coords = [(eigenvectors[mode + 3][atom][0]) * normalization,
                           (eigenvectors[mode + 3][atom][1]) * normalization,
                           (eigenvectors[mode + 3][atom][2]) * normalization]
                
            mode_vectors.append(atom_coords)
        
        new_vectors.append(mode_vectors)

    return new_vectors


def disp_mode(eigenvectors, num_atoms, temperature, phonon_energy, masses_list):
    """
    It generates the displacement vector for a given phonon mode and a given temperature
    It just considers the eigenvectors at gamma-point

    Inputs:
        eigenvectors -> phonon eigenvectors for the desired mode (normalized)
        num_atoms -> number of atoms in the structure
        temperature -> temperature (in K)
        phonon_energy -> energy of the given phonon (in s^-1 (Hz))
        masses_list -> list with the atomic masses of the atoms
    """

    disp_vect = []

    for atom in range(num_atoms):
        amplitude = phonon_amplitude(temperature, phonon_energy, masses_list[atom])

        atom_vect = [eigenvectors[atom][0] * amplitude,
                     eigenvectors[atom][1] * amplitude,
                     eigenvectors[atom][2] * amplitude]
        
        disp_vect.append(atom_vect)

    return disp_vect


def new_POSCAR(old_path, disp_vec, new_path):
    """
    Applies the find vector displacement to a given POSCAR

    Inputs:
        old_path -> path to the old POSCAR file (IN CARTESIAN COORDINATES!!!!!)
        disp_vec -> generated displacement vector
        new_path -> path to the new distorted structure
    """
    old = open(old_path, 'r')
    new = open(new_path, 'w')

    for _ in range(7):
        line = old.readline()
        new.write(line)

    num_atoms = 0
    for it in range(len(line.split())):
        num_atoms = num_atoms + int(line.split()[it])

    line = old.readline()
    new.write(line)

    for atom in range(num_atoms):
        line = old.readline()

        pos_x = float(line.split()[0]) + disp_vec[atom][0]
        pos_y = float(line.split()[1]) + disp_vec[atom][1]
        pos_z = float(line.split()[2]) + disp_vec[atom][2]

        new.write(f'     {pos_x:.9f}     {pos_y:.9f}     {pos_z:.9f}\n')

    old.close()
    new.close()


def reduce_disp(disp_vect_old, num_atoms, red_amount):
    """
    It reduces the amplitude a desired amount (example, we want displacements 50% of the max amplitude)

    Inputs:
        disp_vect_old -> displacemnt vector for all the atoms
        num_atoms -> number of atoms in the structure
        red_amount -> amoun to reduce (between 0 and 1, i.e., 0.5->50%)
    """

    disp_vect = []

    for atom in range(num_atoms):

        atom_vect = [disp_vect_old[atom][0] * red_amount,
                     disp_vect_old[atom][1] * red_amount,
                     disp_vect_old[atom][2] * red_amount]
        
        disp_vect.append(atom_vect)

    return disp_vect


def model_prediction(model, graph, norm_ct):
    """
    Returns the model prediction for a given graph

    Inputs:
        model: model to use in the prediction
        graph: graph structure to make prediction
        norm_ct: array with normalization constants with the format [min, max]
    """

    graph.x = graph.x.to(device).float()
    graph.edge_index = graph.edge_index.to(device).long()
    graph.edge_attr = graph.edge_attr.to(device).float()
    graph.y = graph.y.to(device).float()

    graph = graph.to(device)

    model = model.to(device).float()

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).to(device)

    # Multiply the predicted value by the normalization constant
    prediction_desnorm = prediction[0][0]*(norm_ct[1] - norm_ct[0]) + norm_ct[0]

    return prediction_desnorm


def get_nodes(struct):
    """
    This function recives a unit cell structure (pymatgen structure object) and returns the node list
    
    Inputs:
        struct: structure object
    Outputs:
        node_list: list of nodes with their features
    """
    # get the number of atoms in the unit cell
    atoms_number = struct.num_sites

    # create the node list
    node_list = [None]*atoms_number

    # save nodes in node list with the features of the given atom
    for atom in range(atoms_number):
        node_list[atom] = atoms_dict[(struct.sites[atom]).species_string]

    return node_list


def get_edges(struct, lim_dist):
    """
    This function recives a unit cell structure (pymatgen structure object) and returns the adjacency list, i.e., 
    a list of pairs of nodes that are closer than the desired distance, and the edges list, i.e, a list with the 
    features of each element of the adjecent list, here the euclidean distance. In order to do this a proper 
    supercell is created

    Inputs:
        struct: structure object
        lim_dist: maximum distance to consider edges
    Outputs:
        adjacency_list: list of pairs of nodes that verify to be closer than lim_dist
        edge_list: list of features for all the edges in adjacency list
    """
    # create adjacency and edge lists
    adjacency_list = []
    edge_list = []

    # get the lattice parameters and the smallest parameter
    lattice_parameters = struct.lattice.abc
    min_parameter = min(lattice_parameters)

    # find the minimum supercell (with central cell, n=3,5,...) to consider all the connections for the given limit distance
    n_supercell = 3
    param_supercell = min_parameter
    while param_supercell < lim_dist:
        n_supercell = n_supercell + 2
        param_supercell = min_parameter*(n_supercell - 2) 

    # number for the atoms in the centered cell after creating a supercell
    atoms_centered_cell = math.trunc((n_supercell**3)/2) + 1

    # get the number of atoms in the unit cell
    atoms_number = struct.num_sites

    # create the supercell
    scaling_matrix = [[n_supercell, 0, 0], [0, n_supercell, 0], [0, 0, n_supercell]]
    supercell = struct.make_supercell(scaling_matrix)

    # get the number of atoms in the supercell
    atoms_supercell_number = supercell.num_sites

    # check if there is a connection between two atoms (count just one of the directions, example: just (0,2), not (0,2) and (2,0))
    for atom in range(atoms_number):
        for atom_super in range(atoms_supercell_number - atom*(n_supercell**3)):
            a_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[0]
            b_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[1]
            c_cell = (supercell.sites[atom*(n_supercell**3) + atoms_centered_cell]).coords[2]

            a_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[0]
            b_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[1]
            c_super = (supercell.sites[atom_super + atom*(n_supercell**3)]).coords[2]

            euclidean_distance = ((a_cell - a_super)**2 + (b_cell - b_super)**2 + (c_cell - c_super)**2)**0.5

            if (euclidean_distance <= lim_dist) and (euclidean_distance > 1e-5): 
                edge_pair = [atom, math.trunc((atom_super + atom*(n_supercell**3))/(n_supercell**3))]

                edge_feature = [euclidean_distance]

                # chech if it is self-loop, if not save twice to be undirected
                if edge_pair[0] != edge_pair[1]:
                    adjacency_list.append(edge_pair)
                    edge_pair2 = [edge_pair[1], edge_pair[0]]
                    adjacency_list.append(edge_pair2)

                    edge_list.append(edge_feature)
                    edge_list.append(edge_feature)
                else:
                    adjacency_list.append(edge_pair)
                    edge_list.append(edge_feature)

    return adjacency_list, edge_list
###########################################################################



################################### MAIN ##################################
# Check if a GPU (CUDA) is available 
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("GPU not available. Using CPU.")

# Create a folder for the results
if os.path.exists(outputs_dir):
    shutil.rmtree(outputs_dir)
os.mkdir(outputs_dir)


# Import the normalization constants (inputs are taken by hand from normalized_parameters.txt)
max_node = torch.Tensor([ 88.0000,   3.9800, 244.0000, 298.0000])
min_node = torch.Tensor([ 0.0000,  0.7900,  1.0080, 42.0000])
max_edge = 5.5
min_edge = 0.7115

with open('output_normalization.txt', 'r') as file:
    file.readline()
    line = file.readline()

    max_output = float(line.split()[0])
    min_output = float(line.split()[1])


# Create the model, import the trained model and put it in the evaluation mode
model = model5(features_channels=4, hidden_channels=hidden, seed_model=seed_model_torch, dropout=dropout)
model = model.to(device)

#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

# Generate the graphs and predict the band gap for the generated POSCARs
with open('atoms_dict.json', 'r') as json_file:
    atoms_dict = json.load(json_file)

edge_radius = 5.5 # define the maximum longitude to consider edge connections (angstroms)




# specify data to study


for compound in compounds:
    if os.path.exists(outputs_dir + compound):
        shutil.rmtree(outputs_dir + compound)
    os.mkdir(outputs_dir + compound)

    results_file = open(outputs_dir + compound + '/results.txt', 'w')
    results_file.write('Phonon mode             Temperature (K)             Band gap (eV)                        Phonon energy (THz)\n')

    if compound == 'Ag3SI' or compound == 'Ag3SBr':
        inputs_path = 'data/'
    else:
        inputs_path = '../../../anharmonic/'#'data/'

    # Get the data at gamma-point
    num_atoms, values, vectors, masses = get_phonopy(inputs_path + compound + '/band.yaml')

    # Renormalize the eigenvectors
    norm_vectors = normalize_eigenvectors(vectors[0], num_atoms)

    for temp in temperature_list:
        disp_final = []
        for x in range((num_atoms - 1) * 3):
                disp_final.append(disp_mode(norm_vectors[x], num_atoms, temp, values[0][x + 3], masses))

        for it_mode in range(len(disp_final)):
            # Generate POSCAR with the displacement
            name_POSCAR = outputs_dir + compound + '/' + '/POSCAR-' + str(it_mode + 4).zfill(2) + '-' + str(temp) + '.vasp'
            disp_vector = reduce_disp(disp_final[it_mode], num_atoms, 1)
            if compound == 'Ag3SI' or compound == 'Ag3SBr':
                new_POSCAR(inputs_path + compound + '/POSCAR.vasp', disp_vector, name_POSCAR)
            else:
                new_POSCAR(inputs_path + compound + '/POSCAR', disp_vector, name_POSCAR)
            

            # Open the POSCAR and compute the band gap
            poscar = Poscar.from_file(name_POSCAR)
                
            structure_object = poscar.structure 

            nodes = get_nodes(structure_object)

            adjacency, edges = get_edges(structure_object, edge_radius)

            nodes_torch = torch.tensor(nodes)
            adjacency_torch = torch.tensor(adjacency)
            edges_torch = torch.tensor(edges)

            nodes_torch = (nodes_torch - min_node)/(max_node - min_node)
            edges_torch = (edges_torch - min_edge)/(max_edge - min_edge)

            graph = Data(x=nodes_torch, edge_index=adjacency_torch.t().contiguous(), edge_attr=edges_torch)

            graph.x = graph.x.to(device).float()
            graph.edge_index = graph.edge_index.to(device).long()
            graph.edge_attr = graph.edge_attr.to(device).float()
            graph = graph.to(device)

            batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)

            model = model.to(device).float()

            # Make prediction
            with torch.no_grad():  # Disable gradient calculation for inference
                prediction = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).to(device)

            # Multiply the predicted value by the normalization constant
            predicted_value = prediction[0][0]*(max_output - min_output) + min_output

            print('Band gap predicted!')

            results_file.write(f'{it_mode + 4}                           {str(temp)}                    {predicted_value}                     {1e-12*values[0][it_mode + 3]}\n')
    
    results_file.close()
###########################################################################