# Pol Benítez Colominas, January 2026
# The University of Tokyo and Universitat Politècnica de Catalunya

# Generate structures from an intial structure by using phononic and uniform distortions

import os
import shutil
import random

import numpy as np

from pymatgen.io.vasp import Poscar
from pymatgen.core import Element, Lattice

from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.vasp import write_vasp
from phonopy.file_IO import parse_FORCE_CONSTANTS, parse_FORCE_SETS


#### Variables ####
interval_lattice = [9.586, 9.738]                          # Lattice parameter interval between biggest and smallest solid solution

amplitudes_uniform = [0.2, 0.4, 0.6, 0.8]                  # Array with the different amplitudes condsidered for the uniform displacement
temperatures_phonon = [150, 300, 450, 600]                 # Array with the different temperatures condsidered for the phononic displacement

num_uni_for_value = 100                                    # Number of uniform displaced structures for each amplitude value
num_phon_for_value = 100                                   # Number of phononic displaced structures for each temperature value

struc_path = 'final_structures'


#### Functions ####
def generate_solid_solution(poscar, interval_lattice):
    """
    It generates a random solid solution from the original POSCAR file and sets the proportional lattice parameters
    and also returns the Br concentration

    Inputs:
        poscar: pymatgen poscar structure
        interval_lattice: array with the lattice parameter for Ag3SBr and Ag3SI cases
    """

    random_array = [random.choice(['Br', 'I']) for _ in range(8)]

    it_loop = 0
    for i in range(-8, 0):
        poscar[i] = Element(random_array[it_loop])

        it_loop = it_loop + 1

    num_I = random_array.count('I')
    diff_lattice = interval_lattice[1] - interval_lattice[0]
    new_lattice = interval_lattice[0] + (num_I / 8) * diff_lattice

    new_lattice_vals = Lattice.from_parameters(a=new_lattice, b=new_lattice, c=new_lattice, alpha=90, beta=90, gamma=90)

    poscar.lattice = new_lattice_vals

    species_order = ["Ag", "S", "Br", "I"]
    poscar.sort(key=lambda site: species_order.index(site.species_string))

    Br_concentration = (8 - num_I) / 8

    return poscar, Br_concentration


def get_atoms_array(poscar_file):
    """
    Returns an array with the unique atoms in POSCAR

    Inputs:
        poscar_file: path to the POSCAR file (and name of the file)
    """

    poscar = Poscar.from_file(poscar_file)

    element_symbols = poscar.site_symbols

    unique_elements = []
    for element in element_symbols:
        if element not in unique_elements:
            unique_elements.append(element)

    return unique_elements


def generate_random_displacements(
    poscar_file='POSCAR',
    force_file='FORCE_CONSTANTS',
    force_file_type='FORCE_SETS',
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    temperature=100,
    n_snapshots=1,
    fc_symmetry=True,
    primitive_matrix='auto'
):
    """
    Generate random thermal displacements using Phonopy API
    
    Inputs:
        poscar_file : str
            Path to POSCAR file (unit cell)
        force_file : str
            Path to FORCE_CONSTANTS file
        force_file_type : str
            Type of force file: 'FORCE_CONSTANTS' or 'FORCE_SETS'
        supercell_matrix : list or array
            Supercell matrix (DIM parameter)
        temperature : float
            Temperature for random displacements in Kelvin
        n_snapshots : int
            Number of displaced structures to generate
        fc_symmetry : bool
            Apply symmetry to force constants
        number_struc : int
            Reference number of the structure
        primitive_matrix : str or array
            Primitive cell matrix ('auto' or explicit matrix)
    """
    
    # Read unit cell structure
    unitcell, _ = read_crystal_structure(poscar_file, interface_mode='vasp')
    
    # Create Phonopy object
    phonon = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix
    )
    
    # Load forces depending on file type
    if force_file_type.upper() == 'FORCE_CONSTANTS':
        force_constants = parse_FORCE_CONSTANTS(filename=force_file)
        phonon.set_force_constants(force_constants)    
    elif force_file_type.upper() == 'FORCE_SETS':
        force_sets = parse_FORCE_SETS(filename=force_file)
        phonon.dataset = force_sets
        
        # Produce force constants from force sets
        phonon.produce_force_constants()  
    else:
        raise ValueError(f"Unknown force_file_type: {force_file_type}. Use 'FORCE_CONSTANTS' or 'FORCE_SETS'")
    
    # Apply symmetry if requested
    if fc_symmetry:
        phonon.symmetrize_force_constants()
    
    # Generate random displacements
    phonon.generate_displacements(
        number_of_snapshots=n_snapshots,
        random_seed=None,  # Use None for random seed, or set an integer for reproducibility
        temperature=temperature,
        cutoff_frequency=None  # Can set a cutoff to ignore low-frequency modes
    )
    
    # Get the supercells with displacements
    supercells = phonon.supercells_with_displacements
    
    if supercells is None or len(supercells) == 0:
        print("Error: No displacements generated!")
        return
    
    # Save the displaced structure
    output_file = 'POSCAR_med'
    write_vasp(output_file, supercells[0])


def generate_ss_uni(amplitude, outfile_path):
    """
    It generates a POSCAR for a solid solution distorted with uniform noise at amplitude distortion

    Inputs:
        amplitude: maximum amplitude of the uniform distortion
        outfile_path: name of the final POSCAR file
    """

    # Open the initial structure and generate a random solid solution
    poscar = Poscar.from_file('save/SPOSCAR')
    poscar = poscar.structure
    poscar_file, Br_conc = generate_solid_solution(poscar, interval_lattice)

    # Generate a distortion by a uniform distortion
    disp = amplitude
    for site in poscar_file:
        displacement_vector = np.random.uniform(0, disp, 3)

        site.coords = site.coords + displacement_vector

    # Save the final structure
    poscar_file.to(filename=outfile_path, fmt='Poscar')

    return Br_conc


def generate_ss_phon(temp, outfile_path):
    """
    It generates a POSCAR for a solid solution distorted with phonon noise at a given temperature

    Inputs:
        temp: temperature of the phonon distortion
        outfile_path: name of the final POSCAR file
    """

    # Generate the phonon distortion in a supercell and save it as an intermidiate structure
    generate_random_displacements(force_file='save/FORCE_CONSTANTS', poscar_file='save/POSCAR', force_file_type='FORCE_CONSTANTS', temperature=temp)

    # Open the POSCAR file of the saved supercell
    poscar = Poscar.from_file('POSCAR_med')
    poscar = poscar.structure

    # Generate the random solid solution and save it
    poscar_file, Br_conc = generate_solid_solution(poscar, interval_lattice)
    poscar_file.to(filename=outfile_path, fmt='Poscar')

    return Br_conc


#### Main ####
# Create a dir to store the structures
struc_path
if os.path.exists(struc_path):
    shutil.rmtree(struc_path)
os.mkdir(struc_path)

# Create a file to save the structure values
struc_file = open('structures_info.txt', 'w')
struc_file.write('Struc-#      Distortion      Parameter      Br concentration\n')

structure_number = 1

# Generate the uniform displaced structures
for amplitude in amplitudes_uniform:
    for _ in range(num_uni_for_value):
        Br_conc = generate_ss_uni(amplitude, struc_path + '/POSCAR-' + str(structure_number).zfill(4))
        struc_file.write(f'{structure_number}          Uniform      {amplitude}      {Br_conc}\n')

        structure_number = structure_number + 1

# Generate the phononic displaced structures
for temp in temperatures_phonon:
    for _ in range(num_phon_for_value):
        Br_conc = generate_ss_phon(temp, struc_path + '/POSCAR-' + str(structure_number).zfill(4))
        struc_file.write(f'{structure_number}          Phononic     {temp}      {Br_conc}\n')

        structure_number = structure_number + 1

# Close the structure values file
struc_file.close()