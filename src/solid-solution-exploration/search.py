# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Computes all the possible solid solutions permuations and relaxes the structures
# and compute their energies with MLIPs (MACE-based)

import os
import shutil
import itertools
from copy import deepcopy

import warnings

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import BFGS, FIRE
from ase.io.trajectory           import Trajectory
from ase.io.vasp                 import write_vasp
from ase.constraints import ExpCellFilter, FixAtoms

from mace.calculators import mace_mp
import torch
torch.serialization.add_safe_globals([slice])

warnings.simplefilter('ignore')

######################## MACE stuff ########################
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Using GPU.")
else:
    device = 'cpu'
    print("GPU not available. Using CPU.")

model = 'CAP_finetuned_model_stagetwo_compiled.model'    # name of the model
fmax = 0.05        # max force allowed
smax = 200         # max number of steps for the relaxation
############################################################


######################### Functions ########################
def find_all_permutations(elements):
    """
    Find all the possible permutations for a list of elements and returns all the possible lists

    Inputs:
        elements: array with the elements
    """

    unique_permutations = set(itertools.permutations(elements))

    unique_permutations_list = [list(p) for p in unique_permutations]

    return unique_permutations_list

def get_atoms_list(stucture, atom_type):
    """
    Returns two list of atoms for a given structure, one with all the atoms != to atom, and another all
    the atoms == to atom

    Inputs:
        structure: unit cell structure in pymatgen format
        atom_type: the type of atom    
    """

    atom_list = [str(site.specie) for site in structure.sites]

    without_atom = []
    with_atom = []

    for atom in atom_list:
        if atom_type == atom:
            with_atom.append(atom)
        else:
            without_atom.append(atom)

    return without_atom, with_atom

def sort_atoms(species, coords, atom1, atom2):
    """
    It recives a list of species and coords and returns both lists ordered first by atom1 and then atom2
    """

    new_species = []
    new_coords = []

    for x in range(len(species)):
        if (species[x] != atom1) and (species[x] != atom2):
            new_species.append(species[x])
            new_coords.append(coords[x])

    for x in range(len(species)):
        if species[x] == atom1:
            new_species.append(species[x])
            new_coords.append(coords[x])

    for x in range(len(species)):
        if species[x] == atom2:
            new_species.append(species[x])
            new_coords.append(coords[x])

    return new_species, new_coords
############################################################


########################### Main ###########################
# Define a list with the 7 different solid solutions
solid_solutions = [['Br', 'I', 'I', 'I', 'I', 'I', 'I', 'I'], ['Br', 'Br', 'I', 'I', 'I', 'I', 'I', 'I'],
                   ['Br', 'Br', 'Br', 'I', 'I', 'I', 'I', 'I'], ['Br', 'Br', 'Br', 'Br', 'I', 'I', 'I', 'I'],
                   ['Br', 'Br', 'Br', 'Br', 'Br', 'I', 'I', 'I'], ['Br', 'Br', 'Br', 'Br', 'Br', 'Br', 'I', 'I'],
                   ['Br', 'Br', 'Br', 'Br', 'Br', 'Br', 'Br', 'I']]
ss_names_list = ['Ag24S8Br1I7', 'Ag24S8Br2I6', 'Ag24S8Br3I5', 'Ag24S8Br4I4', 'Ag24S8Br5I3', 'Ag24S8Br6I2', 'Ag24S8Br7I1']

# Open the POSCAR
structure = Poscar.from_file('POSCAR').structure

list_without, _ = get_atoms_list(structure, 'Br')

# Run for all the possible permuations
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')

it_ss = 0
for ss in solid_solutions:
    if os.path.exists('results/' + ss_names_list[it_ss]):
        shutil.rmtree('results/' + ss_names_list[it_ss])
    os.mkdir('results/' + ss_names_list[it_ss])

    permutations = find_all_permutations(ss)

    list_permutations_values = []
    
    it_permutation = 1
    for permutation in permutations:
        solid_solution = deepcopy(structure)

        solid_solution_atoms = list_without + permutation

        new_species, new_coords = sort_atoms(solid_solution_atoms, solid_solution.frac_coords, 'Br', 'I')

        solid_solution_structure = Structure(
            lattice=solid_solution.lattice,
            species=new_species,
            coords=new_coords,
            coords_are_cartesian=False
        )

        # Create ase object and perform md with MACE
        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(solid_solution_structure)

        atoms.calc = mace_mp(model=model, device=device)

        # Fix the symmetry (just change the lattice parameters)
        constraint = FixAtoms(indices=[atom.index for atom in atoms]) # fix the atoms positions in the relaxation
        atoms.set_constraint(constraint)

        mask = [1, 1, 1, 0, 0, 0]
        atoms_filter = ExpCellFilter(atoms, mask=mask) # allow lattice parameters to change

        # Perform the relaxation
        dyn = BFGS(atoms_filter) # relax the structure
        dyn.run(fmax=fmax, steps=smax)

        # Save the relaxed solid solution and fix the POSCAR format
        write_vasp('POSCAR-after-relax', atoms=atoms, direct=True)

        old_POSCAR = open('POSCAR-after-relax', 'r') # fix the POSCAR structure format
        new_POSCAR = open('results/' + ss_names_list[it_ss] + '/POSCAR-' + str(it_permutation).zfill(3), 'w')

        for x in range(7):
            line = old_POSCAR.readline()
            new_POSCAR.write(line)

            if x == 6: # read the number of atoms
                num_atoms = 0
                for atom in range(len(line.split())):
                    num_atoms = num_atoms + int(line.split()[atom])

        old_POSCAR.readline()

        line = old_POSCAR.readline()
        new_POSCAR.write(line)

        for x in range(num_atoms):
            line = old_POSCAR.readline()
            new_POSCAR.write(f'   {float(line.split()[0])}   {float(line.split()[1])}   {float(line.split()[2])}\n')

        old_POSCAR.close()
        new_POSCAR.close()

        # Determine the energy of the structure and save the value
        energy = atoms.get_potential_energy()

        list_permutations_values.append([it_permutation, energy / len(solid_solution_atoms)])

        it_permutation = it_permutation + 1


    sorted_list = sorted(list_permutations_values, key=lambda x: x[1], reverse=False)

    with open('results/' + ss_names_list[it_ss] + '/results_file.txt', 'w') as file:
        file.write('Structure       E(eV/atom)\n')
        
        for ss in range(len(sorted_list)):
            file.write(f'{sorted_list[ss][0]}       {sorted_list[ss][1]}\n')

    it_ss = it_ss + 1
############################################################
