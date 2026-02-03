# Pol Benítez Colominas, February 2025
# Universitat Politècnica de Catalunya

# Determines the FORCE_SETS for a structure


import subprocess
import yaml
import warnings

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.optimize import BFGS, FIRE
from ase.md import Langevin
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp, write_vasp_xdatcar
from ase.constraints import ExpCellFilter, FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from mace.calculators import mace_mp
import torch
torch.serialization.add_safe_globals([slice])

warnings.simplefilter('ignore')


### Variables definition
poscar_name = 'original_POSCAR'
dimension_supercell = '2 2 2'
device = 'cuda'
model = '../CAP_finetuned_model_stagetwo_compiled.model'
temperature = 200


### Read the original structure, relax it and save it with the name POSCAR
crystal_structure = Structure.from_file(poscar_name) # read the initial structure
ase_adaptor = AseAtomsAdaptor()
atoms = ase_adaptor.get_atoms(crystal_structure)

atoms.calc = mace_mp(model=model, device=device) # load the MACE calculator

#atoms_filter = ExpCellFilter(atoms) # allow lattice parameters to change

# Fix the symmetry (just change the lattice parameters)
constraint = FixAtoms(indices=[atom.index for atom in atoms]) # fix the atoms positions in the relaxation
atoms.set_constraint(constraint)

mask = [1, 1, 1, 0, 0, 0]
atoms_filter = ExpCellFilter(atoms, mask=mask) # allow lattice parameters to change

dyn = BFGS(atoms_filter) # relax the structure
dyn.run(fmax=0.05, steps=200)

write_vasp('POSCAR', atoms=atoms, direct=True) # save the relaxed structure


### Use phonopy to generate the distorted structures
command = f'phonopy -d --dim="{dimension_supercell}"'  
result = subprocess.run(command, shell=True, capture_output=True, text=True)

### Save the number of atoms in the supercell and the number of distorted structures
with open('phonopy_disp.yaml', 'r') as f:
    data = yaml.safe_load(f)

atom_info = data['primitive_cell']['points']
num_atoms_cell = len(atom_info)
num_atoms = num_atoms_cell # in the supercell
for component in range(len(dimension_supercell.split())): # diagonal supercell assumed
    num_atoms = num_atoms*int(dimension_supercell.split()[component])

num_distorted_struc = len(data['displacements'])


### Compute the forces with MACE for all the distorted structures
forces_array = []
for dist in range(num_distorted_struc):
    crystal_structure = Structure.from_file(f'POSCAR-{str(dist + 1).zfill(3)}') # read the distorted structure
    ase_adaptor = AseAtomsAdaptor()
    atoms = ase_adaptor.get_atoms(crystal_structure)

    atoms.calc = mace_mp(model=model, device=device) # load mace calculator 

    forces = atoms.get_forces() # get the forces
    forces_array.append(forces)


### Generate the FORCE_SETS file
force_sets = open('FORCE_SETS', 'w')

force_sets.write(f'{num_atoms}\n')
force_sets.write(f'{num_distorted_struc}\n')
force_sets.write('\n')

for dist in range(num_distorted_struc):
    atom_ind = data['displacements'][dist]['atom']
    force_vec = data['displacements'][dist]['displacement']

    force_sets.write(f'{atom_ind}\n')
    force_sets.write(f'  {force_vec[0]}   {force_vec[1]}   {force_vec[2]}\n')
    for atom in range(num_atoms):
        force_sets.write(f'     {forces_array[dist][atom][0]}   {forces_array[dist][atom][1]}   {forces_array[dist][atom][2]}\n')
    force_sets.write('\n')

force_sets.close()