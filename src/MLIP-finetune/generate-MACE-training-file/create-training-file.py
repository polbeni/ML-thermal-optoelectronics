# Pol Benítez Colominas, October 2025
# The University of Tokyo and Universitat Politècnica de Catalunya

# Generates the training file for MACE from a set of DFT calculations

import os
import glob

from ase.io import read, write

# Find all the calculations in the specified folder (here the results are inside folder with the name struc-#)
root_dir = '/Users/pol/work/upc/ml-gap-prediction/database-CAP/database-uniform/results-PBEsol/solid-solutions'
pattern = os.path.join(root_dir, 'struc-*', 'vasprun.xml')
vasprun_files = sorted(glob.glob(pattern))

# Define the name for your output file
out_file = 'dataset_training.extxyz'

# Read the files and save all the results for valid calculations
all_structures = []

for vasp_file in vasprun_files:
    folder = os.path.dirname(vasp_file)

    try:
        # Read the final structure and the related data
        atoms = read(vasp_file, format='vasp-xml', index=-1)
        atoms.info['config_name'] = os.path.basename(folder)

        # Save enery with the new key name
        if 'energy' in atoms.info:
            energy = atoms.info['energy']
        elif hasattr(atoms, 'calc') and hasattr(atoms.calc, 'results'):
            energy = atoms.calc.results.get('energy', None)

        if hasattr(atoms, 'calc') and hasattr(atoms.calc, 'results'):
            free_energy = atoms.calc.results.get('free_energy', None)

        if energy is not None:
            atoms.info['REF_energy'] = float(energy)
        if free_energy is not None:
            atoms.info['REF_free_energy'] = float(free_energy)

        # Save stress with the new key name
        stress = atoms.get_stress(voigt=False)
        atoms.info['REF_stress'] = ' '.join(map(str, stress.flatten()))

        # Save forces with the new key name
        atoms.arrays['REF_forces'] = atoms.get_forces()

        # Clear calculator results to prevent ASE from rewriting energy/stress/forces
        if hasattr(atoms, 'calc') and hasattr(atoms.calc, 'results'):
            atoms.calc.results.clear()
        atoms.calc = None  # detach calculator entirely

        # Delete default fields from info/arrays
        for key in ['energy', 'stress']:
            atoms.info.pop(key, None)
        atoms.arrays.pop('forces', None)

        # Save the atoms object in the all_structures array
        all_structures.append(atoms)

        print(f'Data saved for {vasp_file}')
    
    except Exception as e:
        print(f'Error reading {vasp_file}: {e}')

# Write all collected structures to one extended XYZ
if all_structures:
    write(out_file, all_structures, format='extxyz')
    print(f'Wrote {len(all_structures)} structures to {out_file}')
else:
    print('No valid structures found!')