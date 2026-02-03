# Pol Benítez Colominas, February 2026
# The University of Tokyo and Universitat Politècnica de Catalunya

# Get band gaps from DFT calculations and generate materials.txt file necessary for graph generation


def electronic_bandGap(file_name):
    """
    This functions uses DOSCAR file generated in VASP simulations and returns the Fermi energy
    the band gap, and the energies of the band gap (respect the exchange-correlation functional
    used).

    file_name: path of the DOSCAR file
    """
    
    file = open(file_name, "r")

    for x in range(6):
        actual_string = file.readline()
        if x == 5:
            fermiEnergy = float(actual_string.split()[3])

    file.close()

    file = open(file_name, "r")

    for x in range(6):
        file.readline()

    for x in file:
        actual_string = x

        if (float(actual_string.split()[0]) <= fermiEnergy+0.1) and (float(actual_string.split()[0]) >= fermiEnergy-0.1):
            density_bandGap = float(actual_string.split()[2])

            break

    file.close()

    file = open(file_name, "r")

    for x in range(6):
        file.readline()

    for x in file:
        actual_string = x

        if float(actual_string.split()[2]) == density_bandGap:
            minEnergy = float(actual_string.split()[0])

            break   

    for x in file:
        actual_string = x

        if float(actual_string.split()[2]) != density_bandGap:
            maxEnergy = float(actual_string.split()[0])

            break 
    bandGap = maxEnergy - minEnergy

    file.close()
    
    return fermiEnergy, minEnergy, maxEnergy, bandGap

struc_file = open('structures_info.txt', 'r')
struc_file.readline()
final_file = open('results.txt', 'w')
final_file.write('Struc-#      Distortion      Parameter      Band gap (eV)\n')


materials = open('materials.txt', 'w')
materials.write('material id       band gap (eV)       type\n')

for struc in range (1120):
    _, _, _, bg = electronic_bandGap(f'calculations/struc-{str(struc + 1).zfill(4)}/DOSCAR')
    
    line = struc_file.readline()
    final_file.write(f'{line.split()[0]}      {line.split()[1]}      {line.split()[2]}      {bg}\n')

    materials.write(f'struc-{line.split()[0].zfill(4)}       {bg}       {line.split()[1]}\n')

struc_file.close()
final_file.close()

materials.close()