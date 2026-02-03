# Machine Learning Workflow for Thermal Band Gap Correction of Solid Solutions
Considering the thermal effects of lattice vibrations and electron–phonon coupling on materials’ optoelectronic properties is computationally expensive, especially for anharmonic systems. Conventional first-principles methods are typically limited to small systems containing only a few atoms. Consequently, for complex materials such as perovskite solid solutions, computing thermal corrections to their band gaps has been particularly challenging.

We have developed an accelerated workflow to overcome these limitations and enable the study of larger, more complex systems by combining Graph Neural Networks (GNNs) and Machine Learning Interatomic Potentials (MLIPs). In this workflow, GNNs substitute direct DFT band gap calculations, while MLIPs are used for several key tasks: ionic relaxations, exploration of chemical disorder, phonon dispersion calculations, and molecular dynamics (MD) simulations.

By generating a relatively small dataset, typically hundreds to a few thousand perturbed structures, with computed electronic structures, energies, forces, and stresses (using any exchange–correlation functional, such as PBEsol or HSEsol), we can train the GNN and finetune a MLIP specific to the target solid solution system.

The workflow consist of the following steps:
1. DFT dataset generation.
2. GNN training and MLIP finetuning.
3. Exploration of chemical disorder and verification of vibrational stability with the fine-tuned MLIP.
4. MD simulations with the fine-tuned MLIP at different temperatures for the identified solid solution structures.
5. Band gap prediction using the trained GNN for MD snapshots at each temperature.
6. Thermal band gap renormalization, computed as the average band gap over all snapshots at a given temperature.

## Functionalities

The available functionalities are:
- Solid solution chemical disorder exploration.
- Vibrational stability determination (at harmonic and anharmonic level).
- Training dataset generation.
- MLIP finetuning (MACE).
- Graph generation.
- GNN training.
- Band gap thermal correction.
- Band gap phonon effect.

## Installation

To download the repository, use:

```bash
$ git clone https://github.com/polbeni/ML-thermal-optoelectronics
```

## Requirments

The workflow requires different machine learning and chemoinformatics python packages. It is recommended to create a dedicated python environment and install the different packages by using the `requirements.txt` files that can be found in the `env` dir:
```bash
$ pip install -r requirements.txt
```
It should work in MacOS and GNU/Linux machines without problems.

Please, take into account that when using the scripts, GPU (with CUDA) will be used preferably over CPU. However, if not CUDA detected CPU will be used.

## How to cite

If you use this repository, please cite it as follows:
```
@article{benitez2025physics,
  title={Why Physics Still Matters: Improving Machine Learning Prediction of Material Properties with Phonon-Informed Datasets},
  author={Ben{\'\i}tez, Pol and L{\'o}pez, Cibr{\'a}n and Saucedo, Edgardo and Mizoguchi, Teruyasu and Cazorla, Claudio},
  journal={arXiv preprint arXiv:2511.15222},
  year={2025}
}
```

## Authors

This code and repository are being developed by:
- Pol Benítez Colominas (pol.benitez@upc.edu)
- Claudio Cazorla (claudio.cazorla@upc.edu)
