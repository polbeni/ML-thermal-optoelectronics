# Machine Learning workflow to thermally correct the band gap of solid solutions
Implements an accelerated workflow that combines GNNs and MLIPs to apply thermal corrections to the band gap of semiconductor solid solutions.


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
