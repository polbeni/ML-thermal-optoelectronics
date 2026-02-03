#!/bin/bash

dynaphopy input-anharmonic XDATCAR -ts 0.001 -sfc FORCE_CONSTANTS -psm 2

phonopy -ps phonopy-anharmonic.conf