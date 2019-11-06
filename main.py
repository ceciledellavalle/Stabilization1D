# -*- coding : utf-8 -*-


# -*- coding : utf-8 -*-

###############################################################################
#########################      MAIN - PARAMETERS      #########################
###############################################################################

######################### DEFINITION OF STABILISATION PROBLEM
###              1D transport equation using finite differences 
###              Euler Explicit upstream downstream scheme

### IMPORTATION PYTHON
# System
import sys
# Math - Matrices
import numpy as np
import math
# locals
from features import Water_Tank

### PARAMETERS DEFINITION
tank_length = 10 # meters
slope_gamma = 0.05 # h = 0.5m
mesh_nb = 100
eigen_nb = 20

### PROBLEM CONSTRUCTION
wt1 = Water_Tank(tank_length,slope_gamma,mesh_nb,eigen_nb)
wt1.OperatorA()
wt1.Eigen_Basis()
print(wt1.eigenval)