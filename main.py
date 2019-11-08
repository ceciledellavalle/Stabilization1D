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
slope_gamma = 0.005 # h = 0.05m
mesh_nb = 1000
eigen_nb = 20
# Speed of convergence to stabilisation
cvg_speed = 1

### PROBLEM CONSTRUCTION
wt1 = Water_Tank(tank_length,slope_gamma,cvg_speed,mesh_nb,eigen_nb)
wt1.OperatorA()
wt1.Eigen_Basis()
print(wt1.eigenval)
print(math.pi*np.linspace(0,eigen_nb-1,eigen_nb)/tank_length)