# -*- coding : utf-8 -*-

###############################################################################
#########################     CLASS CONSTRUCTION      #########################
###############################################################################
#
######################### DEFINITION OF STABILISATION PROBLEM
###              1D transport equation using finite differences 
###              Euler Explicit upstream downstream scheme

### IMPORTATION PYTHON
# System
import sys
# Math - Matrices
import numpy as np
import math

class Water_Tank:
    """Classe représantant le bac d'eau à stabiliser, 
    avec une longueur L et une pente gamma."""
    # Contructeur
    def __init__(self, l,g,n,m):
        """Constructeur de la classe.
        Il qui prend en objet les paramètres physiques du systems :
        l --  Longueur de la cuve
        g --  Pente gamma
        et les paramètres numériques de la discrétisation :
        n -- taille du maillage de la hauteur h, et vitesse v
        m -- taille du maillage de la base réduite de vecteur propres sur axe imaginaire positif"""
        self.length = l
        self.gamma = g
        self.lgamma = 2/g*(1-math.sqrt(1-g*l))
        self.n = int(n)
        self.operator = np.zeros((n,n))
        try:
            # number of eigen must be odd 
            # 2m+1 < n total number of eigen must be inferior to dimension n
            assert int(2*m+1) < n
        except ValueError:
            print("La dimension finale dans l'espace des vecteurs propres est trop grande.")   
            sys.exit() 
        self.m = int(2*m+1)
        self.basis = np.zeros((m,n))
        self.eigenval = np.zeros(m)
    #
    # Calcul de l'operateur 
    def OperatorA(self):
        # Operator of Transport
        # upstream scheme
        flow_up = np.eye(self.n) - np.diag(np.ones(self.n-1),-1)
        flow_up[0,0] = 0
        flow_down = np.diag(np.ones(self.n-1),1) - np.eye(self.n)
        flow_down[-1,-1] = 0
        flow_mix = np.concatenate((\
        np.concatenate((flow_up,np.zeros((self.n,self.n))),axis=1),\
        np.concatenate((np.zeros((self.n,self.n)),-flow_down),axis=1)\
        ))
        # CL x= 0
        flow_mix[0,-1] = 1
        # CL x = L
        flow_mix[-1,0] = 1
        #
        # Operator of Control
        delta_J = -3/4*self.lgamma/self.length*self.gamma/(1-self.gamma/2)\
        *np.concatenate((\
        np.concatenate((np.zeros((self.n,self.n)), 1/3*np.diag(np.linspace(0,self.length,self.n))),axis=1),\
        np.concatenate((-1/3*np.diag(np.linspace(0,self.length,self.n)),np.zeros((self.n,self.n))),axis=1)\
        ))
        #
        # Sum of operators
        self.operator = flow_mix + delta_J
    #
    # Calcul de la base de vecteurs propres
    def Eigen_Basis(self):
        # Construction of the dynamic 
        eigenValues, eigenVectors = np.linalg.eig(self.operator)
        eigenValues_modulus = np.abs(eigenValues)
        idx = eigenValues_modulus.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        self.basis = eigenVectors[:,:self.m]
        self.eigenval = eigenValues[:self.m]
    
