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
# Plots
import matplotlib.pyplot as plt
from matplotlib import animation, rc

class Water_Tank:
    """Classe représantant le bac d'eau à stabiliser, 
    avec une longueur L et une pente gamma."""
    # Contructeur
    def __init__(self, l,g,cvg,n,m):
        """Constructeur de la classe.
        Il qui prend en objet les paramètres physiques du systems :
        l   --  Longueur de la cuve
        g   --  Pente gamma
        cvg -- paramètre nu, convergence
        et les paramètres numériques de la discrétisation :
        n   -- taille du maillage de la hauteur h, et vitesse v
        m   -- taille du maillage de la base réduite de vecteur propres sur axe imaginaire positif
        """
        self.length = l
        self.n = int(n)
        self.nt = int(math.floor(n*10/l)) # total time = 10 (arbitrary)
        self.gamma = g
        self.lgamma = 2/g*(1-math.sqrt(1-g*l))
        self.nu = cvg
        # etat (h,v)
        self.state = np.zeros((2*n,self.nt))
        # transport operator in the transformed space
        self.operator = np.zeros((n,n))
        # Eigen value space parameters
        self.m = int(2*m+1)
        self.basis = np.zeros((n,m))
        self.eigenval = np.zeros(m)
        # 
        try:
            # the slope gamma is not superior to (h=1)/l
            assert g < 1/l
        except ValueError:
            print("La pente de stabilisation est trop grande.")   
            sys.exit() 
        #
        try:
            # number of eigen must be odd 
            # 2m+1 < n total number of eigen must be inferior to dimension n
            assert int(2*m+1) < n
        except ValueError:
            print("La dimension finale dans l'espace des vecteurs propres est trop grande.")   
            sys.exit() 

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
        # CL x = 0
        flow_mix[0,-1] = 1
        # CL x = L
        flow_mix[-1,0] = 1
        #
        flow_mix = self.length/self.n*flow_mix
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
    #
    # Transformation dans le systeme d'etat
    def Solver_WaterTank(self):
        # Definition of control BK
        control_BK =-2*math.tanh(self.nu*self.length)*np.ones(self.m) # to be determined
        #
        # Initialisation
        xi = np.zeros((self.m,self.nt))
        inv_hg = np.sqrt(np.ones(self.n)-self.gamma*np.linspace(0,self.length,self.n))**-1
        interim_matrix = np.concatenate((\
        np.concatenate((inv_hg,np.eye(self.n)),axis=1),\
        np.concatenate((-inv_hg,np.eye(self.n)),axis=1)\
        ))
        # Initial condition
        xi[:,0] = self.basis.dot(interim_matrix.dot(np.ones(2*self.n)))
        # Iterative solving
        dt = 1/self.nt
        for j in range(0,self.nt):
            xi[:,j+1] = xi[:,j] + dt*self.eigenval + dt*control_BK
        # Transformation en Etat final
        self.state = np.linalg.pinv(self.basis)\
        .dot(np.linalg.inv(interim_matrix).dot(xi)) #pseudo inverse moore penrose and svd
    #
    # Visualisation
    def WT_Visualization(self):
         # Set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.length), ylim=(0, 1))
        line, = ax.plot([], [], lw=2)
        # time_vector = np.linspace(0,10,self.nt)
        # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        # Initialization function: plot the background of each frame
        def Init():
            line.set_data([], [])
            # time_text.set_text('')
            return line, 
    # Animation function.  This is called sequentially
        def Animate1DT(i):
            x = np.linspace(0,self.n,self.length)
            # hauteur
            y = self.state[:self.n,i]
            line.set_data(x, y)
            return line, 
        anim = animation.FuncAnimation(fig, Animate1DT, \
        init_func=Init,\
        frames=self.nt, interval=50, blit=True)  
        plt.show()
        return anim

    
      
        
        
    
