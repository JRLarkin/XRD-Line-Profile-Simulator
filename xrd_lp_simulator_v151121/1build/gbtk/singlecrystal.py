# singlecrystal.py
""" This is the singlecrystal module containing definitions of single crystals
Author:  Chris Race
Date:    19th June 2019
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
import math
import pandas as pd
from . import crystaltools as ct
from . import lattice

# For visualisation only
# try:
#     import plotly
#     import plotly.figure_factory as ff
#     import plotly.graph_objs as go
#     plotly.offline.init_notebook_mode(connected=True)
# except ModuleNotFoundError:
#     pass
    

DISLOCATION_TOL = 1e-3

class SingleCrystal(object):
    """A singlecrystal holds details of the lattice type, size and orientation of a single crystal supercell.
    
    Attributes
    ----------
    """
    
    def __init__(self, latticetype, lengths=None, angles=None):
        """Initialise an empty supercell"""
        self.lattice = lattice.Lattice(latticetype, lengths, angles)
        self.a = 1.0
        self.strain = np.identity(3)
        self.num_atoms = 0
        self.supercell_size_set = False
        self.debug = False
        self.atom_arrays_calculated = False
        self.offset = np.zeros(3, dtype=float)
        
    def set_debug(self):
        """Turn on debug info"""
        self.debug = True
        
    def unset_debug(self):
        """Turn off debug info"""
        self.debug = False
    
    def set_lattice_parameter(self,a):
        """Set the value of the primary lattice parameter"""
        self.a = a
    
    def set_size(self, size):
        """Set size of crystal in multiples of lattice vectors"""
        if len(size) != 3:
            raise RuntimeError("Supercell size specification must contain 3 elements")
        self.supercell_size = np.array(size)
        self.supercell_vectors = np.zeros((3,3), dtype=float)
        for s in range(3):
            self.supercell_vectors[s,:] = self.a * size[s] * self.lattice.cell_vectors[s,:]
        self.supercell_size_set = True
        
    def set_offset(self, offset):
        """Set offest vector to shift atom positions from origin by this amount"""
        self.offset = np.array(offset)
        return
    
    def calculate_atom_array_simple(self):
        """Generates an array of atom positions from the lattice vectors"""
        if not self.supercell_size_set:
            raise RuntimeError("Supercell size must be specified before filling atom arrays")
        
        nbasis = self.lattice.num_basis
        cellvectors = self.lattice.cell_vectors
        basis = self.lattice.basis_coords
        
        r = []
        t = []
        
        x = np.arange(0,int(self.supercell_size[0]), 1)
        y = np.arange(0,int(self.supercell_size[1]), 1)
        z = np.arange(0,int(self.supercell_size[2]), 1)
        V = len(x) * len(y) * len(z)
        indices = (np.stack(np.meshgrid(x, y, z)).T).reshape(V, 3)
        for i in range(V):
            for p in range(nbasis):
                pos = self.a * ( (indices[i,0] + basis[p,0] + self.offset[0])*cellvectors[0,:] + (indices[i,1] + basis[p,1] + self.offset[1])*cellvectors[1,:] + (indices[i,2] + basis[p,2] + self.offset[2])*cellvectors[2,:] )
                r.append(pos.tolist())
                t.append(self.lattice.atom_types[p])
        self.num_atoms = len(r)
        self.atom_arrays_calculated = True
        return np.array(r), np.array(t)

        