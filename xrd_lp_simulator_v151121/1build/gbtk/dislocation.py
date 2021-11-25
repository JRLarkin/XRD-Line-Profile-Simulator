# dislocation.py
""" This is the dislocation module containing helper functions for handling dislocations
Author:  Chris Race
Date:    19th June 2019
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
import math
import pandas as pd
import random
from . import crystaltools as ct

# For visualisation only
try:
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
except ModuleNotFoundError:
    pass
    
DISLOCATION_TOL = 1e-3

class DislocationLoop(object):
    def __init__(self, r0, major, minor, b):
        self.r0 = np.array(r0)
        self.ax_maj = np.array(major)
        self.A = np.linalg.norm(self.ax_maj)
        self.ax_min = np.array(minor)
        self.B = np.linalg.norm(self.ax_min)
        self.b = np.array(b)
        #self.n = np.array(n)/np.linalg.norm(np.array(n))
        self.n = np.cross(self.ax_maj,self.ax_min)/np.linalg.norm(self.ax_maj)/np.linalg.norm(self.ax_min)
        self.bnorm = np.abs(np.dot(self.b,self.n))
        if np.dot(self.b,self.n) > 0:
            self.int = True  # Loop is interstitial character
        else:
            self.int = False # Loop is vacancy character
            
    def insert_loop(self, supercell, r, t):
        c = 0.5*np.einsum('ij->j',supercell)
        rp = ct.translate_coordinates(r, c-self.r0, supercell)
        rp = rp - c
        if self.int:
            rp = rp + 0.5*self.bnorm
        h = np.dot(rp,self.n)
        R = rp - np.einsum('i,j->ij', h, self.n)
        p = np.dot(R,self.ax_maj)/np.linalg.norm(self.ax_maj)
        q = np.dot(R,self.ax_min)/np.linalg.norm(self.ax_min)
        P = np.sqrt(p*p/self.A/self.A + q*q/self.B/self.B)
        H = h/np.sqrt(self.A*self.B)
        d = 0.5 * np.einsum('i,j->ij', omega_approx(P,H, r_nom=1.2), self.b)
        if self.int:
            disc_mask = np.logical_and.reduce([ P < 1.0, h-0.5*self.bnorm <= self.bnorm/2.0, h-0.5*self.bnorm > -1.0*self.bnorm/2.0 ])
            r_new = np.concatenate((r + d, r[disc_mask]-0.5*self.b))
            t_new = np.concatenate((t, np.full(np.shape(r[disc_mask])[0],2)))
            return r_new, t_new, r+d, r[disc_mask]-0.5*self.b
        else:
            disc_mask = np.logical_and.reduce([ P < 1.0, h <= self.bnorm/2.0, h > -1.0*self.bnorm/2.0 ])
            r_new = r[np.invert(disc_mask)] + d[np.invert(disc_mask)]
            t_new = t[np.invert(disc_mask)]
            return r_new, t_new, r+d, r[disc_mask]

def find_loop_vectors(kind, character, plane=None, variant=None):
    print(kind, character, plane, variant)
    if kind == 'c2+p':
        if character == 'i':
            b = (1.0/6.0)*np.array([2.0,-2.0,0,3.0])
            major = (1.0/3.0)*np.array([2.0,-1.0,-1.0,0.0])
            minor = (1.0/np.sqrt(3.0))*np.array([0.0,1.0,-1.0,0.0])
        if character == 'v':
            b = (-1.0/6.0)*np.array([2.0,-2.0,0,3.0])
            major = (1.0/3.0)*np.array([2.0,-1.0,-1.0,0.0])
            minor = (1.0/np.sqrt(3.0))*np.array([0.0,1.0,-1.0,0.0])
    if kind == 'a':
        if plane == '2nd':
            if variant == '1':
                b = (-1.0/3.0)*np.array([1.0,-2.0,1.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/3.0)*np.array([2.0,-1.0,-1.0,0.0])
            if variant == '2':
                b = (-1.0/3.0)*np.array([-2.0,1.0,1.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/3.0)*np.array([-1.0,-1.0,2.0,0.0])
            if variant == '3':
                b = (-1.0/3.0)*np.array([1.0,1.0,-2.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/3.0)*np.array([-1.0,2.0,-1.0,0.0])
        if plane == '1st':
            if variant == '1':
                b = (-1.0/3.0)*np.array([1.0,-2.0,1.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/np.sqrt(3.0))*np.array([1.0,0.0,-1.0,0.0])
            if variant == '2':
                b = (-1.0/3.0)*np.array([-2.0,1.0,1.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/np.sqrt(3.0))*np.array([0.0,-1.0,1.0,0.0])
            if variant == '3':
                b = (-1.0/3.0)*np.array([1.0,1.0,-2.0,0.0])
                major = np.array([0.0,0.0,0.0,1.0])
                minor = (1.0/np.sqrt(3.0))*np.array([-1.0,1.0,0.0,0.0])
        if character == 'v':
            b = -1.0*b
    return b, major, minor
        
            
def omega_approx(P,H, r_nom=1.0):
    Q = np.sqrt(P*P + H*H)
    theta_perp = 2.0*np.arctan2(1.0,Q)
    theta_par = np.sign(H)*( np.arctan2((P+r_nom),np.abs(H)) - np.arctan2((P-r_nom),np.abs(H)) ) 
    return theta_perp * theta_par / np.pi**2
    
def heuman_lambda(B,k):
    """Compute Heuman Lambda function"""
    kp = np.sqrt(1-k*k)
    E = special.ellipe(k)
    K = special.ellipk(k)
    incE = special.ellipeinc(B,kp)
    incF = special.ellipkinc(B,kp)
    return 2.0/np.pi * (E*incF + K*incE - K*incF )

def omega_ellipse_NOT_WORKING(a,b,p,q,h):
    ## a must be greater then b
    A = p*p/(a*a*h*h) + q*q/(b*b*h*h) - 1.0/(a*a) - 1.0/(h*h)
    B = p*h/(a*a*h*h)
    C = q*h/(b*b*h*h)
    D = 1.0/(a*a) - 1.0/(b*b)
    #print(A,B,C,D)
    
    a1 = -1.0*(A/B + B/D + (C**2)/(B*D))
    b1 = A/D - 1.0
    c1 = B/D
    #print(a1,b1,c1)
    
    #     # This approach attempts to calculate on of the roots of the cubic directly
    #     r = 2.0*a1**3 - 9.0*a1*b1 + 27.0*c1
    #     s = np.sqrt( r**2 - 4.0*(a1**2 - 3.0*b1)**3 +0.0j)
    #     w3 = -a1/3.0 + (1.0 - 1j*np.sqrt(3.0))/6.0 * np.cbrt( 
    #         (r+s)/2.0 + (1.0 + 1j*np.sqrt(3.0))/6.0*np.cbrt((r-s)/2.0)
    #     )

    # Instead get python to calculate the roots
    ws = np.roots(np.array([1.0,a1,b1,c1]))
    g = np.arctan(ws)
    p = -1.0*np.arctan( (A*np.sin(2.0*g) + 2.0*B*np.cos(2.0*g)) / (2.0*C*np.sin(g)) )
    zp = 1.0
    A1 = 1.0/(a*a) + A*(np.sin(g))**2 + B*np.sin(2.0*g)
    B1 = 2.0*C*np.sin(g)*np.cos(p) - 2.0*B*np.cos(2.0*g)*np.sin(p) - A*np.sin(2.0*g)*np.sin(p)
    C1 = (np.sin(p))**2/(a*a) + (np.cos(p))**2/(b*b) - B*np.sin(2.0*g)*(np.sin(p))**2 - C*np.cos(g)*np.sin(2.0*p) + A*(np.cos(g))**2*(np.sin(p))**2
    Delta = B1*B1-4.0*A1*C1
    if Delta[0]<0.0:
        w = ws[0]
    elif Delta[1]<0.0:
        w = ws[1]
    elif Delta[2]<0.0:
        w = ws[2]
    g = np.arctan(w)
    p = -1.0*np.arctan( (A*np.sin(2.0*g) + 2.0*B*np.cos(2.0*g)) / (2.0*C*np.sin(g)) )
    zp = 1.0
    A1 = 1.0/(a*a) + A*(np.sin(g))**2 + B*np.sin(2.0*g)
    B1 = 2.0*C*np.sin(g)*np.cos(p) - 2.0*B*np.cos(2.0*g)*np.sin(p) - A*np.sin(2.0*g)*np.sin(p)
    C1 = (np.sin(p))**2/(a*a) + (np.cos(p))**2/(b*b) - B*np.sin(2.0*g)*(np.sin(p))**2 - C*np.cos(g)*np.sin(2.0*p) + A*(np.cos(g))**2*(np.sin(p))**2
    F1 = zp*zp * ( B*np.sin(2.0*g)*(np.cos(p))**2 - (np.cos(p))**2/(a*a) - (np.sin(p))**2/(b*b) - A*(np.cos(g))**2*(np.cos(p))**2 - C*np.cos(g)*np.sin(2.0*p) )
    X1 = np.sign(A1-C1) * np.sqrt( B1*B1 + (A1-C1)*(A1-C1) )
    A2 = 0.5 * ( A1 + C1 + X1 )
    C2 = 0.5 * ( A1 + C1 - X1 )
    a = np.sqrt( F1/A2 )
    b = np.sqrt( F1/C2 )
    B = np.arcsin( np.sqrt( 1.0 / (1.0 + b*b) ) )
    k = np.sqrt( (a*a - b*b) / (1.0 + a*a) )
    #print(B)
    #print(k)
    omega = 2.0*np.pi*(1.0 - heuman_lambda(B,k))
    # Why doesn't it work for round or nearly round shapes
    # Why don't the values exactly match the paper? Do I need higher precision
    return(a*a-b*b)
    
    

def u_perp(x,lam,t):
    u = np.zeros(np.shape(x)[0])
    u[np.where(t*x>0)] = 1 - np.exp(-t*x[np.where(t*x>0)]/lam)
    return u

def u_par(x,lam,boost=False):
    mult = 1.0
    if boost:
        mult = np.exp(1.0/lam)
    u = np.zeros(np.shape(x)[0])
    u[np.where(x>0)] = -np.exp(-x[np.where(x>0)]/lam)
    u[np.where(x<0)] = np.exp(x[np.where(x<0)]/lam)
    return mult*u

def create_2d_dislocation_array(n_base, n_disl, pad, a, disl_type='1/2[110]', filename='lammps.txt', lambda_x=7, lambda_y=2, boost=False):
    #n_base = 100, n_disl = 10, pad = 10, lambda_x = 7, lambda_y = 2
    if disl_type=='1/2[110]':
        b = 1.0
        a = a/np.sqrt(2)
        nmotif = 2
        motif = np.array([ [0.0,0.0,0.0], [0.5,0.5,0.5]])
        cell = np.array([ [1,0,0], [0,1,0], [0,0,np.sqrt(2)] ])
        supercell = a*np.array([n_base*cell[0,:], n_base*cell[1,:], cell[2,:] ])
    elif disl_type=='[100]':
        b = 1.0
        nmotif = 4
        motif = np.array([ [0.0,0.0,0.0], [0.5,0.5,0.0], [0.0,0.5,0.5], [0.5,0.0,0.5] ])
        cell = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
        supercell = a*np.array([n_base*cell[0,:], n_base*cell[1,:], cell[2,:] ])
    else:
        raise RuntimeError("disl_type must be 1/2[110] or [100]")

    size = np.array([n_base,n_base])
    size[0] = size[0] + n_disl/2
    # Dislocation centres
    p_disl = np.zeros((n_disl,2), dtype=int)
    random.seed()
    p_disl[:,0] = random.sample(range(pad,size[0]-pad), n_disl)
    p_disl[:,1] = np.random.randint(pad,size[1]-(1+pad), n_disl)
    # Dislocation sense
    t_disl = np.full(n_disl, -1, dtype=int)
    t_disl[random.sample(range(n_disl),int(n_disl/2))] = 1
    # Generate lattice sites
    r = []
    for i in range(size[0]):
        if i in p_disl[:,0]:
            dis_ind = np.where(p_disl[:,0]==i)[0][0]
            if t_disl[dis_ind]==1:
                y_min = 0
                y_max = p_disl[dis_ind,1]
            else:
                y_min = p_disl[dis_ind,1]
                y_max = size[1] 
        else:
            y_min = 0
            y_max = size[1]    
        for j in range(y_min,y_max):
            r.append([1.0*i,1.0*j])
    r = np.array(r)

    # Adjust coordinates
    u = np.zeros(np.shape(r)[0])
    for s in range(0,n_disl):
        u = u + 0.5*b * u_perp(r[:,1]-p_disl[s,1],lambda_y, t_disl[s]) * u_par(r[:,0]-p_disl[s,0],lambda_x,boost)
    # Displacement field
    r[:,0] = r[:,0] + u[:]
    # Rescaling
    r[:,0] = r[:,0] * n_base/size[0]
    # Boundary fix
    target = 0.5*(r[np.where(r[:,1]<0.5)[0],0] + r[np.where( (r[:,1]>=size[1]-(1.5)) & (r[:,1]<size[1]-(0.5)) )[0],0])
    fix_rows = pad
    for s in range(fix_rows):
        alpha = 1.0 - (s+1)/(fix_rows+1)
        r[np.where( (r[:,1]>=size[1]-(s+1.5)) & (r[:,1]<size[1]-(s+0.5)) )[0],0]  = (1.0-alpha)* r[np.where( (r[:,1]>=size[1]-(s+1.5)) & (r[:,1]<size[1]-(s+0.5)) )[0],0] + alpha * target
        r[np.where( (r[:,1]<(s+0.5)) & (r[:,1]>(s-0.5)) )[0],0]  = (1.0-alpha)* r[np.where( (r[:,1]<(s+0.5)) & (r[:,1]>(s-0.5)) )[0],0] + alpha * target
    # Generate crystal
    natoms = np.shape(r)[0]
    atom_pos = np.zeros((nmotif*natoms,3))
    for i in range(natoms):
        for s in range(nmotif):
            atom_pos[i*nmotif+s,:2] = a*(r[i,:] + motif[s,:2])
            atom_pos[i*nmotif+s,2] = a*motif[s,2]*cell[2,2]

    ct.write_lammps(supercell, atom_pos, atom_type=None, filename=filename, num_types=1)
    
def create_2d_res_ran_dislocation_array(n_base, n_cell, n_disl, pad, a, disl_type='1/2[110]', filename='lammps.txt', lambda_x=7, lambda_y=2, boost=False):
    #n_base = 100, n_disl = 10, pad = 10, lambda_x = 7, lambda_y = 2
    if n_base%n_cell != 0:
        raise RuntimeError("length of supercell must be divisible by length of sub-cell")
    num_cells = int((n_base*n_base)/(n_cell*n_cell))
    num_cells_x =int((n_base)/(n_cell))
    if n_disl%num_cells != 0:
        raise RuntimeError("Number of dislocations must be divisible by number of sub-cells")
    else:
        n_disl_cell = int(n_disl/num_cells)
    
    if disl_type=='1/2[110]':
        b = 1.0
        a = a/np.sqrt(2)
        nmotif = 2
        motif = np.array([ [0.0,0.0,0.0], [0.5,0.5,0.5]])
        cell = np.array([ [1,0,0], [0,1,0], [0,0,np.sqrt(2)] ])
        supercell = a*np.array([n_base*cell[0,:], n_base*cell[1,:], cell[2,:] ])
    elif disl_type=='[100]':
        b = 1.0
        nmotif = 4
        motif = np.array([ [0.0,0.0,0.0], [0.5,0.5,0.0], [0.0,0.5,0.5], [0.5,0.0,0.5] ])
        cell = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
        supercell = a*np.array([n_base*cell[0,:], n_base*cell[1,:], cell[2,:] ])
    else:
        raise RuntimeError("disl_type must be 1/2[110] or [100]")

    size = np.array([n_base,n_base])
    size_cell = np.array([n_cell,n_cell])
    size[0] = size[0] + n_disl/2
    size_cell[0] = size_cell[0] + n_disl/2/num_cells_x
    # Dislocation centres
    p_disl = np.zeros((n_disl,2), dtype=int)
    # Dislocation sense
    t_disl = np.full(n_disl, -1, dtype=int)
    
    random.seed()
    for i in range(num_cells_x):
        p_disl[i*num_cells_x*n_disl_cell:(i+1)*num_cells_x*n_disl_cell,0] = random.sample(range(i*size_cell[0]+pad,(i+1)*size_cell[0]-pad), n_disl_cell*num_cells_x)
        for j in range(num_cells_x):
            p_disl[(i*num_cells_x+j)*n_disl_cell : (i*num_cells_x+(j+1))*n_disl_cell,1] = np.random.randint(j*size_cell[1]+pad,(j+1)*size_cell[1]-pad, n_disl_cell)
            t_disl[random.sample(range((i*num_cells_x+j)*n_disl_cell, (i*num_cells_x+(j+1))*n_disl_cell),int(n_disl_cell/2))] = 1

    # Generate lattice sites
    r = []
    for i in range(size[0]):
        if i in p_disl[:,0]:
            dis_ind = np.where(p_disl[:,0]==i)[0][0]
            if t_disl[dis_ind]==1:
                y_min = 0
                y_max = p_disl[dis_ind,1]
            else:
                y_min = p_disl[dis_ind,1]
                y_max = size[1] 
        else:
            y_min = 0
            y_max = size[1]    
        for j in range(y_min,y_max):
            r.append([1.0*i,1.0*j])
    r = np.array(r)

    # Adjust coordinates
    u = np.zeros(np.shape(r)[0])
    for s in range(0,n_disl):
        u = u + 0.5*b * u_perp(r[:,1]-p_disl[s,1],lambda_y, t_disl[s]) * u_par(r[:,0]-p_disl[s,0],lambda_x,boost)
    # Displacement field
    r[:,0] = r[:,0] + u[:]
    # Rescaling
    r[:,0] = r[:,0] * n_base/size[0]
    # Boundary fix
    target = 0.5*(r[np.where(r[:,1]<0.5)[0],0] + r[np.where( (r[:,1]>=size[1]-(1.5)) & (r[:,1]<size[1]-(0.5)) )[0],0])
    fix_rows = pad
    for s in range(fix_rows):
        alpha = 1.0 - (s+1)/(fix_rows+1)
        r[np.where( (r[:,1]>=size[1]-(s+1.5)) & (r[:,1]<size[1]-(s+0.5)) )[0],0]  = (1.0-alpha)* r[np.where( (r[:,1]>=size[1]-(s+1.5)) & (r[:,1]<size[1]-(s+0.5)) )[0],0] + alpha * target
        r[np.where( (r[:,1]<(s+0.5)) & (r[:,1]>(s-0.5)) )[0],0]  = (1.0-alpha)* r[np.where( (r[:,1]<(s+0.5)) & (r[:,1]>(s-0.5)) )[0],0] + alpha * target
    # Generate crystal
    natoms = np.shape(r)[0]
    atom_pos = np.zeros((nmotif*natoms,3))
    for i in range(natoms):
        for s in range(nmotif):
            atom_pos[i*nmotif+s,:2] = a*(r[i,:] + motif[s,:2])
            atom_pos[i*nmotif+s,2] = a*motif[s,2]*cell[2,2]

    ct.write_lammps(supercell, atom_pos, atom_type=None, filename=filename, num_types=1)