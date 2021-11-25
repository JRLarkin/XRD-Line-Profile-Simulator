import os
import sys
import gbtk.singlecrystal as sc
import gbtk.crystaltools as ct
import gbtk.dislocation as dislocation
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('a', type=float)
parser.add_argument('c', type=float)
parser.add_argument('nx', type=int)
parser.add_argument('ny', type=int)
parser.add_argument('nz', type=int)
parser.add_argument('rx', type=float)
parser.add_argument('ry', type=float)
parser.add_argument('rz', type=float)
parser.add_argument('R', type=float)
parser.add_argument('kind', type=str)
parser.add_argument('character', type=str)
parser.add_argument('plane', type=str)
parser.add_argument('variant', type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()

a = args.a
c = args.c

x = np.array([1,0,0])
y = np.array([-np.cos(np.pi/3),np.sin(np.pi/3),0])
u = np.array([-np.cos(np.pi/3),-np.sin(np.pi/3),0])
z = np.array([0,0,c/a])

#b = {}
#b['c/2+p'] = (1/6)*(2*x - 2*y + 3*z)
#b['a'] = (1/3)*(-1*x + 2*y -1*u)

test_sc = sc.SingleCrystal('hcp-ortho', lengths=[c/a])
test_sc.set_debug()
test_sc.set_lattice_parameter(a)
test_sc.set_size([args.nx, args.ny, args.nz])
r,t = test_sc.calculate_atom_array_simple()

r_new = np.copy(r)
t_new = np.copy(t)

loop_idx, loop_centre = ct.locate_nearest_atom([args.rx, args.ry, args.rz],r)

b, major, minor = dislocation.find_loop_vectors(args.kind, args.character, args.plane, args.variant)

b = a*(b[0]*x + b[1]*y + b[2]*u + b[3]*z)
major = args.R*(major[0]*x + major[1]*y + major[2]*u + major[3]*z)
major = major/(c/a)   # HACK ALERT - line added to ensure a circular a-loop.
minor = args.R*(minor[0]*x + minor[1]*y + minor[2]*u + minor[3]*z)

#print(b)
#print(major)
#print(minor)

test_loop = dislocation.DislocationLoop(loop_centre, major, minor, b)
r_new, t_new, r_disp, r_disc = test_loop.insert_loop(test_sc.supercell_vectors, r_new, t_new)

ct.write_lammps(test_sc.supercell_vectors, r_new, t_new, filename=args.output_file, num_types=2)