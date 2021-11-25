# crystaltools.py
""" This is the crystaltools module. It contains a number of helper functions for
handling and visualising crystal structures
Author:  Chris Race
Date:    3rd January 2017
Contact: christopher.race@manchester.ac.uk
"""
import numpy as np
import math
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull

CRYSTALTOOLS_TOL = 1e-3

def check_common_factors(indices, limit=100):
    common_factors = False
    cf = 2
    while (not common_factors) and (cf <= limit):
        if (np.product(np.array(indices)%cf==0) == 1):
            common_factors = True
        cf = cf + 1
    return common_factors

def indices_in_basis(vector,basis):
    """Express the direction of the given vector in the basis such that the components are integer"""
    return integer_indices(components_in_basis(vector,basis))
    
def components_in_basis(vector,basis):
    """Calculate the components of the vector in the given basis"""
    return np.einsum('i,ij',vector,np.linalg.inv(basis))

def integer_indices(vector, limit=10000, tol=1e-3):
    """Return a multiple of the given vector such that the components are integer"""
    #vp = vector/np.min(vector)
    vp = vector/np.min(np.abs(vector[np.where(vector!=0.0)]))
    found = False
    i = 1
    while (not found) and (i<=limit):
        #print(i,vp*i,np.round(vp*i),np.sum(abs(np.round(vp*i)-vp*i)))
        #if abs(np.sum(vp*i%1.0)) < tol:
        if np.sum(abs(np.round(vp*i)-vp*i)) < tol:
            found = True
        i = i + 1
    vp = vp*(i-1)
    if found:
        return np.round(vp,0)
    else:
        return np.array([0.0,0.0,0.0])

def vector_in_basis(components,basis_vectors):
    """Return a vector in the basis in which the given basis_vectors are expressed
    with components of multiples of the given basis_vectors """
    return np.einsum('i,ij',components,basis_vectors)
        
def rotation_matrix(axis,theta):
    """Calculate and return a rotation matrix from an axis angle combination"""
    #print(axis)
    axis = axis/math.sqrt(np.dot(axis,axis))
    c = math.cos(theta)
    s = math.sin(theta)
    x,y,z = axis
    return np.array([[c+x*x*(1.0-c), x*y*(1.0-c)-z*s, x*z*(1.0-c)+y*s],
                     [y*x*(1.0-c)+z*s, c+y*y*(1.0-c), y*z*(1.0-c)-x*s], 
                     [z*x*(1.0-c)-y*s, z*y*(1.0-c)+x*s, c+z*z*(1.0-c)]])

def rotation_matrix_into_direction(vector, direction):
    """Find rotation matrix to rotate a vector into a given direction"""
    
    v = np.cross(vector,direction)/np.linalg.norm(vector)/np.linalg.norm(direction)
    c = np.dot(vector,direction)/np.linalg.norm(vector)/np.linalg.norm(direction)
    vmat = np.array([
        [0,-v[2],v[1]],
        [v[2],0,-v[0]],
        [-v[1],v[0],0]
    ])
    if c == -1:
        rotation = -np.identity(3)
    else:
        rotation = np.identity(3) + vmat + np.dot(vmat,vmat)*(1.0/(1.0+c))
    return rotation
                     
def rotate_cell(cellvectors,axis,angle):
    R = rotation_matrix(axis,angle)
    return np.transpose(np.dot(R,np.transpose(cellvectors)))
        
def get_ortho_bounding_box(boxvectors):
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    initmin = 999999.0
    boundingbox = np.array([[initmin,-initmin],[initmin,-initmin],[initmin,-initmin]])
    for i in range(8):
        corner = corners[i,0]*boxvectors[0] + corners[i,1]*boxvectors[1] + corners[i,2]*boxvectors[2]
        for s in range(3):
            if corner[s] < boundingbox[s,0]:
                boundingbox[s,0] = corner[s]
            if corner[s] > boundingbox[s,1]:
                boundingbox[s,1] = corner[s]
    return boundingbox

# def get_bounding_box(boxvectors, cellvectors):
#     corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
#     initmin = 999999.0
#     boundingbox = np.array([[initmin,-initmin],[initmin,-initmin],[initmin,-initmin]])
#     for i in range(8):
#         corner = corners[i,0]*boxvectors[0] + corners[i,1]*boxvectors[1] + corners[i,2]*boxvectors[2]
#         for s in range(3):
#             dist = np.dot(cellvectors[s,:],corner)/np.linalg.norm(cellvectors[s,:])
#             if dist < boundingbox[s,0]:
#                 boundingbox[s,0] = dist
#             if dist > boundingbox[s,1]:
#                 boundingbox[s,1] = dist
#     return boundingbox
#
# def get_repeats(supercellvectors,cellvectors,lattice_param=1.0, axis=None,angle=None):
#     """Return the number of repeats along each lattice vector required to fill a box with atoms"""
#     if axis is not None:
#         rcellvectors = rotate_cell(cellvectors,axis,angle)
#     else:
#         rcellvectors = cellvectors
#     boundingbox = get_bounding_box(supercellvectors, rcellvectors)
#     for s in range(3):
#         boundingbox[s,0] = int(boundingbox[s,0]/lattice_param - 1)
#         boundingbox[s,1] = int(boundingbox[s,1]/lattice_param + 1)
#     return boundingbox

def get_bounding_box(boxvectors, cellvectors):
    corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    initmin = 999999.0
    boundingbox = np.array([[initmin,-initmin],[initmin,-initmin],[initmin,-initmin]])
    for i in range(8):
        corner = corners[i,0]*boxvectors[0] + corners[i,1]*boxvectors[1] + corners[i,2]*boxvectors[2]
        for s in range(3):
            dist = np.dot(cellvectors[s,:],corner)/np.linalg.norm(cellvectors[s,:])**2
            if dist < boundingbox[s,0]:
                boundingbox[s,0] = dist
            if dist > boundingbox[s,1]:
                boundingbox[s,1] = dist
    #print(boundingbox)
    return boundingbox

def get_repeats(supercellvectors,cellvectors, axis=None,angle=None):
    """Return the number of repeats along each lattice vector required to fill a box with atoms"""
    if axis is not None:
        rcellvectors = rotate_cell(cellvectors,axis,angle)
    else:
        rcellvectors = cellvectors
    boundingbox = get_bounding_box(supercellvectors, rcellvectors)
    for s in range(3):
        boundingbox[s,0] = int(boundingbox[s,0] - 1)
        boundingbox[s,1] = int(boundingbox[s,1] + 1)
    return boundingbox
    
def fill_box(supercellvectors, cellvectors, basis, repeats, basis_types=None):
    """Return a list of atoms filling a box defined by supercellvectors, using a set of lattice
    vectors given by cellvectors and a basis. Repeats contains the search bounds required to fill the box"""
    nbasis = len(basis)
    if basis_types is not None:
        if len(basis_types) != nbasis:
            raise RuntimeError("Array of atom types must match length of basis")
    r = []
    t = []
    
    x = np.arange(int(repeats[0,0]),int(repeats[0,1]), 1)
    y = np.arange(int(repeats[1,0]),int(repeats[1,1]), 1)
    z = np.arange(int(repeats[2,0]),int(repeats[2,1]), 1)
    V = len(x) * len(y) * len(z)
    indices = (np.stack(np.meshgrid(x, y, z)).T).reshape(V, 3)
    for i in range(V):
        for p in range(nbasis):
            pos = (indices[i,0] + basis[p,0])*cellvectors[0,:] + (indices[i,1] + basis[p,1])*cellvectors[1,:] + (indices[i,2] + basis[p,2])*cellvectors[2,:]
            if is_in_cell(supercellvectors, pos):
                r.append(pos.tolist())
                if basis_types is not None:
                    t.append(basis_types[p])
    # #print(repeats)
    # for i in range(int(repeats[0,0]),int(repeats[0,1])):
    #     for j in range(int(repeats[1,0]),int(repeats[1,1])):
    #         for k in range(int(repeats[2,0]),int(repeats[2,1])):
    #             for p in range(nbasis):
    #                 pos = (i + basis[p,0])*cellvectors[0,:] + (j + basis[p,1])*cellvectors[1,:] + (k + basis[p,2])*cellvectors[2,:]
    #                 if is_in_cell(supercellvectors, pos):
    #                     r.append(pos.tolist())
    #                     if basis_types is not None:
    #                         t.append(basis_types[p])
    if basis_types is not None:
        return len(r),r,t
    else:
        return len(r),r
    
def is_in_cell(cellvectors, pos, tol=1e-6):
    """Is the postion pos within the cell"""
    incell = True
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(pos,cross)/np.dot(cellvectors[i,:],cross)
        if test < 0.0-tol or test >= 1.0-tol:
            incell = False
    return incell
    
def wrap_to_cell(cellvectors, pos):
    """Wrap a coordinate into a cell assuming periodic boundary conditions"""
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(pos,cross)/np.dot(cellvectors[i,:],cross)
        if test < 0.0:
            pos = pos + cellvectors[i,:]
        elif test >= 1.0:
            pos = pos - cellvectors[i,:]
    return pos
    
def translate_coordinates(r, dr, cellvectors):
    """Translate all the coordiantes in the array r by the vector dr, then rewrap to the box specified by cellvectors"""
    rp = np.array(r) + dr
    for i in range(np.shape(rp)[0]):
        rp[i,:] = wrap_to_cell(cellvectors, rp[i,:])
    return rp

def wrap_vector_to_cell(cellvectors, vec):
    """Wrap a vector into a cell assuming periodic boundary conditions"""
    for i in range(3):
        cross = np.cross(cellvectors[(i+1)%3,:],cellvectors[(i+2)%3,:])
        test = np.dot(vec,cross)/np.dot(cellvectors[i,:],cross)
        if test < -0.5:
            vec = vec + cellvectors[i,:]
        elif test >= 0.5:
            vec = vec - cellvectors[i,:]
    return vec

def calculate_corners_ppp(cellvectors,repeats):
    """Calculate the coordinates of a parallelipiped"""
    corners = np.zeros([8,3], dtype=float)
    cornerindex = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])
    for i in range(8):
        for s in range(3):
            corners[i,:] = corners[i,:] + repeats[s,cornerindex[i,s]]*cellvectors[s,:]
    return corners
    
def calculate_edges_ppp():
    """Define the edges of a parallelipiped consistent with the ordering in calculate_corners_ppp function"""
    edges = np.array([[0,1],[0,2],[0,3],[1,4],[2,4],[4,7],[5,7],[3,5],[3,6],[6,7],[1,6],[2,5]])
    return edges

def vis_data_lines(points):
    """Return data for visualising a series of line segments in a plotly scatter plot
    segments run from points[i,0,:] to points[i,1,:]"""
    data = []
    for i in range(len(points)):
        data.append(points[i,0,:].tolist())
        data.append(points[i,1,:].tolist())
        data.append([None,None,None]) 
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(), dataarray[:,2].tolist()
    
def vis_data_vectors(points, vectors):
    """Return data for visualising a set of vectors in a plotly scatter plot
    vectors run from points[i,:] to points[i,:]+vectors[i,:]"""
    data = []
    for i in range(len(points)):
        data.append(points[i,:].tolist())
        data.append((points[i,:] + vectors[i,:]).tolist())
        data.append([None,None,None]) 
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(), dataarray[:,2].tolist()
    
def vis_data_box(corners, edges):
    """Return data for visualising a box in a plotly scatter plot"""
    data = []
    for i in range(len(edges)):
        data.append(corners[edges[i,0],:])
        data.append(corners[edges[i,1],:])
        data.append([None,None,None])
    dataarray = np.array(data)
    return dataarray[:,0].tolist(), dataarray[:,1].tolist(),dataarray[:,2].tolist()

def vis_data_box_ppp(cellvectors, repeats):
    """Return a list of points for visualising a parallelipiped in a plotly scatter plot"""
    corners = calculate_corners_ppp(cellvectors, repeats)
    edges = calculate_edges_ppp()
    x,y,z = vis_data_box(corners, edges)
    return x,y,z
    
def lammps_dump_to_ppd(dump_cell):
    """Take a supercell as specified in a lammps dump file and convert to three edge vectors of a parallelipiped"""
    xy = dump_cell[0,2]; xz = dump_cell[1,2]; yz = dump_cell[2,2]
    xlo = dump_cell[0,0] - min(0.0, xy, xz, xy+xz)
    xhi = dump_cell[0,1] - max(0.0, xy, xz, xy+xz)
    ylo = dump_cell[1,0] - min(0.0, yz)
    yhi = dump_cell[1,1] - max(0.0, yz)
    zlo = dump_cell[2,0]
    zhi = dump_cell[2,1]
    ppd = np.array([
        [xhi-xlo, 0.0, 0.0],
        [xy, yhi-ylo, 0.0],
        [xz, yz, zhi-zlo]
    ])
    return ppd
    
def lammps_box_to_ppd(box_cell):
    """Take a supercell as specified in a lammps input file and convert to three edge vectors of a parallelipiped"""
    xy = box_cell[0,2]; xz = box_cell[1,2]; yz = box_cell[2,2]
    xlo = box_cell[0,0]
    xhi = box_cell[0,1]
    ylo = box_cell[1,0]
    yhi = box_cell[1,1]
    zlo = box_cell[2,0]
    zhi = box_cell[2,1]
    ppd = np.array([
        [xhi-xlo, 0.0, 0.0],
        [xy, yhi-ylo, 0.0],
        [xz, yz, zhi-zlo]
    ])
    return ppd

def mirror_boundary_atoms(cell, atom_data, tol=0.02):
    """ Augment atom list so that atoms on boundaries and corners are reproduced on opposite side (a cosmetic adjustment)"""
    n_atoms = np.shape(atom_data)[0]
    r_new = []
    for i in range(n_atoms):
        r_new.append(atom_data[i,:].tolist())
        pattern = []
        for s in range(3):
            cross = np.cross(cell[(s+1)%3,:],cell[(s+2)%3,:])
            test = np.dot(atom_data[i,0:3],cross)/np.dot(cell[s,:],cross)
            this_pattern = np.array([0,0,0], dtype=int)
            if test < tol:
                this_pattern[s] = 1
                pattern.append(this_pattern.tolist())
            elif test > (1.0-tol):
                this_pattern[s] = -1
                pattern.append(this_pattern.tolist())
        if np.shape(pattern)[0]==2:
            pattern.append((np.array(pattern[0]) + np.array(pattern[1])).tolist())
        elif np.shape(pattern)[0]==3:
            pattern.append((np.array(pattern[0]) + np.array(pattern[1])).tolist())
            pattern.append((np.array(pattern[1]) + np.array(pattern[2])).tolist())
            pattern.append((np.array(pattern[2]) + np.array(pattern[0])).tolist())
            pattern.append((np.array(pattern[0]) + np.array(pattern[1]) +  + np.array(pattern[2])).tolist())
        pattern = np.array(pattern)
        for s in range(np.shape(pattern)[0]):
            r_shift = np.copy(atom_data[i,:])
            shift=np.zeros(3)
            for t in range(3):
                r_shift[0:3] = r_shift[0:3] + pattern[s,t]*cell[t,:]
                shift = shift + pattern[s,t]*cell[t,:]
            r_new.append(r_shift.tolist())
    return np.array(r_new)
    
def locate_nearest_atom(r0, r):
    idx = np.linalg.norm(r - r0, axis=1).argmin()
    return idx, r[idx,:]
    
def find_bonds(cell, r, rnn, wrap=False):
    """Return an array of atom pairs that are closer together than rnn"""
    n_atoms = np.shape(r)[0]
    pairs = []
    for i in range(n_atoms):
        for j in range(i+1,n_atoms):
            if wrap:
                d = r[i,:]-r[j,:]
                d = wrap_vector_to_cell(cell, d)
                d = np.linalg.norm(d)
            else:
                d = np.linalg.norm(r[i,:]-r[j,:])
            if d <= rnn:
                pairs.append([i+1,j+1])
    return np.array(pairs)

def find_neighbours(cell, r, r_nn, wrap=True, max_neigh=16):
    n_atoms = np.shape(r)[0]
    n_neigh = np.zeros(n_atoms, dtype=int)
    neigh = np.zeros((n_atoms,max_neigh), dtype=int)
    neigh_sep = np.zeros((n_atoms,max_neigh), dtype=float)
    neigh_disp = np.zeros((n_atoms,max_neigh,3), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                d = r[i,:]-r[j,:]
                if wrap:
                    sep = np.linalg.norm(wrap_vector_to_cell(cell, d))
                else:
                    sep = np.linalg.norm(d)
                if sep <= r_nn:
                    neigh[i,n_neigh[i]] = j
                    neigh_sep[i,n_neigh[i]] = sep
                    neigh_disp[i,n_neigh[i],:] = d
                    n_neigh[i] = n_neigh[i] + 1
    return n_neigh, neigh, neigh_sep, neigh_disp
    
def get_interstices(cell,r,pad,r_cluster):
    """Take a cell specification and list of atom positions and find the interstices via a voronoi tesselation
    The atom array is augmented by an amount specified via <pad> on all faces and edges to mimic periodic boundaries
    The detected vertices of the voronoi mesh are then pruned if they are closer than <r_cluster>"""
    # Generate a padded cell and augmented atom list
    trans = np.array([
        [1,0,0], [0,1,0], [0,0,1], 
        [1,1,0], [0,1,1], [1,0,1], [-1,1,0], [0,-1,1], [1,0,-1],
        [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]
    ])
    cell_pad = np.zeros((3,3))
    shift = np.zeros(3)
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * (1.0 + 2.0*pad/np.linalg.norm(cell[s,:]))
        shift = shift + cell[s,:] * 1.0*pad/np.linalg.norm(cell[s,:])
    r_pad = r.tolist()
    for i in range(np.shape(trans)[0]):
        r_pad.extend((r+np.dot(trans[i,:],cell)).tolist())
        r_pad.extend((r-np.dot(trans[i,:],cell)).tolist())
    r_to_prune = np.array(r_pad) + shift
    r_pad = []
    for i in range(np.shape(r_to_prune)[0]):
        if is_in_cell(cell_pad, r_to_prune[i,:], tol=1e-6):
            r_pad.extend([r_to_prune[i,:].tolist()])
    # Get voronoi tesselation
    vor = Voronoi(np.array(r_pad))
    # First trim out distant vertices
    v_to_prune = vor.vertices
    v_trim = []
    for i in range(np.shape(v_to_prune)[0]):
        if is_in_cell(cell_pad, v_to_prune[i,:], tol=1e-6):
            v_trim.extend([v_to_prune[i,:].tolist()])
    # Now check for clusters of vertices close together
    v_to_cluster = np.array(v_trim)
    num_v = np.shape(v_to_cluster)[0]
    check_v = np.full(num_v, True, dtype=bool)
    v_clustered = []
    for i in range(num_v):
        if check_v[i]:
            cluster_r = [v_to_cluster[i,:].tolist()]
            n_cluster = 1
            check_v[i] = False
            for j in range(i+1,num_v):
                if check_v[j]:
                    if np.abs(np.linalg.norm(v_to_cluster[i,:] - v_to_cluster[j,:])) < r_cluster:
                        cluster_r.extend([v_to_cluster[j,:].tolist()])
                        n_cluster = n_cluster + 1
                        check_v[j] = False
            #print(n_cluster,cluster_r)
            if n_cluster > 1:
                v_clustered.extend([np.mean(np.array(cluster_r),axis=0).tolist()])
            else:
                v_clustered.extend(cluster_r)
    # Now trim atoms and vertices back to original cell
    # r_to_prune = np.array(r_pad) - shift
    # r_trim = []
    # for i in range(np.shape(r_to_prune)[0]):
    #     if is_in_cell(cell, r_to_prune[i,:], tol=1e-6):
    #         r_trim.extend([r_to_prune[i,:].tolist()])
    v_to_prune = np.array(v_clustered) - shift
    v_trim = []
    for i in range(np.shape(v_to_prune)[0]):
        if is_in_cell(cell, v_to_prune[i,:], tol=1e-6):
            v_trim.extend([v_to_prune[i,:].tolist()])
    return np.array(v_trim)
    
def get_voronoi_volumes(cell,r,pad):
    """Take a cell specification and list of atom positions and find the volumes associated with each atom via a voronoi tesselation
    The atom array is augmented by an amount specified via <pad> on all faces and edges to mimic periodic boundaries"""
    n_atoms = np.shape(r)[0]
    # Generate a padded cell and augmented atom list
    trans = np.array([
        [1,0,0], [0,1,0], [0,0,1], 
        [1,1,0], [0,1,1], [1,0,1], [-1,1,0], [0,-1,1], [1,0,-1],
        [1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]
    ])
    cell_pad = np.zeros((3,3))
    shift = np.zeros(3)
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * (1.0 + 2.0*pad/np.linalg.norm(cell[s,:]))
        shift = shift + cell[s,:] * 1.0*pad/np.linalg.norm(cell[s,:])
    r_pad = r.tolist()
    for i in range(np.shape(trans)[0]):
        r_pad.extend((r+np.dot(trans[i,:],cell)).tolist())
        r_pad.extend((r-np.dot(trans[i,:],cell)).tolist())
    r_to_prune = np.array(r_pad) + shift
    r_pad = []
    for i in range(np.shape(r_to_prune)[0]):
        if is_in_cell(cell_pad, r_to_prune[i,:], tol=1e-6):
            r_pad.extend([r_to_prune[i,:].tolist()])
    # Get voronoi tesselation
    vor = Voronoi(np.array(r_pad))
    # Now get volume for each point
    vol = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(vor.vertices[indices]).volume
    return np.array(vol)[:n_atoms]
    
def multiply_cell(cell,r,mults):
    """Take a cell specification and list of atom positions and return a larger cell with integer numbers of copies of the original positions.
    If other attributes are included in the array r, these will also be replicated"""
    n_cols = np.shape(r)[1]
    cell_pad = np.zeros((3,3))
    for s in range(3):
        cell_pad[s,:] = cell[s,:] * mults[s]
    r_pad = []
    for i in range(mults[0]):
        for j in range(mults[1]):
            for k in range(mults[2]):
                shift = np.zeros(n_cols)
                shift[:3] = i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                r_pad.extend( (r + shift).tolist() )
    return cell_pad, np.array(r_pad)

def read_lammps_dump(file):
    """Read a lammps dump file and return the number of atoms, the supercell specification and a subset of atom data"""
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>3 and words[3] == 'ATOMS' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex+1):
        f.readline()
    num_atoms = int(f.readline().split()[0])
    f.close()
    # Get supercell shape
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'BOX' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    supercell_shape = np.zeros((3,3), dtype=float)
    f = open(file)
    for s in range(lineIndex+1):
        f.readline()
    for s in range(3):
        words = f.readline().split()
        if len(words)==3:
            supercell_shape[s,:] = [words[0], words[1], words[2]]
        elif len(words)==2:
            supercell_shape[s,:2] = [words[0], words[1]]
    f.close()
    supercell = lammps_dump_to_ppd(supercell_shape)
    # Get atom data
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'ATOMS' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    words = f.readline().split()
    fields = 6
    x_in = None; y_in = None; z_in = None; type_in = None; pe_in = None; id_in = None
    for s in range(len(words)):
        if words[s] == 'x':
            x_in = s-2
        elif words[s] == 'y':
            y_in = s-2
        elif words[s] == 'z':
            z_in = s-2
        elif words[s] == 'type':
            type_in = s-2
        elif words[s] == 'c_peatom':
            pe_in = s-2
        elif words[s] == 'id':
            id_in = s-2
    atom_data = np.zeros((num_atoms,fields), dtype=float)
    for s in range(num_atoms):
        words = f.readline().split()
        if x_in is not None:
            atom_data[s,0] = words[x_in]
        if y_in is not None:
            atom_data[s,1] = words[y_in]
        if z_in is not None:
            atom_data[s,2] = words[z_in]
        if type_in is not None:
            atom_data[s,3] = words[type_in]
        if pe_in is not None:
            atom_data[s,4] = words[pe_in]
        if id_in is not None:
            atom_data[s,5] = words[id_in]
    f.close()
    return num_atoms, supercell, atom_data
    
def read_lammps_input(file):
    """Read a lammps input file"""
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>1 and words[1] == 'atoms' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    num_atoms = int(f.readline().split()[0])
    f.close()
    
    box_cell = np.zeros((3,3), dtype=float)
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>2 and words[2] == 'xlo' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex):
        f.readline()
    for s in range(3):
        words = f.readline().split()
        box_cell[s,0] = float(words[0])
        box_cell[s,1] = float(words[1])
    f.close()
    supercell = lammps_box_to_ppd(box_cell)
    
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>3 and words[3] == 'xy' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    if found_line:
        f = open(file)
        for s in range(lineIndex):
            f.readline()
        words = f.readline().split()
        for s in range(3):
            box_cell[s,2] = float(words[s])
        f.close()
        
    atom_data = np.zeros((num_atoms,6), dtype=float)
    f = open(file)
    found_line = False
    for l, line in enumerate(f):
        words = line.split()
        if (len(words)>0 and words[0] == 'Atoms' and found_line == False):
            lineIndex = l
            found_line = True
    f.close()
    f = open(file)
    for s in range(lineIndex+2):
        f.readline()
    for s in range(num_atoms):
        words = f.readline().split()
        atom_data[s,0] = float(words[2]); atom_data[s,1] = float(words[3]); atom_data[s,2] = float(words[4])
        atom_data[s,3] = float(words[1])
        atom_data[s,5] = float(words[0])
    f.close()    
    return num_atoms, supercell, atom_data
    
    
def read_poscar(file, contcar=False):
    """Read a vasp POSCAR or CONTCAR and return the number of atoms, the supercell specification and atom data for positions and types"""
    supercell = np.zeros((3,3))
    header = 8
    sd = False
    direct = False
    f = open(file, 'r')
    f.readline()
    scale = float(f.readline())
    
    for s in range(3):
        words = f.readline().split()
        for t in range(3):
            supercell[s,t] = scale * float(words[t])
    if contcar:
        f.readline()
        header = 9
    words = f.readline().split()
    n_types = len(words)
    n_atoms = np.zeros(n_types, dtype=int)
    for i, word in renumerate(words):
        n_atoms[i] = int(word)
    line = f.readline()
    if line[0] == 'S' or line[0] == 's':
        sd = True
    line = f.readline()
    if line[0] == 'D' or line[0] == 'd':
        direct = True
        
    atom_data = np.zeros((n_atoms,4))   
    if sd:
        atom_type = np.zeros(n_atoms)
    type_count = 1 
    for i in range(n_atoms):
        words = f.readline().split()
        for s in range(3):
            if direct:
                atom_pos[i,:3] = atom_pos[i,:3] + float(words[s])*supercell[s,:]
            else:
                atom_pos[i,s] = float(words[s])
        atom_type[i] = 1
        cons == 'T T T'
        if sd:
            cons = (words[3] + ' ' + words[4] + ' ' + words[5])
        if cons == 'T T T':
            atom_data[i,3] = type_count
        elif cons == 'F F F':
            atom_data[i,3] = type_count + num_types
        if i == np.sum(n_atoms[:type_count]):
            type_count = type_count + 1
    return np.sum(n_atoms), supercell, atom_data

def write_print_file(r, bond, filename, r_atom=0.15, r_bond=0.05):
    fo = open(filename, 'w')
    fo.write('atoms=[\n')
    for i in range(np.shape(r)[0]):
        fo.write('[ ')
        for s in range(3):
            fo.write(str(r[i,s])+ ', ')
        fo.write(str(r_atom) + ' ]')
        if i<np.shape(r)[0]-1:
            fo.write(',\n')
        else:
            fo.write('\n')
    fo.write('];\n')
    fo.write('\n')
    fo.write('bonds=[\n')
    for i in range(np.shape(bond)[0]):
        fo.write('[ ')
        for s in range(2):
            fo.write(str(bond[i,s])+ ', ')
        fo.write(str(r_bond) + ' ]')
        if i<np.shape(r)[0]-1:
            fo.write(',\n')
        else:
            fo.write('\n')
    fo.write('];\n')
    fo.close()
    

def write_lammps(supercell, atom_pos, atom_type=None, filename='lammps.txt', num_types=1):
    """Write out a supercell in Lammps format"""
    fo = open(filename,'w')
    header = '#Lammps coordinate file'
    fo.write(header)
    fo.write('\n')
    fo.write(str(np.shape(atom_pos)[0]) + ' atoms\n')
    fo.write('\n')
    fo.write(str(num_types) + ' atom types\n')
    fo.write('\n')
    fo.write('0.0 ' + str(supercell[0,0]) + ' xlo xhi\n')
    fo.write('0.0 ' + str(supercell[1,1]) + ' ylo yhi\n')
    fo.write('0.0 ' + str(supercell[2,2]) + ' zlo zhi\n')
    if abs(supercell[1,0]) + abs(supercell[2,0]) + abs(supercell[2,1]) > CRYSTALTOOLS_TOL:
        fo.write(str(supercell[1,0]) + ' ' + str(supercell[2,0]) + ' ' + str(supercell[2,1]) + ' xy xz yz\n')
    fo.write('\n')
    fo.write('Atoms\n')
    fo.write('\n')
    count = 1
    for i in range(np.shape(atom_pos)[0]):
        fo.write(str(count) + ' ')
        if atom_type is not None: 
            fo.write(str(int(atom_type[i])) + ' ') 
        else:
            fo.write('1 ') 
        fo.write(str(atom_pos[i,0]) + ' ' + str(atom_pos[i,1]) + ' ' + str(atom_pos[i,2]) + '\n')
        count = count + 1
    fo.flush()
    fo.close()
    return