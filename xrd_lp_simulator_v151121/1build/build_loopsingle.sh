#!/bin/bash
#$ -pe smp.pe 8 # min2, max32

##  USAGE: ./build_loopsingle.sh [loop radius]  ##

folder=output_files 

# Sample arguments: int a on 2nd order plane
a=3.232
c=5.147

           #  VV no. cells in x (orthorhombic) for 10^7 atoms: 136 cells (4 per cell!)
Nx=220     # 220Each Nx is 1/2 a *whole* unit cell, use 2*Nx for desired number of cells!
Ny=110     # 110Each Ny is y. Use as normal.
Nz=110     # 110Each Nz is 1/2 a *whole* unit cell BUT! c ~ 2*a, so just use as normal (c is actually 1/2*c!)

rx=`echo $a $Nx | awk '{print $1*$2/2.0}'`        # echo reads a and Nx into a string, awk prints $1 (first argument: a) * (2nd arg: Nx) / 2 (middle of lattice).
ry=`echo $a $Ny | awk '{print 1.73*$1*$2/2.0}'`
rz=`echo $c $Nz | awk '{print $1*$2/2.0}'`

kind=a       # a is prismatic, c2+p is basal (NOT c/2+p: FILEPATH ERROR!)
character=i  # (i)nterstitial or (v)acancy
plane=1st    # 1st or 2nd order (101b0 or 112b0)
variant=1    # selects one of three symmetrical planes

radii=$1                    # Radius input: inline argument!!

for R in "${radii[@]}"; do    # R is loop radius (circular)

	filename=${folder}/single_${kind}_${character}_${plane}_${variant}_r_${R}.txt
	echo $filename
	python ./create_dislocation_simple.py $a $c $Nx $Ny $Nz $rx $ry $rz $R $kind $character $plane $variant $filename
done
