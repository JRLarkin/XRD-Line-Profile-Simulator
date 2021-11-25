#-----------------------------------------------------------#
# Scripts for Building & Generating Damaged a-Zr Supercells
# Version 151121 (15th November 2021)
# C.P. Race, J. Larkin
# Department of Materials, The University of Manchester
#-----------------------------------------------------------#

This collection of Python programs and Bash scripts are designed to createa virtual atomistic supercell of zirconium atoms in an alpha-HCP
configuration. The programs then produce a model of irradiation-induced defect loops (typically on the order of ~1-15nm in diameter).
Files are provided for an effective thermodynamic relaxation in LAMMPS molecular dynamics suite, and then a step is included for use
on a computer cluster (HTCondor at The University of Manchester in this case) which calculates matrices of pairwise separations for all atoms
in the lattice. This generates a master histogram of all weighted contributions from each neighbour pair. The master histogram is passed through
the Debye scattering equation to produce a theoretical line-profile as would be observed in x-ray powder diffractometry.

The motivation is to explore the emergence of 'shoulder features' surrounding Bragg diffraction peaks arising from radiation-damaged lattices.
Initial Python and bash scripts are courtesy of Dr. C.P. Race, and have been modified and tailored for this specific use by J. Larkin.


## GETTING STARTED ##
The process is divided into four steps, mainly using python (.py) and bash (.sh) files.
A local unix-based terminal is recommended for running files locally, as some require input arguments from the user.
GitBash or a similar unix-based terminal is recommended for use on Windows PCs.

~~

## STEP 1: BUILDING LATTICES ##

The '1build' folder contains the following assets;

gbtk				- this is a series of python packages that "create_dislocation_simple.py" refers to when building a crystal supercell.
output_files			- atom co-ordinate files (.txt) are sent here on completion.
build_loopsingle.sh		- this is the bash script that should be run by the user.
create_dislocation_simple.py	- the main python code that calls parameters from "build_loopsingle.sh" and writes the output atom co-ordinate file.

To use, first open build_loopsingle.sh in a text editor and modify the following parameters;
a, c		- lattice parameters. 3.232, 5.147 for zirconium.
Nx, Ny, Nz	- number of orthorhombic unit cells in each x,y,z direction. x-dimension is approximately half the size of y & z, so double for roughly cubic cell.
kind		- input either "a" for a prismatic <a>-type loop, or "c2+p" for a basal <c>-type loop.
character	- input either "i" for interstitial loop, or "v" for vacancy loop.
plane		- input either "1st" for 1st-order loop in (10-10) plane, or "2nd" for 2nd-order loop in (11-20) plane.
variant		- HCP unit cell has three symmetrical planes, input either "1", "2" or "3" to specify.

Then, run the bash script with the syntax "./build_loopsingle.sh [loop radius]". [loop radius] is a float.
Defect loop will be placed in the cell centred around the atom closest to the exact centre of the cell.

The output file will be in the following format;
single_[kind]_[character]_[plane]_1_r_[loop radius].txt
For example - "single_a_i_1st_1_r_10.txt".

~~

## STEP 2: PERFORMING LAMMPS RELAX ##

The raw .txt file must now be minimised. The "2relax" folder should be divided into subfolders, each with its own set of the following assets for a LAMMPS minimisation;

in.HCP.lmp		- LAMMPS input file. Ensure that "read_data" points to the correct file. "dump 2" creates a co-ordinate file of the final configuration. "dump 1" not needed.
min_pll_bash		- bash script used to submit the files to CSF-3.
[output file].txt	- atom co-ordinate file generated using Step 1.
Zr_1.eam.fs		- empirical atom potential file referred to by "in.HCP.lmp".

This folder is designed to run on the Computational Shared Facility (CSF-. 3) A subfolder with all of these assets should be sent to the "scratch/" directory for each supercell.
Then, submission to CSF-3 is via the following command; "qsub min_pll_bash".
This will run a "fire" style minimisation to reach a minimum energy potential of 1e-6 eV. Typically takes several ~1000 iterations.

The output file will be in the following format;
relaxed0000.min
For example - "relaxed2917.min".
The number (2917) is the amount of iterations taken to minimise the cell. THIS SHOULD BE REMOVED FOR THE NEXT STEP! - e.g. relaxed2917.min --> relaxed.min.

The process will also generate a command-line log ("cmdlog"), as well as ".e" and ".o" text files. If the latter two files are empty, this indicates a successful minimisation.

~~

## STEP 3: CALCULATING PAIRWISE HISTOGRAMS ##

The minimised atom co-ordinates can now be used to generate a series of histograms with the intensity contributions from each pair of atoms, assuming single-slit diffraction.
A folder with the entire contents of "condor_jobfiles" (inside "3condor") should be uploaded to the "scratch/" directory of UoM's CONDOR computer cluster. The assets are;

build_jobs.py			- this divides the job into a series of smaller jobs, to be sent to separate computers. "n_per_job" line decides the number of atoms for each smaller job.
checkrun.sh			- during or following a run, this can be used to determine the number of remaining jobs. "Found n/N omissions" refers to jobs still awaiting completion.
collate_data.py			- ONLY AFTER all jobs have completed, use this to collate all histograms into one master histogram.
delete_histograms.sh		- deletes all histograms and "joblist.txt" for quick reset.
relaxed.min			- atom co-ordinate file generated using Step 2. IMPORTANT: make sure numerical suffix is removed from "relaxed000.min" before use.
resubmit.sh			- in case of any jobs 'falling over', use this to resubmit individual jobs (shouldn't be needed).
run_jobs.sh			- once "build_jobs.py" has constructed all necessary files, use this to send all individual jobs to CONDOR.
xrd_debye_blurred_sphere.py	- python script used in all jobs to cut out a sphere of atoms within the supercell and calculate pairwise interactions.


ORDER OF FILE USE;
1: "python build_jobs.py"
2: "./run_jobs.sh"
3: "./checkrun.sh"
4: "python collate_data.py"

The output file will be in the following format;
histogram_total.txt

~~

## STEP 4: GENERATING THEORETICAL LINE PROFILE ##

Finally, histogram_total.txt can be downloaded from CONDOR and handled locally. "4profile" contains a python script that uses the Debye Scattering Equation to calculate I(Q).
The output file is the XRD line profile: "profile_total.txt". Data are in two columns - intensity I (AU), and Q. This can be opened and modified with suitable graphical plotting software.

Line profiles are currently being handled in CMWP-fit v.201225 (T. Ungar, 2020).


#-----------------------------------------------------------#
For additional information contact;

Jake Larkin | PhD Student | The University of Manchester | MIDAS Research Group
Room 7.001, Royce Institute Hub Building | Department of Materials | The University of Manchester | M13 9PL
Tel: +44 (0)161 306 4838 | Email: jake.larkin@postgrad.manchester.ac.uk
#-----------------------------------------------------------#