# CoupLB:Coupled Lattice Boltzmann for LAMMPS — Complete Source
CoupLB — Coupled Lattice Boltzmann method for fluid-structure
interaction in LAMMPS. Supports 2D (D2Q9) and 3D (D3Q19), MPI domain
decomposition for multi-node HPC, OpenMP threading, BGK collision with
Guo forcing, half-way bounce-back walls, and immersed boundary method
coupling with Roma / Peskin delta kernels.

Package directory: src/COUPLB/
Files
FilePurposecouplb_lattice.hGrid, descriptors, macroscopic fields, diagnosticscouplb_collision.hBGK collision with Guo forcingcouplb_streaming.hPull streaming, bounce-back, MPI exchangecouplb_boundary.hWall setup utilitiescouplb_ibm.hIBM delta functions, interpolation, force spreadingfix_couplb.hFix class declarationfix_couplb.cppFix class implementation
