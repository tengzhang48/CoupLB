# CoupLB: Coupled Lattice Boltzmann for LAMMPS

**CoupLB** — High-performance coupled Lattice Boltzmann method for fluid-structure interaction simulations in LAMMPS. Supports 2D (D2Q9) and 3D (D3Q19) lattices with native MPI domain decomposition for multi-node HPC, OpenMP threading, BGK collision with Guo forcing scheme, half-way bounce-back boundaries, and immersed boundary method (IBM) coupling with Roma/Peskin delta kernels.

## Package Structure

Located in LAMMPS package directory: `src/COUPLB/`

| File | Purpose |
|------|---------|
| `couplb_lattice.h` `couplb_lattice.cpp`| Grid management, lattice descriptors, macroscopic field computation, diagnostics |
| `couplb_lattice.cpp` | Grid management, lattice descriptors, macroscopic field computation, diagnostics |
| `couplb_collision.h` | BGK collision operator with Guo forcing scheme |
| `couplb_streaming.h` | Pull streaming implementation, bounce-back boundaries, MPI halo exchange |
| `couplb_boundary.h` | Wall boundary setup utilities |
| `couplb_ibm.h` | IBM delta functions (Roma/Peskin), velocity interpolation, force spreading |
| `fix_couplb.h` | LAMMPS `Fix` class declaration |
| `fix_couplb.cpp` | LAMMPS `Fix` class implementation |

## Key Features

✅ **Native LAMMPS Integration**  
- Implemented as a proper LAMMPS `fix` with seamless domain decomposition  
- No external coupling overhead or synchronization issues

✅ **Parallel Safety**  
- Reverse force exchange mechanism prevents force loss at subdomain boundaries  
- Thread-safe OpenMP parallelization with proper reductions  
- MPI-optimized halo exchanges with wall-aware skipping

✅ **Numerical Robustness**  
- Delta function sqrt argument clamping prevents NaNs from FP rounding  
- Density clamping with diagnostic counters for instability detection  
- Runtime validation of force scaling and Mach number limits

✅ **Physical Accuracy**  
- Guo forcing scheme preserves Galilean invariance  
- Wall-aware IBM interpolation (handles stationary/moving walls correctly)  
- Weight normalization in force spreading ensures conservation

## Validation

Validated against analytical solutions with <1% error:
- 2D/3D Poiseuille flow (gravity-driven)
- 2D Couette flow (moving walls)
- Mass/momentum conservation checks at every timestep

## Installation

As a standard LAMMPS package:
```bash
cd lammps/src
make yes-couplb    # or add to packages/ folder manually
make mpi           # or your preferred build target
