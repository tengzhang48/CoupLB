# CoupLB: Coupled Lattice Boltzmann Package for LAMMPS

**CoupLB** is a high-performance, native LAMMPS package that implements the Lattice Boltzmann Method (LBM) for fluid-structure interaction simulations. It is designed to run seamlessly alongside Molecular Dynamics (MD) using the Immersed Boundary Method (IBM).

## 1. Overview

CoupLB provides a fully parallelized fluid solver that operates on a fixed Eulerian grid while interacting with Lagrangian MD particles. It is implemented as a standard LAMMPS `fix` class.

### Key Features
* **Lattices:** Supports **D2Q9** (2D) and **D3Q19** (3D) standard lattices.
* **Collision:** BGK collision operator with **Guo forcing scheme** for accurate body forces.
* **Coupling:** Immersed Boundary Method (IBM) using **Roma** (3-point) or **Peskin** (4-point) delta kernels.
* **Parallelism:**
    * Spatial domain decomposition via MPI (matches LAMMPS domains).
    * Ghost-cell communication with optimized reverse-force exchange.
    * Hybrid MPI + OpenMP threading support.
* **Boundaries:** Half-way bounce-back, moving velocity walls, and periodic boundaries.
* **I/O:** Native VTK output (`.vti` + `.pvd`) for ParaView and binary checkpointing.

---

## 2. Package Structure

The source files should be placed in `src/COUPLB/`.

| File | Description |
| :--- | :--- |
| **fix_couplb.cpp / .h** | Main `Fix` class. Handles time integration, I/O scheduling, and LAMMPS interface. |
| **couplb_lattice.cpp / .h** | Grid memory management, macroscopic field computation ($\rho, \mathbf{u}$), and lattice constants. |
| **couplb_collision.h** | Implementation of the BGK collision step and external forcing. |
| **couplb_streaming.h** | Stream-collide algorithm, MPI ghost exchanges, and boundary conditions. |
| **couplb_boundary.h** | Primitives for setting wall types and velocities. |
| **couplb_ibm.h** | IBM interpolation (velocity to particle) and spreading (force to fluid). |
| **couplb_io.h** | Handles VTK (.vti) writing, PVD time-series generation, and binary checkpoints. |

---

## 3. Compilation Instructions

CoupLB requires a C++11 compliant compiler and an MPI library. It has no external dependencies (e.g., no FFTW required).

### Option A: Using CMake (Recommended)

1.  Create the file `src/COUPLB/CMakeLists.txt` with the following content:
    ```cmake
    # src/COUPLB/CMakeLists.txt
    if(NOT PKG_COUPLB)
      return()
    endif()

    file(GLOB COUPLB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    file(GLOB COUPLB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

    list(APPEND LAMMPS_SOURCES ${COUPLB_SOURCES})
    list(APPEND LAMMPS_HEADERS ${COUPLB_HEADERS})
    ```

2.  Build LAMMPS with the package enabled:
    ```bash
    mkdir build && cd build
    cmake ../cmake -DPKG_COUPLB=yes -DBUILD_MPI=yes
    make -j 8
    ```

### Option B: Using Traditional Make

1.  Navigate to the LAMMPS source directory:
    ```bash
    cd src
    ```
2.  Install the package and compile:
    ```bash
    make yes-couplb
    make mpi
    ```

---

## 4. Usage Syntax

The simulation is activated using the `fix` command in your LAMMPS input script.

```lammps
fix ID group couplb Nx Ny Nz viscosity rho0 [keywords...]
Required ArgumentsArgumentTypeDescriptionIDstringUnique ID for the fix (e.g., 1, fluid).groupstringGroup of atoms to couple (use all if no coupling).couplbstyleThe style name.Nx, Ny, NzintNumber of lattice nodes in X, Y, Z directions.viscosityfloatKinematic viscosity ($\nu$) in lattice units.rho0floatRest density of the fluid.Optional KeywordsKeywordArgumentsDefaultDescriptionnsubN1Number of LBM sub-steps per single MD timestep.gravitygx gy gz0 0 0Constant body force applied to the fluid.dxvalueLx/NxPhysical spacing of grid nodes. Used for unit conversion.wall_ylo hi0 0Y-boundary type: 0=periodic, 1=no-slip wall, 2=moving wall.wall_zlo hi0 0Z-boundary type (3D only).wall_velvx vy vz0 0 0Velocity vector for moving walls (type 2).kerneltyperomaIBM Delta function: roma (3-point) or peskin4 (4-point).vtkN prefix0 (off)Write VTK snapshot every N steps to prefix_step.vti.checkpointN prefix0 (off)Write binary restart file every N steps.restartprefixnoneLoad fluid state from a checkpoint at initialization.outputN file0 (off)Write simple ASCII velocity profiles (legacy format).check_everyN0Frequency to check stability (Mach number, density limits).5. Output and VisualizationVTK OutputIf vtk is enabled, CoupLB writes:.vti files: One file per frame containing the full volumetric data (Density, Velocity, Force)..pvd file: A ParaView Data file that links the .vti files to simulation time.To visualize: Open the .pvd file in ParaView. It will automatically load the time series.Log OutputCoupLB prints diagnostics to the LAMMPS screen/log:Ma: Maximum Mach number (Warning if > 0.3, Error if > 0.5).Mass/Mom: Total fluid mass and momentum conservation checks.6. Checkpointing and RestartsCoupLB fluid state is stored separately from LAMMPS atom data. To restart a simulation, you must restore both the particles and the fluid.Saving StateUse the checkpoint keyword to save binary dumps of the distribution functions ($f_i$):Code snippetfix 1 all couplb ... checkpoint 5000 restarts/fluid_backup
This generates files like restarts/fluid_backup.10000.0.clbk, restarts/fluid_backup.10000.1.clbk, etc.Loading StateTo restart, use the restart keyword pointing to the prefix of the saved files.Important: The fluid checkpoint does not automatically update the LAMMPS global timestep. You must manually sync them using reset_timestep.Example Restart Script:Code snippet# 1. Read particle data
read_restart      restart.10000

# 2. Define lattice with 'restart' keyword
#    Point to the prefix used during the previous run
fix 1 all couplb 100 50 50 0.01 1.0 restart restarts/fluid_backup

# 3. Sync timestep (Crucial for PVD and file numbering)
reset_timestep    10000

# 4. Continue simulation
run               5000
7. Example Input ScriptCode snippetunits           lj
atom_style      atomic
boundary        p p p

region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 random 100 34564 box

mass            1 1.0
pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0

# CoupLB Fix: 3D, 32^3 grid
fix 1 all couplb 32 32 32 0.1 1.0 &
    gravity 0.001 0.0 0.0 &
    vtk 100 dumps/flow &
    checkpoint 1000 restarts/fluid &
    check_every 100

timestep        0.005
run             2000
