# CoupLB: Coupled Lattice Boltzmann Package for LAMMPS

**CoupLB** is a native LAMMPS package that implements the Lattice Boltzmann Method (LBM) coupled with Molecular Dynamics via the Immersed Boundary Method (IBM). It runs as a standard LAMMPS `fix` with full MPI parallelism.

---

## 1. Features

- **Lattices:** D2Q9 (2D) and D3Q19 (3D)
- **Collision:** BGK with Guo forcing scheme
- **Fluid–structure coupling:** Immersed Boundary Method with Roma (3-point) or Peskin (4-point) delta kernels
- **Boundaries:** Half-way bounce-back walls, moving velocity walls (Couette), periodic
- **Parallelism:** MPI domain decomposition (mirrors LAMMPS proc grid), optional OpenMP threading
- **I/O:** Single-file VTK output with PVD time series for ParaView, binary checkpoint/restart

---

## 2. Source Files

All files go in `src/COUPLB/` (or directly in `src/`).

| File | Description |
|:-----|:------------|
| `fix_couplb.cpp` / `.h` | Main fix class: time integration, keyword parsing, I/O scheduling |
| `couplb_lattice.cpp` / `.h` | Grid storage (SoA layout), macroscopic field computation, lattice descriptors |
| `couplb_collision.h` | BGK collision with Guo source term |
| `couplb_streaming.h` | Streaming step, MPI ghost exchange (distributions, velocities, forces) |
| `couplb_boundary.h` | Wall type and velocity assignment |
| `couplb_ibm.h` | IBM interpolation (fluid → particle) and force spreading (particle → fluid) |
| `couplb_io.h` | VTK writer (gathered to rank 0), PVD time series, binary checkpoint I/O |

---

## 3. Compilation

CoupLB requires a **C++14** compliant compiler and an MPI library. No external dependencies.

### Option A: Traditional Make

```bash
cd lammps/src
# Copy CoupLB files into src/ (or use make yes-couplb if installed as a package)
make mpi
```

### Option B: CMake

Create `src/COUPLB/CMakeLists.txt`:

```cmake
if(NOT PKG_COUPLB)
  return()
endif()

file(GLOB COUPLB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB COUPLB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

list(APPEND LAMMPS_SOURCES ${COUPLB_SOURCES})
list(APPEND LAMMPS_HEADERS ${COUPLB_HEADERS})
```

Then build:

```bash
mkdir build && cd build
cmake ../cmake -DPKG_COUPLB=yes -DBUILD_MPI=yes
make -j 8
```

---

## 4. Usage

### Syntax

```
fix ID group couplb Nx Ny Nz viscosity rho0 [keywords...]
```

### Required Arguments

| Argument | Type | Description |
|:---------|:-----|:------------|
| `Nx Ny Nz` | int | Number of LB nodes in each direction. Must satisfy `Nx * dx = Lx` (similarly for y, z). |
| `viscosity` | float | Kinematic viscosity ν in lattice units. Sets τ = ν / cs² + 0.5. |
| `rho0` | float | Reference fluid density. |

### Optional Keywords

| Keyword | Arguments | Default | Description |
|:--------|:----------|:--------|:------------|
| `nsub` | N | 1 | LBM sub-steps per MD step. **Incompatible with IBM particles.** |
| `gravity` | gx gy gz | 0 0 0 | Constant body force on fluid (lattice units). |
| `dx` | value | Lx/Nx | Physical grid spacing for unit conversion. Must satisfy Nx × dx = Lx. If omitted, computed automatically. |
| `wall_y` | lo hi | 0 0 | Y-boundary type: 0 = periodic, 1 = no-slip wall, 2 = moving wall. |
| `wall_z` | lo hi | 0 0 | Z-boundary type (3D only). |
| `wall_vel` | vx vy vz | 0 0 0 | Velocity for type-2 (moving) walls, in lattice units. |
| `kernel` | type | roma | IBM delta function: `roma` (3-point) or `peskin4` (4-point). |
| `vtk` | N prefix | off | Write VTK field snapshot every N steps. |
| `checkpoint` | N prefix | off | Write binary checkpoint every N steps. |
| `restart` | prefix | none | Load fluid state from checkpoint at initialization. |
| `output` | N file | off | Write y-averaged ASCII velocity profiles every N steps. |
| `check_every` | N | 0 | Stability diagnostics frequency (Mach number, mass, momentum). |

---

## 5. Unit Conversion

CoupLB operates internally in lattice units (dx = 1, dt = 1, cs² = 1/3). The coupling to LAMMPS physical units is controlled by two scale factors:

```
vel_scale   = dx_phys / dt_lbm
force_scale = rho0 × dx_phys^(D+1) / dt_lbm²
```

where `dt_lbm = timestep / nsub` and `D` is the spatial dimension.

The `dx` keyword sets `dx_phys`. If omitted, it defaults to `Lx / Nx`. VTK output writes velocities and forces in **physical (LAMMPS) units**, scaled by `vel_scale` and `force_scale` respectively. Density is written in lattice units.

---

## 6. Parallel Decomposition

CoupLB mirrors the LAMMPS processor grid. Use the `processors` command to control the decomposition:

```
processors 2 4 2    # 16 MPI ranks: 2×4×2
```

**Constraints:**

- Each grid dimension must be evenly divisible by the number of processors in that direction: `Nx % px == 0`, etc.
- **Minimum cells per rank:** 3 for Roma kernel, 4 for Peskin4 kernel (per direction). Below this, IBM stencils get clipped.
- For wall-bounded flows, split along the wall-normal direction first (it's usually the largest).

| Grid | Good decompositions | Local grid per rank |
|:-----|:-------------------|:-------------------|
| 32×32×32 | 2×2×2 (8) | 16×16×16 |
| 64×64×64 | 4×4×4 (64) | 16×16×16 |
| 8×32×8 | 1×4×1 (4) | 8×8×8 |
| 16×64×16 | 2×4×2 (16) | 8×16×8 |

---

## 7. Output and Visualization

### VTK Output

When `vtk N prefix` is enabled, CoupLB writes:

- **`prefix_STEP.vti`** — Single VTK ImageData file containing the full 3D field (density, velocity, force, node type). All ranks gather data to rank 0 for writing. Suitable for grids up to ~128³.
- **`prefix.pvd`** — ParaView collection file linking all `.vti` files with simulation time. Updated after each VTK dump, so it remains valid even if the run crashes.

**To visualize:** Open the `.pvd` file in ParaView → Apply. Use the time slider to scrub through frames.

Fields written:

| Field | Components | Units |
|:------|:-----------|:------|
| `rho` | 1 (scalar) | Lattice units |
| `velocity_phys` | 3 (vector) | LAMMPS physical units |
| `force_phys` | 3 (vector) | LAMMPS physical units |
| `type` | 1 (scalar) | 0 = fluid, 1 = no-slip wall, 2 = moving wall |

### ASCII Profiles

When `output N file` is enabled, CoupLB writes y-averaged profiles (rho, ux, uy) at each y-index, useful for validating Poiseuille and Couette flows.

### Log Diagnostics

When `check_every N` is enabled, CoupLB prints to the LAMMPS log:

- **Ma:** Maximum Mach number. Warning at Ma > 0.3, error at Ma > 0.5.
- **mass:** Total fluid mass (should be conserved).
- **mom:** Total fluid momentum vector.
- **density clamps:** Count of nodes where density was clamped (indicates numerical trouble).

---

## 8. Checkpointing and Restarts

CoupLB fluid state is stored **separately** from LAMMPS atom data. A full restart requires restoring both.

### Writing Checkpoints

```
fix 1 all couplb ... checkpoint 10000 ckpt/fluid
```

This writes binary files `ckpt/fluid.0.clbk`, `ckpt/fluid.1.clbk`, ... (one per MPI rank). Each checkpoint **overwrites** the previous one at the same prefix.

The files contain the full distribution function f_i for exact restart. File format: magic number, version, step, grid metadata, then the raw f[] array.

### Loading a Checkpoint

```
fix 1 all couplb ... restart ckpt/fluid
reset_timestep 10000
```

**Important notes:**

- The `restart` keyword only loads the fluid state. Use `read_restart` or `read_data` separately for particles.
- `reset_timestep` is required to sync the LAMMPS timestep with the fluid state. Without it, output file numbering will be incorrect.
- The checkpoint **must** be loaded with the same processor count and grid decomposition as when it was written.

### Example Restart Script

```lammps
# Restore particles
read_restart particles.10000

# Define fix with restart keyword
fix 1 all couplb 64 64 64 0.1 1.0 &
    restart ckpt/fluid &
    vtk 5000 flow_vtk &
    checkpoint 10000 ckpt/fluid

# Sync LAMMPS timestep to checkpoint step
reset_timestep 10000

# Continue
run 10000
```

---

## 9. Examples

### 2D Poiseuille Flow

```lammps
units           lj
dimension       2
boundary        p f p
atom_style      atomic

region          box block 0 8 0 32 -0.5 0.5
create_box      1 box
create_atoms    1 single 4 0.5 0          # Dummy atom near wall

group           noatoms empty
mass            1 1.0
pair_style      none
timestep        1.0

fix flow noatoms couplb 8 32 1 0.1 1.0 &
    wall_y 1 1 &
    gravity 5e-6 0.0 0.0 &
    output 5000 poiseuille_2d.dat &
    check_every 2000

run 50000
```

Analytical solution: u_max = g H² / (8ν) = 5e-6 × 1024 / 0.8 = 6.4e-3

### 3D Poiseuille Flow (Parallel)

```lammps
units           lj
dimension       3
boundary        p f p
atom_style      atomic

processors      1 4 1

region          box block 0 8 0 32 0 8
create_box      1 box
create_atoms    1 single 4 0.5 4
mass            1 1.0

group           noatoms empty
pair_style      none
timestep        1.0

fix flow noatoms couplb 8 32 8 0.1 1.0 &
    wall_y 1 1 &
    gravity 5e-6 0.0 0.0 &
    output 5000 poiseuille_3d.dat &
    vtk 10000 poiseuille_vtk &
    check_every 2000

run 50000
```

### Sphere Drag (IBM Validation)

```lammps
units           lj
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map array

processors      2 2 2

variable        N equal 64
variable        dx equal 0.5
variable        L equal ${N}*${dx}
variable        hL equal ${L}/2

region          box block -${hL} ${hL} -${hL} ${hL} -${hL} ${hL}
create_box      1 box

# Create 12 icosahedron vertices at R=3.0 for sphere surface
# ... (see in.sphere_drag for full vertex list)

mass            1 0.0833333333
pair_style      none

group           sphere type 1
fix             push sphere addforce 4.1667e-4 0 0
fix             body sphere rigid/nve single

fix flow sphere couplb ${N} ${N} ${N} 0.1 1.0 &
    dx ${dx} &
    kernel roma &
    vtk 5000 sphere_vtk &
    check_every 1000

timestep        0.005
run             20000
```

---

## 10. Numerical Guidelines

### Stability

- **Relaxation time:** τ must be > 0.5. Values in [0.6, 1.5] give best accuracy. τ > 2.0 increases viscous damping of small features.
- **Mach number:** Keep Ma = u_max / cs < 0.3 for compressibility errors below ~10%. CoupLB errors out at Ma > 0.5.
- **Density:** Should remain close to rho0. Large density fluctuations indicate the flow is too compressible.

### Accuracy

- **Grid resolution:** For IBM-coupled spheres, D/dx ≥ 10 is recommended (D = sphere diameter).
- **IBM markers:** 12 icosahedron points give ~30% error vs Stokes drag. Use 42+ points (subdivided icosahedron) for < 5% error.
- **Domain size for periodic flows:** L/D ≥ 5 to limit periodic image effects. Apply Hasimoto correction for quantitative drag comparison.

### Performance

- VTK output gathers all data to rank 0. For grids larger than ~128³, output frequency should be kept low.
- Checkpoints write one file per rank with no gathering — fast and scalable.
- The LBM step dominates runtime (~91% in typical runs). MPI communication is < 1%.

---

## 11. Known Limitations

- **No turbulence models:** BGK collision only. For high-Re flows, consider MRT or entropic stabilization (not yet implemented).
- **Single-pass IBM forcing:** No iterative correction for no-slip enforcement. Expect O(10%) velocity errors near immersed boundaries with coarse marker spacing.
- **Fixed Lagrangian volume:** `dv_lag = 1.0` is hardcoded in the IBM spread function. For resolved particles, this should match the surface element area.
- **No adaptive mesh refinement.**
- **Checkpoint portability:** Checkpoints are tied to the same processor count and decomposition. Cannot restart on a different number of ranks.
