# CoupLB: Coupled Lattice Boltzmann Package for LAMMPS

**CoupLB** is a native LAMMPS package that implements the Lattice Boltzmann Method (LBM) coupled with Molecular Dynamics via the Immersed Boundary Method (IBM). It runs as a standard LAMMPS `fix` with full MPI parallelism.

---

## 1. Features

- **Lattices:** D2Q9 (2D) and D3Q19 (3D)
- **Collision:** BGK with Guo forcing scheme
- **Fluid–structure coupling:** Immersed Boundary Method with Roma (3-point) or Peskin (4-point) delta kernels
- **Boundaries:** No-slip (half-way bounce-back), moving velocity (Couette), free-slip (specular reflection), open (zero-gradient extrapolation), periodic
- **Sub-stepping:** Multiple MD steps per LBM step (`md_per_lb`) with momentum-conserving IBM coupling each sub-step
- **External forcing:** Constant or time-varying body forces via LAMMPS equal-style variables
- **Parallelism:** MPI domain decomposition (mirrors LAMMPS proc grid), optional OpenMP threading
- **I/O:** Single-file VTK output with PVD time series for ParaView, binary checkpoint/restart

---

## 2. Source Files

All files go in `src/COUPLB/` (or directly in `src/`).

| File | Description |
|:-----|:------------|
| `fix_couplb.cpp` / `.h` | Main fix class: time integration, keyword parsing, IBM coupling, I/O scheduling |
| `couplb_lattice.cpp` / `.h` | Grid storage (SoA layout), macroscopic field computation, lattice descriptors (D2Q9, D3Q19) |
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
fix ID group couplb Nx Ny Nz nu rho0 [keywords...]
```

### Required Arguments

| Argument | Type | Description |
|:---------|:-----|:------------|
| `Nx Ny Nz` | int | Number of LB nodes in each direction. Must satisfy `Nx × dx = Lx` (similarly for y, z). |
| `nu` | float | Kinematic viscosity in **LAMMPS units**. Converted internally to lattice units via ν\* = ν × dt\_LB / dx². The resulting τ = ν\*/cs² + 0.5 must be > 0.5. |
| `rho0` | float | Reference fluid density in **LAMMPS units**. Used as the initial uniform density and as the density reference for wall boundary conditions. |

### Optional Keywords

| Keyword | Arguments | Default | Description |
|:--------|:----------|:--------|:------------|
| `md_per_lb` | N | 1 | MD (LAMMPS) steps per LBM step. Enables sub-stepping: the LBM advances once every N MD steps, with fresh IBM interpolation and force spreading each sub-step. |
| `xi_ibm` | value | 1.0 | IBM relaxation factor (0 < ξ ≤ 1). Scales the penalty force that couples particles to the fluid. Values less than 1.0 soften the coupling. |
| `gravity` | gx gy gz | 0 0 0 | Body force acceleration on the fluid, in **LAMMPS units**. Each component can be a numeric constant or a LAMMPS equal-style variable reference (`v_varname`). Converted internally via g\* = g × dt\_LB² / dx. |
| `dx` | value | Lx/Nx | Physical grid spacing in **LAMMPS units**. Must satisfy Nx × dx = Lx. If omitted, computed automatically from the simulation box. |
| `wall_x` | lo hi | 0 0 | X-boundary type: 0 = periodic, 1 = no-slip, 2 = moving wall, 3 = free-slip, 4 = open. |
| `wall_y` | lo hi | 0 0 | Y-boundary type: 0 = periodic, 1 = no-slip, 2 = moving wall, 3 = free-slip, 4 = open. |
| `wall_z` | lo hi | 0 0 | Z-boundary type (3D only): 0 = periodic, 1 = no-slip, 2 = moving wall, 3 = free-slip, 4 = open. |
| `wall_vel` | vx vy vz | 0 0 0 | Velocity for type-2 (moving) walls, in **LAMMPS units**. Converted to lattice units internally. |
| `kernel` | type | roma | IBM delta function: `roma` (3-point) or `peskin4` (4-point). |
| `vtk` | N prefix | off | Write VTK field snapshot every N LAMMPS steps. |
| `checkpoint` | N prefix | off | Write binary checkpoint every N LAMMPS steps. |
| `restart` | prefix | none | Load fluid state from checkpoint at initialization. |
| `output` | N file | off | Write y-averaged ASCII velocity profiles every N LAMMPS steps. |
| `check_every` | N | 0 | Stability diagnostics frequency (Mach number, mass, momentum, density clamp count). |

---

## 5. Unit Conversion

All user-facing parameters are specified in **LAMMPS units**. CoupLB converts to lattice units internally (where dx\* = 1, dt\* = 1, cs² = 1/3).

### Key Relationships

The LBM timestep in LAMMPS units is:

```
dt_LB = timestep × md_per_lb
```

The two fundamental conversion scales are:

```
vel_scale   = dx / dt_LB           (lattice velocity → LAMMPS velocity)
force_scale = rho0 × dx^(D+1) / dt_LB²   (lattice force → LAMMPS force)
```

where D is the spatial dimension (2 or 3).

### Parameter Conversions

| Parameter | User provides (LAMMPS units) | Internal (lattice units) | Conversion |
|:----------|:----------------------------|:-------------------------|:-----------|
| Viscosity | ν | ν\* = ν × dt\_LB / dx² | Determines τ = ν\*/cs² + 0.5 |
| Gravity | g (acceleration) | g\* = g × dt\_LB² / dx | Body force per unit mass |
| Wall velocity | v\_wall | v\*\_wall = v\_wall / vel\_scale | Must satisfy Ma < 0.5 |
| IBM force | F\_particle (LAMMPS) | f\_grid = −F / force\_scale | Newton's third law |

### VTK Output Units

VTK files write fields in **LAMMPS physical units**, scaled by `vel_scale` and `force_scale` respectively. Density (`rho`) is written in lattice units (should remain close to `rho0`).

---

## 6. IBM Coupling

CoupLB applies IBM forces to all atoms in the fix group. The coupling operates in the `post_force` callback.

### 1:1 Coupling (md_per_lb = 1)

Each LAMMPS step, one LBM step is taken, followed by IBM coupling:

```
F_particle = ξ × m × (u_fluid − v_particle) / dt
F_grid     = −F_particle / force_scale
```

where ξ is `xi_ibm`, m is the particle mass, u\_fluid is the interpolated fluid velocity at the particle position (converted to LAMMPS units), and v\_particle is the particle velocity.

### Sub-stepped Coupling (md_per_lb > 1)

With N = `md_per_lb`, the LBM advances once on the first sub-step of each cycle, and IBM interpolation + spreading occurs every sub-step:

```
F_particle = ξ × m × (u_fluid − v_particle) / dt_MD
F_grid     = −F_particle / (N × force_scale)
```

The 1/N scaling on the grid force ensures exact momentum conservation over one LBM interval: the sum of particle impulses equals the negative of the total fluid impulse. Ghost force exchange is deferred to the last sub-step to prevent double-counting.

### Delta Kernels

- **Roma (3-point):** Support radius 1.5 lattice spacings. Requires ≥ 3 interior cells per rank per direction.
- **Peskin4 (4-point):** Support radius 2.0 lattice spacings. Requires ≥ 4 interior cells per rank per direction.

The interpolation handles wall nodes: fluid nodes (type 0) contribute velocity, no-slip walls (type 1) contribute zero velocity but are included in the weight sum, and moving walls (type 2) contribute their prescribed velocity.

---

## 7. External Forcing (Gravity)

The `gravity` keyword accepts constant values or LAMMPS equal-style variable references for time-varying body forces:

```lammps
# Constant gravity
fix lb all couplb 64 64 64 0.1 1.0 gravity 1e-4 0 0

# Oscillating body force via variable
variable freq equal 1.0
variable Gx equal 1e-4*sin(2*PI*v_freq*time)
fix lb all couplb 64 64 64 0.1 1.0 gravity v_Gx 0 0
```

Gravity is applied as a body force density: each fluid node receives f = g\* × ρ(x), where g\* is the lattice-unit acceleration and ρ is the local density. Variable-style gravity is re-evaluated at every LBM step.

---

## 8. Parallel Decomposition

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

### MPI Communication

CoupLB performs three types of ghost exchange along each dimension:

1. **Distribution exchange** — After streaming, distribution functions f\_i are communicated to ghost layers.
2. **Velocity exchange** — Before IBM interpolation, macroscopic fields (ρ, u) are communicated so IBM stencils near subdomain boundaries see correct values.
3. **Reverse force exchange** — After IBM spreading, forces deposited on ghost cells are accumulated back onto the owning rank's interior cells. This prevents silent force loss at subdomain boundaries.

---

## 9. Output and Visualization

### VTK Output

When `vtk N prefix` is enabled, CoupLB writes:

- **`prefix_STEP.vti`** — Single VTK ImageData file containing the full 3D field. All ranks gather data to rank 0 for writing. Suitable for grids up to ~128³.
- **`prefix.pvd`** — ParaView collection file linking all `.vti` files with simulation time. Updated after each VTK dump, so it remains valid even if the run crashes.

**To visualize:** Open the `.pvd` file in ParaView → Apply. Use the time slider to scrub through frames.

Fields written:

| Field | Components | Units |
|:------|:-----------|:------|
| `rho` | 1 (scalar) | Lattice units |
| `velocity_phys` | 3 (vector) | LAMMPS physical units |
| `force_phys` | 3 (vector) | LAMMPS physical units |
| `type` | 1 (scalar) | 0 = fluid, 1 = no-slip wall, 2 = moving wall, 3 = free-slip, 4 = open |

### ASCII Profiles

When `output N file` is enabled, CoupLB writes y-averaged profiles (rho, ux, uy) at each y-index, useful for validating Poiseuille and Couette flows. Columns: `step j y rho ux uy`, where `y` is the physical y-coordinate in LAMMPS length units, `rho` is the physical density (lattice density scaled by `rho`), and `ux`, `uy` are velocities in LAMMPS physical units (lattice velocity scaled by `vel_scale`).

### Log Diagnostics

When `check_every N` is enabled, CoupLB prints to the LAMMPS log:

- **Ma:** Maximum Mach number. Warning at Ma > 0.3, error at Ma > 0.5.
- **mass:** Total fluid mass (should be conserved in periodic flows).
- **mom:** Total fluid momentum vector.
- **density clamps:** Count of nodes where density was clamped to a minimum (indicates numerical trouble).

---

## 10. Checkpointing and Restarts

CoupLB fluid state is stored **separately** from LAMMPS atom data. A full restart requires restoring both.

### Writing Checkpoints

```
fix 1 all couplb ... checkpoint 10000 ckpt/fluid
```

This writes binary files `ckpt/fluid.0.clbk`, `ckpt/fluid.1.clbk`, ... (one per MPI rank). Each checkpoint **overwrites** the previous one at the same prefix.

The files contain the full distribution function f\_i for exact restart. File format: magic number (`0x434C424B`), version, step, grid metadata (D, Q, local dims, offsets, global dims), then the raw f[] array.

### Loading a Checkpoint

```
fix 1 all couplb ... restart ckpt/fluid
reset_timestep 10000
```

**Important notes:**

- The `restart` keyword only loads the fluid state. Use `read_restart` or `read_data` separately for particles.
- `reset_timestep` is required to sync the LAMMPS timestep with the fluid state. CoupLB prints a reminder when loading a checkpoint.
- The checkpoint **must** be loaded with the same processor count and grid decomposition as when it was written. Grid dimensions, offsets, and global sizes are all validated on load.
- On load, macroscopic fields are recomputed from the restored distributions and the swap buffer is initialized.

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

## 11. Time Integration Sequence

### LBM Step (once per `md_per_lb` MD steps)

```
1. apply_external_force    — add gravity body force to grid
2. compute_macroscopic     — compute ρ, u from f_i (with Guo correction)
3. collide                 — BGK collision with Guo source term
4. exchange                — MPI ghost exchange of distributions
5. stream                  — pull-scheme streaming with bounce-back
6. enforce_wall_ghost_fields — reset wall node ρ, u to prescribed values
7. clear_forces            — zero grid force arrays for next cycle
```

### IBM Coupling (every MD step)

For `md_per_lb = 1`:
```
1. enforce_wall_ghost_fields
2. compute_macroscopic (without Guo correction)
3. exchange_velocity
4. For each particle: interpolate → compute F → apply to particle → spread to grid
5. exchange_forces (reverse communication)
```

For `md_per_lb > 1`:
```
Sub-step 0 (first): LBM step, then IBM coupling with exchange_velocity
Sub-steps 1..N-1: IBM coupling only (re-interpolate with existing fluid field)
Sub-step N-1 (last): exchange_forces after spreading
```

---

## 12. Examples

### 2D Poiseuille Flow

```lammps
units           lj
dimension       2
boundary        p f p
atom_style      atomic

region          box block 0 8 0 32 -0.5 0.5
create_box      1 box
create_atoms    1 single 4 0.5 0

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

### 3D Channel with Oscillating Body Force

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

variable        G0 equal 5e-6
variable        freq equal 0.001
variable        Gx equal v_G0*sin(2*PI*v_freq*step)

fix flow noatoms couplb 8 32 8 0.1 1.0 &
    wall_y 1 1 &
    gravity v_Gx 0.0 0.0 &
    output 5000 oscillating_3d.dat &
    vtk 10000 osc_vtk &
    check_every 2000

run 50000
```

### IBM Coupling with Sub-stepping

```lammps
units           lj
dimension       3
boundary        p f p
atom_style      hybrid bpm/rotational dipole

# ... read data, define bonds, etc.

timestep        0.001

fix lb all couplb 64 48 32 0.1 1.26 &
    dx 0.625 &
    md_per_lb 10 &
    xi_ibm 0.1 &
    wall_y 1 1 &
    gravity v_Gx 0.0 0.0 &
    kernel roma &
    vtk 5000 flow_vtk &
    checkpoint 10000 ckpt/fluid &
    check_every 1000

run 100000
```

---

## 13. Numerical Guidelines

### Stability

- **Relaxation time:** τ must be > 0.5. Values in [0.6, 1.5] give best accuracy. τ > 2.0 increases viscous damping of small features. CoupLB warns if τ < 0.505.
- **Mach number:** Keep Ma = u\_max / cs < 0.3 for compressibility errors below ~10%. CoupLB warns at Ma > 0.3 and errors out at Ma > 0.5.
- **Density:** Should remain close to rho0. CoupLB clamps negative densities to a floor value (max of 10% of rho0 or 1e-10) and reports the count of clamped nodes. Non-finite densities are also caught and clamped.
- **Wall velocity Mach check:** CoupLB checks the wall velocity Mach number at initialization and errors if Ma > 0.5.
- **Poiseuille Mach estimate:** For constant gravity with walls, CoupLB estimates the expected peak velocity u\_max = g\*H²/(8ν\*) and checks its Mach number at initialization.

### Accuracy

- **Grid resolution:** For IBM-coupled bodies, diameter/dx ≥ 10 is recommended.
- **IBM delta kernel normalization:** The interpolation and spreading functions normalize by the sum of delta weights over fluid nodes only, ensuring correct behavior near walls.
- **Sub-stepping momentum conservation:** With `md_per_lb > 1`, the grid reaction force is scaled by 1/(N × force\_scale) so that the total impulse on the fluid over one LBM interval exactly balances the total impulse on the particles.
- **xi\_ibm tuning:** For thin structures in strongly driven flows, values well below 1.0 (e.g. 0.01–0.1) create effectively one-way coupling that may be sufficient and more stable. The momentum-conserving version (xi\_ibm = 1/md\_per\_lb) gives proper two-way coupling but requires careful force scaling.

### Performance

- VTK output gathers all data to rank 0. For grids larger than ~128³, output frequency should be kept low.
- Checkpoints write one file per rank with no gathering — fast and scalable.
- The LBM step dominates runtime. MPI communication is typically < 1%.

---

## 14. Known Limitations

- **No turbulence models:** BGK collision only. For high-Re flows, consider MRT or entropic stabilization (not implemented).
- **Single-pass IBM forcing:** No iterative correction for no-slip enforcement. Expect O(10%) velocity errors near immersed boundaries with coarse marker spacing.
- **Fixed Lagrangian volume:** `dv_lag = 1.0` is hardcoded in the IBM spread function. For resolved particles, this should match the surface element area.
- **No adaptive mesh refinement.**
- **Checkpoint portability:** Checkpoints are tied to the same processor count and decomposition. Cannot restart on a different number of ranks.
- **VTK gathering:** All VTK data is gathered to rank 0 and written as ASCII. This limits scalability for very large grids.