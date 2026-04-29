# CoupLB Keywords & Syntax

```
fix ID group couplb Nx Ny Nz nu rho0 [keywords...]
```

## Required Arguments

| Argument | Type | Description |
|:---------|:-----|:------------|
| `Nx Ny Nz` | int | Number of LB nodes per direction. Must satisfy `Nx × dx = Lx` (similarly for y, z). |
| `nu` | float | Kinematic viscosity in LAMMPS units. τ = ν×dt_LB/dx²/cs² + 0.5 must be > 0.5. |
| `rho0` | float | Reference fluid density in LAMMPS units. |

## Optional Keywords

| Keyword | Arguments | Default | Description |
|:--------|:----------|:--------|:------------|
| `md_per_lb` | N | 1 | MD steps per LBM step. Enables sub-stepping with momentum-conserving IBM. |
| `xi_ibm` | value | 1.0 | IBM relaxation factor (ξ ≥ 0). 0 = no coupling, 1 = nominal, <1 softens. |
| `gravity` | gx gy gz | 0 0 0 | Body force acceleration. Each component can be constant or `v_varname`. |
| `wall_x` | lo hi | 0 0 | X-boundary: 0=periodic, 1=no-slip, 2=moving, 3=free-slip, 4=open. |
| `wall_y` | lo hi | 0 0 | Y-boundary (same types). |
| `wall_z` | lo hi | 0 0 | Z-boundary (3D only, same types). |
| `wall_vel` | vx vy vz | 0 0 0 | Velocity for type-2 (moving) walls in LAMMPS units. |
| `kernel` | {roma\|peskin4} | roma | IBM delta function: 3-point or 4-point. |
| `vtk` | N prefix | off | Write VTK field snapshot every N LAMMPS steps. |
| `vtk_region` | xlo xhi ylo yhi [zlo zhi] | full domain | Clip VTK output to subregion in LAMMPS units. |
| `vtk_solid` | attr1 attr2 ... | off | Write atom attributes as VTK PolyData at VTK frequency. |
| `checkpoint` | N prefix | off | Write binary checkpoint every N steps. |
| `restart` | prefix | none | Load fluid state from checkpoint at initialization. |
| `output` | N file | off | Write y-averaged ASCII velocity profiles every N steps. |
| `check_every` | N | 0 | Stability diagnostics (Ma, mass, momentum, density clamps). |

## Wall Types

| Type | Behavior |
|:-----|:---------|
| 0 | Periodic — no wall (default) |
| 1 | No-slip — half-way bounce-back |
| 2 | Moving wall — bounce-back with prescribed velocity (use `wall_vel`) |
| 3 | Free-slip — specular reflection |
| 4 | Open — zero-gradient extrapolation (momentum leaves domain) |

## Variable-Style Gravity

```lammps
variable freq equal 1.0
variable Gx equal 1e-4*sin(2*PI*v_freq*time)
fix lb all couplb 64 64 64 0.1 1.0 gravity v_Gx 0 0
```

Gravity is re-evaluated at every LBM step from the LAMMPS variable.

## Examples

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

### IBM Coupling with Sub-stepping

```lammps
units           lj
dimension       3
boundary        p f p
atom_style      hybrid bpm/rotational dipole

timestep        0.001

fix lb all couplb 64 48 32 0.1 1.26 &
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
