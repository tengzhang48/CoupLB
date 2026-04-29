# CoupLB Theory & Numerical Guidelines

## Unit Conversion

All user-facing parameters are in LAMMPS units. CoupLB converts to lattice
units internally (dx\* = 1, dt\* = 1, cs² = 1/3).

The LBM timestep in LAMMPS units is:

```
dt_LB = timestep × md_per_lb
```

Conversion scales:

```
vel_scale   = dx / dt_LB           (lattice velocity → LAMMPS velocity)
force_scale = rho0 × dx^(D+1) / dt_LB²   (lattice force → LAMMPS force)
```

where D is the spatial dimension (2 or 3).

### Parameter Conversions

| Parameter | LAMMPS units | Lattice units | Conversion |
|:----------|:------------|:--------------|:-----------|
| Viscosity | ν | ν\* = ν × dt_LB / dx² | τ = ν\*/cs² + 0.5 |
| Gravity | g (acceleration) | g\* = g × dt_LB² / dx | Body force per unit mass |
| Wall velocity | v_wall | v\*_wall = v_wall / vel_scale | Ma < 0.5 required |
| IBM force | F_particle (LAMMPS) | f_grid = −F / force_scale | Newton's third law |

VTK output is written in LAMMPS physical units.

---

## IBM Coupling

CoupLB applies IBM forces to all atoms in the fix group via the `post_force`
callback.

### 1:1 Coupling (md_per_lb = 1)

```
F_particle = ξ × m × (u_fluid − v_particle) / dt
F_grid     = −F_particle / force_scale
```

where ξ is `xi_ibm`, m is particle mass, u_fluid is interpolated fluid
velocity at the particle position, and v_particle is particle velocity.

### Sub-stepped Coupling (md_per_lb > 1)

```
F_particle = ξ × m × (u_fluid − v_particle) / dt_MD
F_grid     = −F_particle / (N × force_scale)
```

The 1/N scaling on grid force ensures exact momentum conservation over one
LBM interval: sum of particle impulses equals negative of total fluid impulse.
Ghost force exchange is deferred to the last sub-step.

### Delta Kernels

- **Roma (3-point):** Support radius 1.5 lattice spacings. ≥ 3 interior cells
  per rank per direction.
- **Peskin4 (4-point):** Support radius 2.0 lattice spacings. ≥ 4 interior
  cells per rank per direction.

Interpolation includes all node types: fluid nodes contribute velocity,
no-slip walls contribute zero velocity, moving walls contribute their
prescribed velocity. Open boundary nodes are excluded from interpolation.

### xi_ibm Tuning

For thin structures in strongly driven flows, values well below 1.0
(e.g., 0.01–0.1) create effectively one-way coupling. The momentum-conserving
version (xi_ibm = 1/md_per_lb) gives proper two-way coupling but requires
careful force scaling.

---

## Stability

- **Relaxation time:** τ must be > 0.5. Values in [0.6, 1.5] give best
  accuracy. τ > 2.0 increases viscous damping. CoupLB warns if τ < 0.505.
- **Mach number:** Ma = u_max / cs < 0.3 for compressibility errors below
  ~10%. Warns at Ma > 0.3, errors at Ma > 0.5.
- **Density:** Should remain close to rho0. CoupLB clamps densities below
  `RHO_MIN` (1e-10) and reports the count of clamped nodes.
- **Wall velocity check:** Wall velocity Mach number checked at init; errors
  if Ma > 0.5.
- **Poiseuille estimate:** For constant gravity with walls, estimates peak
  velocity u_max = g\*×H²/(8ν\*) and checks its Mach number at init.

## Accuracy

- **Grid resolution:** For IBM-coupled bodies, diameter/dx ≥ 10 recommended.
- **Sub-stepping conservation:** With `md_per_lb > 1`, grid reaction force
  is scaled by 1/(N × force_scale) for exact momentum balance.
- **Single-pass IBM:** No iterative correction for no-slip enforcement.
  Expect O(10%) velocity errors near immersed boundaries with coarse marker
  spacing.
- **Fixed Lagrangian volume:** IBM spread uses `dv = dx^D` for all markers.

## Performance

- LBM step dominates runtime. MPI communication typically < 1%.
- VTK output gathers all data to rank 0 — keep frequency low for grids
  larger than ~128³.
- Checkpoints write one file per rank with no gathering.
