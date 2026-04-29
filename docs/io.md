# CoupLB Output & Checkpointing

## VTK Output

When `vtk N prefix` is enabled, CoupLB writes:

- **`prefix_STEP.vti`** — Single VTK ImageData file containing the fluid
  field (full domain, or clipped to `vtk_region`). All ranks gather data
  to rank 0 for writing.
- **`prefix.pvd`** — ParaView collection file linking all `.vti` files
  with simulation time. Updated after each dump; remains valid even if the
  run crashes.

### VTK Fields

| Field | Components | Units |
|:------|:-----------|:------|
| `rho` | 1 (scalar) | LAMMPS physical units |
| `velocity_phys` | 3 (vector) | LAMMPS physical units |
| `force_phys` | 3 (vector) | LAMMPS physical units |
| `type` | 1 (scalar) | 0=fluid, 1=no-slip, 2=moving, 3=free-slip, 4=open |

### Solid VTK Output (`vtk_solid`)

When `vtk_solid attr1 attr2 ...` is enabled (requires `vtk`):

- **`prefix_solid_STEP.vtp`** — VTK PolyData point cloud of atoms in the
  fix group (filtered by `vtk_region` if specified).
- **`prefix_solid.pvd`** — ParaView collection file.

Vector groups (e.g., `vx vy vz`) are written as a single 3-component array.
Otherwise, components are written as individual scalars. All arrays are
Float64.

```lammps
fix ibm robot couplb ... vtk 1000 vtk/robot vtk_solid id type vx vy vz fx fy fz
```

### VTK Region Clipping (`vtk_region`)

```
vtk_region xlo xhi ylo yhi           # 2D
vtk_region xlo xhi ylo yhi zlo zhi   # 3D
```

Clips both fluid `.vti` and solid `.vtp` output to physical subregion in
LAMMPS units. Converted to node indices internally; bounds snapped to exact
node boundaries for consistency between fluid and solid output.

---

## ASCII Profiles

When `output N file` is enabled, writes y-averaged profiles every N steps:

```
step j y rho ux uy
```

Useful for validating Poiseuille and Couette flows. Velocities are in
LAMMPS physical units.

---

## Stability Diagnostics

When `check_every N` is enabled, prints to the LAMMPS log:

- **Ma:** Maximum Mach number (warn > 0.3, error > 0.5)
- **mass:** Total fluid mass
- **mom:** Total fluid momentum vector
- **density clamps:** Count of nodes where density was clamped

---

## Checkpointing and Restarts

### Writing

```
fix 1 all couplb ... checkpoint 10000 ckpt/fluid
```

Writes binary files `ckpt/fluid.0.clbk`, `ckpt/fluid.1.clbk`, ... (one per
MPI rank). Each checkpoint overwrites the previous at the same prefix.
Contains the full distribution function f_i for exact restart.

File format: magic number (`0x434C424B`), version, step, grid metadata
(D, Q, local dims, offsets, global dims), then raw f[] array.

### Loading

```
fix 1 all couplb ... restart ckpt/fluid
reset_timestep 10000
```

Important:
- `restart` only loads the fluid state; use `read_restart` or `read_data`
  separately for particles.
- `reset_timestep` is required to sync the LAMMPS timestep.
- The checkpoint must be loaded with the same processor count and grid
  decomposition as when written. Validated on load.
- On load, macroscopic fields are recomputed and the swap buffer is
  initialized.

### Example Restart Script

```lammps
read_restart particles.10000

fix 1 all couplb 64 64 64 0.1 1.0 &
    restart ckpt/fluid &
    vtk 5000 flow_vtk &
    checkpoint 10000 ckpt/fluid

reset_timestep 10000
run 10000
```
