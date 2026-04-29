# CoupLB Parallel Decomposition

## MPI Domain Decomposition

CoupLB mirrors the LAMMPS processor grid. Use the `processors` command:

```
processors 2 4 2    # 16 MPI ranks: 2×4×2
```

### Constraints

- Each grid dimension must be evenly divisible by the number of processors
  in that direction: `Nx % px == 0`, `Ny % py == 0`, `Nz % pz == 0`.
- **Minimum cells per rank:** 3 for Roma kernel, 4 for Peskin4 kernel
  (per direction). Below this, IBM stencils get clipped.
- For wall-bounded flows, split along the wall-normal direction first.

### Example Decompositions

| Grid | Decomposition | Local grid per rank |
|:-----|:-------------|:-------------------|
| 32×32×32 | 2×2×2 (8) | 16×16×16 |
| 64×64×64 | 4×4×4 (64) | 16×16×16 |
| 8×32×8 | 1×4×1 (4) | 8×8×8 |
| 16×64×16 | 2×4×2 (16) | 8×16×8 |

## MPI Communication

Three types of ghost exchange along each dimension:

1. **Distribution exchange** — After streaming, distribution functions f_i
   are communicated to ghost layers. Uses pairwise MPI_Sendrecv.
2. **Velocity exchange** — Before IBM interpolation, macroscopic fields
   (ρ, u) are communicated so IBM stencils near subdomain boundaries see
   correct values.
3. **Reverse force exchange** — After IBM spreading, forces deposited on
   ghost cells are accumulated back onto the owning rank's interior cells.
   This prevents silent force loss at subdomain boundaries.

Communication tags: `DIST_BASE=5000`, `VEL_BASE=6000`, `FORCE_BASE=7000`,
plus dimension and side offsets.

## OpenMP Threading

Collision, streaming, `compute_macroscopic`, `max_velocity`, and
`compute_diagnostics` all use `#pragma omp parallel for collapse(2)
schedule(static)` with proper reductions. Enabled at build time with
`-fopenmp` (gfortran) or equivalent.

## 2D Pencil Decomposition

A `ycomm` communicator for 2D pencil decomposition is partially implemented
in `fix_couplb.cpp`. When active, it subdivides ranks along the y-dimension
for additional parallelism in 2D simulations.

## Checkpoint Portability

Checkpoints are tied to the same processor count and decomposition grid.
Cannot restart on a different number of ranks. Grid dimensions, offsets,
and global sizes are all validated on load; a mismatch produces a clear
error message.
