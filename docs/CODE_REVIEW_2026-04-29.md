# CoupLB Code Review — 2026-04-29

Deep audit of all 9 source files. 32 findings total.

> Note: this is the original review record. See
> `INDEPENDENT_VERIFICATION_2026-04-29.md` for the follow-up assessment. In
> particular, B1 was downgraded from HIGH to LOW, and a direct
> `error->one` → `error->all` replacement is not recommended without
> restructuring `write_solid_vtk` into a collective failure path.

---

## Actual Bugs

### B1. `error->one()` deadlocks on VTK solid file failure (HIGH)

**File:** `couplb_io.h:554`

```cpp
if (!fp) error->one(FLERR, "CoupLB IO: cannot open {}", fname);
```

`error->one` aborts only the calling rank. Non-root ranks proceed normally
while rank 0 is dead — the next MPI collective deadlocks. Must be
`error->all`.

**Fix:** Change to `error->all`.

---

### B2. `ibm_has_particles` cache prevents late-arriving particles (MEDIUM)

**File:** `fix_couplb.cpp:708-719`

`ibm_has_particles` is set once on the first LBM sub-step. If no particles
exist at that moment, it stays `false` for the entire simulation. Subsequent
particles (added via fix pour, create_atoms, read_restart) never get IBM
forces when `md_per_lb > 1`.

For `md_per_lb == 1` it works because `first` is true every step.

**Fix:** Move the particle-count `MPI_Allreduce` into both `first` and
`!first` paths, or re-evaluate unconditionally.

---

### B3. Free-slip wall nodes bias IBM interpolation (MEDIUM)

**File:** `couplb_ibm.h:79`

```cpp
else if (nt==3) { ws+=d; }  // free-slip: weight added, zero velocity
```

Free-slip (type 3) nodes contribute weight but no velocity. This pulls the
interpolated velocity toward zero whenever the IBM kernel overlaps a
free-slip ghost layer. For a particle near a free-slip wall, fluid velocity
is underestimated. No-slip (type 1) does the same but is physically correct
for zero wall velocity.

**Fix:** Free-slip nodes should contribute specular-reflected velocity
(flip the wall-normal component) or at minimum, not contribute weight.

---

### B4. `enforce_wall_ghost_fields` called twice per LBM step (MEDIUM)

**File:** `fix_couplb.cpp:635` and `:705`

Called at end of `do_lbm_step` and again at start of
`do_ibm_sub_coupling`. Nothing modifies ghost-node macroscopic fields
between these calls (IBM only touches forces). Redundant computation for
every LBM cycle.

**Fix:** Remove one call. The call in `do_lbm_step` alone suffices.

---

### B5. No wall type validation (LOW)

**File:** `fix_couplb.cpp:502-509`

If a user passes `wall_x 5 5`, type-5 wall nodes are created. The
streaming step handles types 0-4 but type 5 falls through the entire chain
silently, leaving `fi_buf` uninitialized for those nodes. Should validate
wall types are in [0,4] at init time.

**Fix:** Add a range check in `setup_boundaries`.

---

### B6. `md_per_lb=0` causes division by zero (LOW)

**File:** `fix_couplb.cpp:618`

```cpp
md_sub_count = (md_sub_count + 1) % md_per_lb;
```

If user passes `md_per_lb 0`, this is undefined behavior (modulo by zero).
`init()` checks this at line 231 but `post_force` could fire before `init`
in edge cases. Should validate in constructor.

**Fix:** Add `if (md_per_lb <= 0) error->all(...)` in constructor.

---

### B7. Delta kernel evaluated twice in spread function (LOW)

**File:** `couplb_ibm.h:107-129`

The first loop computes `ws` by summing delta products. The second loop
recomputes the same delta products scaled by `dvr/ws`. For a particle in
bulk fluid (no walls skipped), all 64 (4³) or 16 (4²) delta evaluations
are done twice.

**Fix:** Cache delta values in a small stack array during the first loop
and reuse in the second.

---

## Robustness Issues

### R1. Checkpoint write has no collective barrier (MEDIUM)

**File:** `couplb_io.h:678`

Each rank opens its own file independently. If the filesystem runs out of
space, some ranks succeed while others fail. The `ok` flag reports the error
per-rank, but the MPI collective state becomes inconsistent — later
collectives may deadlock.

**Fix:** Add an `MPI_Allreduce` on the `ok` flag and a collective barrier
before returning on error.

---

### R2. `vtk_region` 2D args in 3D mode silently expands z (LOW)

**File:** `fix_couplb.cpp:170-186`

If a 3D simulation uses `vtk_region xlo xhi ylo yhi` (4 args), CoupLB
auto-expands z to `[-1e30, 1e30]` and clamps to `[0, Nz]`. No warning
is emitted. User may expect their 2D-style region to clip in z.

**Fix:** Emit a warning via `error->warning`.

---

## Performance Issues

### P1. ASCII VTK format is extremely slow for large grids

**File:** `couplb_io.h:164-165, 207-208`

Uses `fprintf("%.8e\n")` for every data element. For 128³ with 10 field
components: 20.9 million lines of text. Base64-encoded binary VTK would be
3-5x faster and produce smaller files.

---

### P2. `collapse(2)` should be `collapse(3)` for OpenMP loops

**Files:** `couplb_lattice.h:160,208,227`, `couplb_streaming.h:91`

Triply-nested loops use `#pragma omp parallel for collapse(2)`. For thin 3D
slabs (small k range), this can create load imbalance. `collapse(3)` with
`schedule(dynamic, 128)` would give better load balance.

---

### P3. Massive code duplication between 2D/3D in `end_of_step`

**File:** `fix_couplb.cpp:816-858`

The `end_of_step` function has a 40-line `if (is3d) { ... } else { ... }`
block where only the template parameter differs. Could be factored into a
single `do_end_of_step<Lattice>()` template.

---

## Portability

### PORT1. `long` truncates int64_t on 32-bit platforms

**File:** `couplb_io.h:757`

```cpp
step_out = (long)step64;
```

On 32-bit MSVC, `long` is 32 bits. Simulations with > 2³¹ steps would
truncate step numbers in checkpoint output. Affects only 32-bit builds
with very long runs.

---

### PORT2. `vtk_steps` uses `long` instead of `int64_t`

**File:** `fix_couplb.h:71`

```cpp
std::vector<long> vtk_steps;
```

Same 32-bit truncation risk. Should be `int64_t` or `long long`.

---

## Unclear Code / Documentation Gaps

| # | File:Line | Issue |
|---|-----------|-------|
| D1 | fix_couplb.cpp:288-290 | `for (d=0; d<dim+1; d++) dp *= dx` — why `dim+1`? The force scale is `rho * dx^(D+1) / dt²`. Good physics but opaque code. Add a comment. |
| D2 | couplb_ibm.h:13-21 | Roma kernel should cite Roma et al. (1999) reference. |
| D3 | couplb_ibm.h:103-104 | `dvr = dvl / cell_vol` is always 1.0. Either remove or document as future hook for variable marker volumes. |
| D4 | fix_couplb.cpp:727 | "Lagrangian volume for penalty IBM: always dx^D" — explain why no per-particle marker volume. |
| D5 | couplb_ibm.h:119 | `sc = dvr / ws` — `ws` can approach 0 near domain boundaries. `DELTA_TOL=1e-12` guard at line 116 prevents division by zero, but the physical meaning of the 1e-12 cutoff should be documented. |

---

## Summary Table

| Severity | Count | Key items |
|----------|-------|-----------|
| High | 1 | B1: `error->one` deadlock on VTK failure |
| Medium | 4 | B2: IBM caching bug, B3: free-slip IBM bias, B4: double wall enforce, R1: checkpoint barrier |
| Low | 6 | B5-B7, R2, P3 |
| Performance | 3 | P1-P3 |
| Portability | 2 | PORT1-PORT2 |
| Documentation | 5 | D1-D5 |

## Recommended Fix Order

1. **B1** — `error->one` → `error->all` (1 line, prevents deadlock)
2. **B2** — Re-evaluate `ibm_has_particles` (fixes late-arriving particles)
3. **B4** — Remove duplicate `enforce_wall_ghost_fields` call (1 line)
4. **B6** — Validate `md_per_lb > 0` in constructor (2 lines)
5. **B5** — Validate wall type range (2 lines)
6. **P3** — Templatize `end_of_step` to remove duplication
7. **B3** — Fix free-slip IBM interpolation (design change — needs discussion)
8. **R1** — Add MPI barrier after checkpoint write
9. **D3** — Remove or document `dvr` placeholder
