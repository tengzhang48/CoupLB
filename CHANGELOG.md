# Changelog

## 2026-04-29

### Fixed
- Validate `md_per_lb >= 1` and wall types in `[0,4]` at parse time (constructor).
- Remove duplicate `enforce_wall_ghost_fields` call in `do_ibm_sub_coupling`.
- Re-evaluate `ibm_has_particles` every MD substep so late-arriving particles couple immediately rather than waiting for the next LBM cycle.
- Cache delta kernel weights in `IBM::spread()` to avoid redundant evaluations.

### Documentation
- Rewrite `README.md` as a compact landing page.
- Add user-facing docs: `architecture.md`, `io.md`, `keywords.md`, `parallelism.md`, `theory.md`.
- Add `CODE_REVIEW_2026-04-29.md` and `INDEPENDENT_VERIFICATION_2026-04-29.md`.

### Notes
- `couplb_io.h`: B1 (`error->one` vs `error->all`) retained as-is pending future collective restructuring. See verification doc for rationale.
