#ifndef COUPLB_IO_H
#define COUPLB_IO_H

#include "atom.h"
#include "couplb_lattice.h"
#include "error.h"
#include <mpi.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace LAMMPS_NS {
namespace CoupLB {

// Checkpoint file format constants
namespace Checkpoint {
  constexpr uint32_t MAGIC   = 0x434C424B;  // "CLBK"
  constexpr int32_t  VERSION = 1;
}

template <typename Lattice>
class IO {
public:

  // ==================================================================
  // VTK output: single .vti file gathered to rank 0
  //
  // Writes cell-centered fields (rho, velocity, type, force) for
  // the full global grid. All ranks send their interior cells to
  // rank 0, which assembles and writes one file.
  //
  // Memory on rank 0: ~(4 doubles + 1 int) × Nx×Ny×Nz
  //   For 64³: ~20 MB. For 128³: ~160 MB.
  //
  // Caller should ensure macroscopic fields are up-to-date.
  // ==================================================================

  static void write_vtk(const Grid<Lattice>& grid, MPI_Comm comm,
                        long step, const std::string& prefix,
                        const double domain_lo[3], double dx_phys,
                        double vel_scale = 1.0, double force_scale = 1.0,
                        double rho_scale = 1.0,
                        const int* region = nullptr)
  {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int Nx = grid.Nx, Ny = grid.Ny;
    const int Nz = (Lattice::D == 3) ? grid.Nz : 1;

    // Region bounds (default: full domain)
    const int ilo = region ? region[0] : 0;
    const int ihi = region ? region[1] : Nx;
    const int jlo = region ? region[2] : 0;
    const int jhi = region ? region[3] : Ny;
    const int klo = region ? region[4] : 0;
    const int khi = region ? region[5] : Nz;

    const int Rx = ihi - ilo, Ry = jhi - jlo, Rz = khi - klo;
    const size_t ntot = (size_t)Rx * Ry * Rz;

    const int nlx = grid.nx, nly = grid.ny;
    const int nlz = (Lattice::D == 3) ? grid.nz : 1;
    const int ox = grid.offset[0], oy = grid.offset[1];
    const int oz = (Lattice::D == 3) ? grid.offset[2] : 0;

    // Local cells overlapping the region
    const int li0 = std::max(0, ilo - ox), li1 = std::min(nlx, ihi - ox);
    const int lj0 = std::max(0, jlo - oy), lj1 = std::min(nly, jhi - oy);
    const int lk0 = std::max(0, klo - oz), lk1 = std::min(nlz, khi - oz);
    const int rnx = std::max(0, li1 - li0);
    const int rny = std::max(0, lj1 - lj0);
    const int rnz = std::max(0, lk1 - lk0);
    const int nlocal = rnx * rny * rnz;

    // Gather metadata: each rank sends (global_ox, global_oy, global_oz, rnx, rny, rnz)
    int local_info[6] = {ox + li0, oy + lj0, oz + lk0, rnx, rny, rnz};
    std::vector<int> all_info(nprocs * 6);
    MPI_Gather(local_info, 6, MPI_INT, all_info.data(), 6, MPI_INT, 0, comm);

    // Gather counts for Gatherv
    std::vector<int> counts(nprocs), displs(nprocs);
    MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    int total_gathered = 0;
    if (rank == 0) {
      displs[0] = 0;
      for (int r = 1; r < nprocs; r++)
        displs[r] = displs[r-1] + counts[r-1];
      total_gathered = displs[nprocs-1] + counts[nprocs-1];
    }

    // Rank 0 opens file and writes header
    FILE* fp = nullptr;
    if (rank == 0) {
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%06ld.vti", prefix.c_str(), step);
      fp = fopen(fname, "w");
      if (!fp) {
        fprintf(stderr, "CoupLB IO: cannot open %s\n", fname);
      } else {
        const double reg_lo[3] = {
          domain_lo[0] + ilo * dx_phys,
          domain_lo[1] + jlo * dx_phys,
          domain_lo[2] + klo * dx_phys
        };
        fprintf(fp, "<?xml version=\"1.0\"?>\n");
        fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
                    "byte_order=\"LittleEndian\">\n");
        fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
                    "Origin=\"%.8e %.8e %.8e\" "
                    "Spacing=\"%.8e %.8e %.8e\">\n",
                Rx, Ry, Rz,
                reg_lo[0], reg_lo[1], reg_lo[2],
                dx_phys, dx_phys, dx_phys);
        fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Rx, Ry, Rz);
        fprintf(fp, "      <CellData Scalars=\"rho\" Vectors=\"velocity_phys\">\n");
      }
    }

    // Helper: gather one scalar field, reorder on rank 0, write, free
    auto write_scalar = [&](const char* name, auto pack_fn) {
      std::vector<double> lbuf(nlocal);
      int c = 0;
      for (int k = lk0; k < lk0 + rnz; k++)
        for (int j = lj0; j < lj0 + rny; j++)
          for (int i = li0; i < li0 + rnx; i++)
            lbuf[c++] = pack_fn(grid.lidx(i, j, k));

      std::vector<double> gbuf;
      if (rank == 0) gbuf.resize(total_gathered);
      MPI_Gatherv(lbuf.data(), nlocal, MPI_DOUBLE,
                  gbuf.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

      if (rank == 0 && fp) {
        std::vector<double> field(ntot);
        for (int r = 0; r < nprocs; r++) {
          const int rox = all_info[r*6] - ilo;
          const int roy = all_info[r*6+1] - jlo;
          const int roz = all_info[r*6+2] - klo;
          const int rnx_r=all_info[r*6+3], rny_r=all_info[r*6+4], rnz_r=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz_r; kk++)
            for (int jj=0; jj<rny_r; jj++)
              for (int ii=0; ii<rnx_r; ii++) {
                const size_t gn = (size_t)(rox+ii) + (size_t)Rx*((size_t)(roy+jj) + (size_t)Ry*(roz+kk));
                field[gn] = gbuf[displs[r] + idx++];
              }
        }
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", name);
        for (size_t n = 0; n < ntot; n++) fprintf(fp, "%.8e\n", field[n]);
        fprintf(fp, "        </DataArray>\n");
      }
    };

    // Helper: gather one vector field (3 components), reorder, write, free
    auto write_vector = [&](const char* name, auto pack_x, auto pack_y, auto pack_z) {
      std::vector<double> lx(nlocal), ly(nlocal), lz(nlocal);
      int c = 0;
      for (int k = lk0; k < lk0 + rnz; k++)
        for (int j = lj0; j < lj0 + rny; j++)
          for (int i = li0; i < li0 + rnx; i++) {
            const int n = grid.lidx(i, j, k);
            lx[c] = pack_x(n); ly[c] = pack_y(n); lz[c] = pack_z(n);
            c++;
          }

      // Gather each component
      std::vector<double> gx, gy, gz;
      if (rank == 0) { gx.resize(total_gathered); gy.resize(total_gathered); gz.resize(total_gathered); }
      MPI_Gatherv(lx.data(), nlocal, MPI_DOUBLE, gx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
      MPI_Gatherv(ly.data(), nlocal, MPI_DOUBLE, gy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
      MPI_Gatherv(lz.data(), nlocal, MPI_DOUBLE, gz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

      if (rank == 0 && fp) {
        std::vector<double> fx(ntot), fy(ntot), fz(ntot);
        for (int r = 0; r < nprocs; r++) {
          const int rox = all_info[r*6] - ilo;
          const int roy = all_info[r*6+1] - jlo;
          const int roz = all_info[r*6+2] - klo;
          const int rnx_r=all_info[r*6+3], rny_r=all_info[r*6+4], rnz_r=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz_r; kk++)
            for (int jj=0; jj<rny_r; jj++)
              for (int ii=0; ii<rnx_r; ii++) {
                const size_t gn = (size_t)(rox+ii) + (size_t)Rx*((size_t)(roy+jj) + (size_t)Ry*(roz+kk));
                fx[gn] = gx[displs[r]+idx]; fy[gn] = gy[displs[r]+idx]; fz[gn] = gz[displs[r]+idx];
                idx++;
              }
        }
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
                    "NumberOfComponents=\"3\" format=\"ascii\">\n", name);
        for (size_t n = 0; n < ntot; n++)
          fprintf(fp, "%.8e %.8e %.8e\n", fx[n], fy[n], fz[n]);
        fprintf(fp, "        </DataArray>\n");
      }
    };

    // Helper: gather integer scalar field
    auto write_int_scalar = [&](const char* name, auto pack_fn) {
      std::vector<int> lbuf(nlocal);
      int c = 0;
      for (int k = lk0; k < lk0 + rnz; k++)
        for (int j = lj0; j < lj0 + rny; j++)
          for (int i = li0; i < li0 + rnx; i++)
            lbuf[c++] = pack_fn(grid.lidx(i, j, k));

      std::vector<int> gbuf;
      if (rank == 0) gbuf.resize(total_gathered);
      MPI_Gatherv(lbuf.data(), nlocal, MPI_INT,
                  gbuf.data(), counts.data(), displs.data(), MPI_INT, 0, comm);

      if (rank == 0 && fp) {
        std::vector<int> field(ntot);
        for (int r = 0; r < nprocs; r++) {
          const int rox = all_info[r*6] - ilo;
          const int roy = all_info[r*6+1] - jlo;
          const int roz = all_info[r*6+2] - klo;
          const int rnx_r=all_info[r*6+3], rny_r=all_info[r*6+4], rnz_r=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz_r; kk++)
            for (int jj=0; jj<rny_r; jj++)
              for (int ii=0; ii<rnx_r; ii++) {
                const size_t gn = (size_t)(rox+ii) + (size_t)Rx*((size_t)(roy+jj) + (size_t)Ry*(roz+kk));
                field[gn] = gbuf[displs[r] + idx++];
              }
        }
        fprintf(fp, "        <DataArray type=\"Int32\" Name=\"%s\" format=\"ascii\">\n", name);
        for (size_t n = 0; n < ntot; n++) fprintf(fp, "%d\n", field[n]);
        fprintf(fp, "        </DataArray>\n");
      }
    };

    // Stream fields one at a time — peak memory is only ~2×ntot doubles
    const double vs = vel_scale, fs = force_scale, rs = rho_scale;

    write_scalar("rho", [&](int n) { return grid.rho[n] * rs; });

    write_vector("velocity_phys",
      [&](int n) { return grid.ux[n] * vs; },
      [&](int n) { return grid.uy[n] * vs; },
      [&](int n) { return grid.uz[n] * vs; });

    write_int_scalar("type", [&](int n) { return grid.type[n]; });

    write_vector("force_phys",
      [&](int n) { return grid.fx[n] * fs; },
      [&](int n) { return grid.fy[n] * fs; },
      [&](int n) { return grid.fz[n] * fs; });

    // Close file
    if (rank == 0 && fp) {
      fprintf(fp, "      </CellData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </ImageData>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);

      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%06ld.vti", prefix.c_str(), step);
      fprintf(stdout, "CoupLB: VTK written at step %ld -> %s (%dx%dx%d region)\n",
              step, fname, Rx, Ry, Rz);
    }
  }

  // ==================================================================
  // PVD time series: a single .pvd file that references all .vti files.
  // Open this one file in ParaView to get a time slider over all steps.
  //
  // Called after each write_vtk. Overwrites the .pvd with the full
  // list so the file is always valid (even if the run crashes mid-way).
  // ==================================================================

  static void write_pvd(const std::string& pvd_filename,
                        const std::string& vtk_prefix,
                        const std::vector<long>& steps,
                        double dt)
  {
    write_pvd_impl(pvd_filename, vtk_prefix, steps, dt, "vti");
  }

  // ==================================================================
  // PVD time series for solid VTK (.vtp)
  // ==================================================================

  static void write_pvd_vtp(const std::string& pvd_filename,
                            const std::string& vtk_prefix,
                            const std::vector<long>& steps,
                            double dt)
  {
    write_pvd_impl(pvd_filename, vtk_prefix, steps, dt, "vtp");
  }

  // ==================================================================
  // Solid VTK: atom data as VTK PolyData (.vtp)
  //
  // Writes user-specified atom attributes. Auto-detects vector groups
  // (x/y/z, vx/vy/vz, etc.) and writes as 3-component arrays.
  // No VTK library required — writes XML directly.
  //
  // Notes:
  // - Always writes Points from atom->x (x,y,z), independent of attrs.
  // - If x/y/z are requested and all 3 are present, writes a "position"
  //   3-component PointData array (not 3 separate scalars).
  // - All PointData arrays are written as Float64 (even id/type).
  // ==================================================================

  static void write_solid_vtk(MPI_Comm comm, long step,
                             const std::string& prefix,
                             const std::vector<std::string>& attrs,
                             int nlocal, const int* mask, int groupbit,
                             Atom* atom, Error* error,
                             const double* region_lo = nullptr,
                             const double* region_hi = nullptr)
  {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (!atom) error->all(FLERR, "CoupLB IO: write_solid_vtk called with null atom pointer");
    if (!mask) error->all(FLERR, "CoupLB IO: write_solid_vtk called with null mask pointer");

    // Helper: check if atom i is inside region
    auto in_region = [&](int i) -> bool {
      if (!region_lo || !region_hi) return true;
      const double xi = atom->x[i][0];
      const double yi = atom->x[i][1];
      const double zi = atom->x[i][2];
      if (xi < region_lo[0] || xi >= region_hi[0]) return false;
      if (yi < region_lo[1] || yi >= region_hi[1]) return false;
      if (zi < region_lo[2] || zi >= region_hi[2]) return false;
      return true;
    };

    // --- Attribute registry ---
    struct AttrInfo {
      const char* name;
      const char* vector_group; // empty string => scalar
      int component;            // 0/1/2 for vectors
      std::function<double(int)> get;
    };

    auto make_scalar = [](const char* n, std::function<double(int)> fn) -> AttrInfo {
      return {n, "", 0, std::move(fn)};
    };
    auto make_vec = [](const char* n, const char* grp, int c, std::function<double(int)> fn) -> AttrInfo {
      return {n, grp, c, std::move(fn)};
    };

    std::vector<AttrInfo> registry;
    registry.reserve(32);

    registry.push_back(make_scalar("id",   [&](int i) { return static_cast<double>(atom->tag[i]); }));
    registry.push_back(make_scalar("type", [&](int i) { return static_cast<double>(atom->type[i]); }));

    if (atom->molecule)
      registry.push_back(make_scalar("mol", [&](int i) { return static_cast<double>(atom->molecule[i]); }));

    if (atom->rmass)
      registry.push_back(make_scalar("mass", [&](int i) { return atom->rmass[i]; }));
    else if (atom->mass)
      registry.push_back(make_scalar("mass", [&](int i) { return atom->mass[atom->type[i]]; }));

    if (atom->radius)
      registry.push_back(make_scalar("diameter", [&](int i) { return 2.0 * atom->radius[i]; }));

    registry.push_back(make_vec("x",  "position", 0, [&](int i) { return atom->x[i][0]; }));
    registry.push_back(make_vec("y",  "position", 1, [&](int i) { return atom->x[i][1]; }));
    registry.push_back(make_vec("z",  "position", 2, [&](int i) { return atom->x[i][2]; }));

    registry.push_back(make_vec("vx", "velocity", 0, [&](int i) { return atom->v[i][0]; }));
    registry.push_back(make_vec("vy", "velocity", 1, [&](int i) { return atom->v[i][1]; }));
    registry.push_back(make_vec("vz", "velocity", 2, [&](int i) { return atom->v[i][2]; }));

    registry.push_back(make_vec("fx", "force", 0, [&](int i) { return atom->f[i][0]; }));
    registry.push_back(make_vec("fy", "force", 1, [&](int i) { return atom->f[i][1]; }));
    registry.push_back(make_vec("fz", "force", 2, [&](int i) { return atom->f[i][2]; }));

    if (atom->mu) {
      registry.push_back(make_vec("mux", "dipole", 0, [&](int i) { return atom->mu[i][0]; }));
      registry.push_back(make_vec("muy", "dipole", 1, [&](int i) { return atom->mu[i][1]; }));
      registry.push_back(make_vec("muz", "dipole", 2, [&](int i) { return atom->mu[i][2]; }));
    }

    if (atom->omega) {
      registry.push_back(make_vec("omegax", "omega", 0, [&](int i) { return atom->omega[i][0]; }));
      registry.push_back(make_vec("omegay", "omega", 1, [&](int i) { return atom->omega[i][1]; }));
      registry.push_back(make_vec("omegaz", "omega", 2, [&](int i) { return atom->omega[i][2]; }));
    }

    if (atom->torque) {
      registry.push_back(make_vec("tqx", "torque", 0, [&](int i) { return atom->torque[i][0]; }));
      registry.push_back(make_vec("tqy", "torque", 1, [&](int i) { return atom->torque[i][1]; }));
      registry.push_back(make_vec("tqz", "torque", 2, [&](int i) { return atom->torque[i][2]; }));
    }

    auto require_available = [&](const std::string& a) {
      if (a == "mol" && !atom->molecule)
        error->all(FLERR, "fix couplb vtk_solid: attribute 'mol' requires a molecular atom style");
      if (a == "diameter" && !atom->radius)
        error->all(FLERR, "fix couplb vtk_solid: attribute 'diameter' requires atom style sphere (radius)");
      if ((a == "mux" || a == "muy" || a == "muz") && !atom->mu)
        error->all(FLERR, "fix couplb vtk_solid: dipole attributes require dipole atom style (mu)");
      if ((a == "omegax" || a == "omegay" || a == "omegaz") && !atom->omega)
        error->all(FLERR, "fix couplb vtk_solid: omega attributes require atom style sphere (omega)");
      if ((a == "tqx" || a == "tqy" || a == "tqz") && !atom->torque)
        error->all(FLERR, "fix couplb vtk_solid: torque attributes require atom style sphere (torque)");
      if (a == "mass" && !(atom->rmass || atom->mass))
        error->all(FLERR, "fix couplb vtk_solid: attribute 'mass' requires per-atom mass (rmass) or type mass table");
    };

    // --- Resolve requested attributes (hard error on unknown/unavailable) ---
    std::vector<const AttrInfo*> requested;
    requested.reserve(attrs.size());

    for (const auto& a : attrs) {
      require_available(a);
      const AttrInfo* found = nullptr;
      for (const auto& r : registry) {
        if (a == r.name) { found = &r; break; }
      }
      if (!found)
        error->all(FLERR, "fix couplb vtk_solid: unknown attribute '{}'", a);
      requested.push_back(found);
    }

    // --- Group vectors only when all 3 components were requested ---
    struct OutputField {
      std::string name;
      int ncomp; // 1 or 3
      std::function<double(int)> get[3];
    };

    std::vector<OutputField> fields;
    fields.reserve(requested.size());
    std::set<std::string> groups_done;

    for (const auto* r : requested) {
      if (r->vector_group && r->vector_group[0] != '\0') {
        const std::string grp(r->vector_group);
        if (groups_done.count(grp)) continue;

        const AttrInfo* comp[3] = {nullptr, nullptr, nullptr};
        for (const auto* q : requested) {
          if (q->vector_group && strcmp(q->vector_group, r->vector_group) == 0) {
            if (q->component < 0 || q->component > 2)
              error->all(FLERR, "CoupLB IO: internal error: bad component index for '{}'", q->name);
            comp[q->component] = q;
          }
        }

        if (comp[0] && comp[1] && comp[2]) {
          OutputField of;
          of.name = grp;
          of.ncomp = 3;
          of.get[0] = comp[0]->get;
          of.get[1] = comp[1]->get;
          of.get[2] = comp[2]->get;
          fields.push_back(std::move(of));
          groups_done.insert(grp);
        } else {
          OutputField of;
          of.name = r->name;
          of.ncomp = 1;
          of.get[0] = r->get;
          fields.push_back(std::move(of));
        }
      } else {
        OutputField of;
        of.name = r->name;
        of.ncomp = 1;
        of.get[0] = r->get;
        fields.push_back(std::move(of));
      }
    }

    // --- Count local atoms in group AND region ---
    int ngroup = 0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!in_region(i)) continue;
      ngroup++;
    }

    // --- Pack: x,y,z always + requested fields ---
    int doubles_per_atom = 3;
    for (const auto& f : fields) doubles_per_atom += f.ncomp;

    std::vector<double> lbuf;
    lbuf.resize((size_t)ngroup * (size_t)doubles_per_atom);

    int c = 0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!in_region(i)) continue;

      lbuf[c++] = atom->x[i][0];
      lbuf[c++] = atom->x[i][1];
      lbuf[c++] = atom->x[i][2];

      for (const auto& f : fields) {
        for (int d = 0; d < f.ncomp; d++)
          lbuf[c++] = f.get[d](i);
      }
    }

    const int nsend = ngroup * doubles_per_atom;

    std::vector<int> counts(nprocs), displs(nprocs);
    MPI_Gather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    int total = 0;
    if (rank == 0) {
      displs[0] = 0;
      for (int r = 1; r < nprocs; r++)
        displs[r] = displs[r-1] + counts[r-1];
      total = displs[nprocs-1] + counts[nprocs-1];
    }

    std::vector<double> gbuf;
    if (rank == 0) gbuf.resize((size_t)total);

    const double* sendptr = lbuf.empty() ? nullptr : lbuf.data();
    double* recvptr = (rank == 0 && !gbuf.empty()) ? gbuf.data() : nullptr;

    MPI_Gatherv(sendptr, nsend, MPI_DOUBLE,
                recvptr, counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (rank != 0) return;

    if (total % doubles_per_atom != 0)
      error->one(FLERR, "CoupLB IO: solid VTK gather size not divisible by record size");

    const int ntotal = total / doubles_per_atom;

    char fname[512];
    snprintf(fname, sizeof(fname), "%s_solid_%06ld.vtp", prefix.c_str(), step);
    FILE* fp = fopen(fname, "w");
    if (!fp)
      error->one(FLERR, "CoupLB IO: cannot open {}", fname);

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PolyData>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfVerts=\"%d\" "
                "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n",
            ntotal, ntotal);

    // Points
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int i = 0; i < ntotal; i++) {
      const double* d = &gbuf[(size_t)i * (size_t)doubles_per_atom];
      fprintf(fp, "%.8e %.8e %.8e\n", d[0], d[1], d[2]);
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");

    // Verts (one vertex per point)
    fprintf(fp, "      <Verts>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (int i = 0; i < ntotal; i++) fprintf(fp, "%d\n", i);
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (int i = 0; i < ntotal; i++) fprintf(fp, "%d\n", i + 1);
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Verts>\n");

    // PointData
    fprintf(fp, "      <PointData>\n");

    int field_offset = 3;
    for (const auto& f : fields) {
      if (f.ncomp == 1) {
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", f.name.c_str());
        for (int i = 0; i < ntotal; i++) {
          const double val = gbuf[(size_t)i * (size_t)doubles_per_atom + (size_t)field_offset];
          fprintf(fp, "%.8e\n", val);
        }
        fprintf(fp, "        </DataArray>\n");
      } else {
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\">\n",
                f.name.c_str());
        for (int i = 0; i < ntotal; i++) {
          const double* d = &gbuf[(size_t)i * (size_t)doubles_per_atom + (size_t)field_offset];
          fprintf(fp, "%.8e %.8e %.8e\n", d[0], d[1], d[2]);
        }
        fprintf(fp, "        </DataArray>\n");
      }
      field_offset += f.ncomp;
    }

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </PolyData>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);

    fprintf(stdout, "CoupLB: solid VTK written at step %ld -> %s (%d atoms, %d fields)\n",
            step, fname, ntotal, (int)fields.size());
  }

private:

  static void write_pvd_impl(const std::string& pvd_filename,
                             const std::string& prefix,
                             const std::vector<long>& steps,
                             double dt, const char* ext)
  {
    FILE* fp = fopen(pvd_filename.c_str(), "w");
    if (!fp) {
      fprintf(stderr, "CoupLB IO: cannot open %s\n", pvd_filename.c_str());
      return;
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
                "byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <Collection>\n");

    for (size_t i = 0; i < steps.size(); i++) {
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%06ld.%s", prefix.c_str(), steps[i], ext);

      const char* base = strrchr(fname, '/');
      base = base ? base + 1 : fname;

      const double time = steps[i] * dt;
      fprintf(fp, "    <DataSet timestep=\"%.8e\" file=\"%s\"/>\n", time, base);
    }

    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
  }

public:

  // ==================================================================
  // Checkpoint: binary dump of distribution function f[]
  //
  // One file per MPI rank: prefix.RANK.clbk
  // Contains header (dimensions, step) + full f[] array.
  // On read, macroscopic fields are recomputed from f[].
  //
  // File format:
  //   [4B magic][4B version][8B step]
  //   [4B D][4B Q]
  //   [4B nx][4B ny][4B nz]
  //   [4B ox][4B oy][4B oz]
  //   [4B Nx][4B Ny][4B Nz]
  //   [8B x (Q x ntotal) distribution function f]
  // ==================================================================

  static bool write_checkpoint(const Grid<Lattice>& grid, MPI_Comm comm,
                               long step, const std::string& prefix)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    char fname[512];
    snprintf(fname, sizeof(fname), "%s.%d.clbk", prefix.c_str(), rank);

    FILE* fp = fopen(fname, "wb");
    if (!fp) {
      fprintf(stderr, "CoupLB IO: rank %d cannot open %s for writing\n",
              rank, fname);
      return false;
    }

    // Header
    uint32_t magic   = Checkpoint::MAGIC;
    int32_t  version = Checkpoint::VERSION;
    int64_t  step64  = step;
    int32_t  D = Lattice::D, Q = Lattice::Q;
    int32_t  dims[3] = {grid.nx, grid.ny, grid.nz};
    int32_t  offs[3] = {grid.offset[0], grid.offset[1], grid.offset[2]};
    int32_t  glob[3] = {grid.Nx, grid.Ny, grid.Nz};

    bool ok = true;
    ok = ok && (fwrite(&magic,   4, 1, fp) == 1);
    ok = ok && (fwrite(&version, 4, 1, fp) == 1);
    ok = ok && (fwrite(&step64,  8, 1, fp) == 1);
    ok = ok && (fwrite(&D, 4, 1, fp) == 1);
    ok = ok && (fwrite(&Q, 4, 1, fp) == 1);
    ok = ok && (fwrite(dims, 4, 3, fp) == 3);
    ok = ok && (fwrite(offs, 4, 3, fp) == 3);
    ok = ok && (fwrite(glob, 4, 3, fp) == 3);

    // Distribution function (full array including ghost cells)
    size_t nf = grid.f.size();
    ok = ok && (fwrite(grid.f.data(), sizeof(double), nf, fp) == nf);
    fclose(fp);

    if (!ok) {
      fprintf(stderr, "CoupLB IO: rank %d write error in %s\n", rank, fname);
      return false;
    }

    if (rank == 0)
      fprintf(stdout, "CoupLB: checkpoint written at step %ld -> %s.*.clbk\n",
              step, prefix.c_str());

    return true;
  }

  static bool read_checkpoint(Grid<Lattice>& grid, MPI_Comm comm,
                              const std::string& prefix, long& step_out)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);

    char fname[512];
    snprintf(fname, sizeof(fname), "%s.%d.clbk", prefix.c_str(), rank);

    FILE* fp = fopen(fname, "rb");
    if (!fp) {
      fprintf(stderr, "CoupLB IO: rank %d cannot open %s for reading\n",
              rank, fname);
      return false;
    }

    // Read and validate header — check every fread return value
    uint32_t magic;
    if (fread(&magic, 4, 1, fp) != 1 || magic != Checkpoint::MAGIC) {
      fprintf(stderr, "CoupLB IO: rank %d bad magic in %s\n", rank, fname);
      fclose(fp); return false;
    }

    int32_t version;
    if (fread(&version, 4, 1, fp) != 1 || version != Checkpoint::VERSION) {
      fprintf(stderr, "CoupLB IO: rank %d unsupported version %d in %s\n",
              rank, version, fname);
      fclose(fp); return false;
    }

    int64_t step64;
    if (fread(&step64, 8, 1, fp) != 1) {
      fprintf(stderr, "CoupLB IO: rank %d truncated header (step) in %s\n",
              rank, fname);
      fclose(fp); return false;
    }
    step_out = (long)step64;

    int32_t D, Q;
    if (fread(&D, 4, 1, fp) != 1 || fread(&Q, 4, 1, fp) != 1) {
      fprintf(stderr, "CoupLB IO: rank %d truncated header (D/Q) in %s\n",
              rank, fname);
      fclose(fp); return false;
    }
    if (D != Lattice::D || Q != Lattice::Q) {
      fprintf(stderr, "CoupLB IO: rank %d D/Q mismatch in %s "
              "(file: D=%d Q=%d, expected: D=%d Q=%d)\n",
              rank, fname, D, Q, Lattice::D, Lattice::Q);
      fclose(fp); return false;
    }

    int32_t dims[3], offs[3], glob[3];
    if (fread(dims, 4, 3, fp) != 3 || fread(offs, 4, 3, fp) != 3 ||
        fread(glob, 4, 3, fp) != 3) {
      fprintf(stderr, "CoupLB IO: rank %d truncated header (dims) in %s\n",
              rank, fname);
      fclose(fp); return false;
    }

    // Validate grid dimensions match
    if (dims[0] != grid.nx || dims[1] != grid.ny || dims[2] != grid.nz) {
      fprintf(stderr, "CoupLB IO: rank %d local grid mismatch in %s "
              "(file: %dx%dx%d, current: %dx%dx%d)\n",
              rank, fname, dims[0], dims[1], dims[2],
              grid.nx, grid.ny, grid.nz);
      fclose(fp); return false;
    }
    if (offs[0] != grid.offset[0] || offs[1] != grid.offset[1] ||
        offs[2] != grid.offset[2]) {
      fprintf(stderr, "CoupLB IO: rank %d offset mismatch in %s "
              "(file: %d,%d,%d, current: %d,%d,%d)\n",
              rank, fname, offs[0], offs[1], offs[2],
              grid.offset[0], grid.offset[1], grid.offset[2]);
      fclose(fp); return false;
    }
    if (glob[0] != grid.Nx || glob[1] != grid.Ny || glob[2] != grid.Nz) {
      fprintf(stderr, "CoupLB IO: rank %d global grid mismatch in %s\n",
              rank, fname);
      fclose(fp); return false;
    }

    // Read distribution function — check for truncated file
    size_t nf = grid.f.size();
    size_t nread = fread(grid.f.data(), sizeof(double), nf, fp);
    fclose(fp);

    if (nread != nf) {
      fprintf(stderr, "CoupLB IO: rank %d short read in %s (%zu/%zu doubles)\n",
              rank, fname, nread, nf);
      return false;
    }

    // Copy to swap buffer so first stream() has valid data
    grid.f_buf = grid.f;

    // Recompute macroscopic fields from restored distributions
    grid.compute_macroscopic(false);

    if (rank == 0) {
      fprintf(stdout, "CoupLB: checkpoint loaded from step %ld <- %s.*.clbk\n",
              step_out, prefix.c_str());
      fprintf(stdout, "CoupLB WARNING: LAMMPS timestep is NOT updated by checkpoint.\n"
                      "  Use 'reset_timestep %ld' in your input script if needed.\n",
              step_out);
    }

    return true;
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_IO_H