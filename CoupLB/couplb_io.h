#ifndef COUPLB_IO_H
#define COUPLB_IO_H

#include "couplb_lattice.h"
#include <mpi.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
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
                        double rho_scale = 1.0)
  {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int Nx = grid.Nx, Ny = grid.Ny;
    const int Nz = (Lattice::D == 3) ? grid.Nz : 1;
    const size_t ntot = (size_t)Nx * Ny * Nz;

    const int nlx = grid.nx, nly = grid.ny;
    const int nlz = (Lattice::D == 3) ? grid.nz : 1;
    const int nlocal = nlx * nly * nlz;

    // Gather metadata: each rank sends (ox, oy, oz, nlx, nly, nlz)
    int local_info[6] = {grid.offset[0], grid.offset[1],
                         (Lattice::D == 3) ? grid.offset[2] : 0,
                         nlx, nly, nlz};
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
        fprintf(fp, "<?xml version=\"1.0\"?>\n");
        fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
                    "byte_order=\"LittleEndian\">\n");
        fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
                    "Origin=\"%.8e %.8e %.8e\" "
                    "Spacing=\"%.8e %.8e %.8e\">\n",
                Nx, Ny, Nz,
                domain_lo[0], domain_lo[1], domain_lo[2],
                dx_phys, dx_phys, dx_phys);
        fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Nx, Ny, Nz);
        fprintf(fp, "      <CellData Scalars=\"rho\" Vectors=\"velocity_phys\">\n");
      }
    }

    // Helper: gather one scalar field, reorder on rank 0, write, free
    auto write_scalar = [&](const char* name, auto pack_fn) {
      std::vector<double> lbuf(nlocal);
      int c = 0;
      for (int k = 0; k < nlz; k++)
        for (int j = 0; j < nly; j++)
          for (int i = 0; i < nlx; i++)
            lbuf[c++] = pack_fn(grid.lidx(i, j, k));

      std::vector<double> gbuf;
      if (rank == 0) gbuf.resize(total_gathered);
      MPI_Gatherv(lbuf.data(), nlocal, MPI_DOUBLE,
                  gbuf.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

      if (rank == 0 && fp) {
        std::vector<double> field(ntot);
        for (int r = 0; r < nprocs; r++) {
          const int ox=all_info[r*6], oy=all_info[r*6+1], oz=all_info[r*6+2];
          const int rnx=all_info[r*6+3], rny=all_info[r*6+4], rnz=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz; kk++)
            for (int jj=0; jj<rny; jj++)
              for (int ii=0; ii<rnx; ii++) {
                const size_t gn = (size_t)(ox+ii) + (size_t)Nx*((size_t)(oy+jj) + (size_t)Ny*(oz+kk));
                field[gn] = gbuf[displs[r] + idx++];
              }
        }
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", name);
        for (size_t n = 0; n < ntot; n++) fprintf(fp, "%.8e\n", field[n]);
        fprintf(fp, "        </DataArray>\n");
      }
      // gbuf and field freed here
    };

    // Helper: gather one vector field (3 components), reorder, write, free
    auto write_vector = [&](const char* name, auto pack_x, auto pack_y, auto pack_z) {
      std::vector<double> lx(nlocal), ly(nlocal), lz(nlocal);
      int c = 0;
      for (int k = 0; k < nlz; k++)
        for (int j = 0; j < nly; j++)
          for (int i = 0; i < nlx; i++) {
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
          const int ox=all_info[r*6], oy=all_info[r*6+1], oz=all_info[r*6+2];
          const int rnx=all_info[r*6+3], rny=all_info[r*6+4], rnz=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz; kk++)
            for (int jj=0; jj<rny; jj++)
              for (int ii=0; ii<rnx; ii++) {
                const size_t gn = (size_t)(ox+ii) + (size_t)Nx*((size_t)(oy+jj) + (size_t)Ny*(oz+kk));
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
      for (int k = 0; k < nlz; k++)
        for (int j = 0; j < nly; j++)
          for (int i = 0; i < nlx; i++)
            lbuf[c++] = pack_fn(grid.lidx(i, j, k));

      std::vector<int> gbuf;
      if (rank == 0) gbuf.resize(total_gathered);
      MPI_Gatherv(lbuf.data(), nlocal, MPI_INT,
                  gbuf.data(), counts.data(), displs.data(), MPI_INT, 0, comm);

      if (rank == 0 && fp) {
        std::vector<int> field(ntot);
        for (int r = 0; r < nprocs; r++) {
          const int ox=all_info[r*6], oy=all_info[r*6+1], oz=all_info[r*6+2];
          const int rnx=all_info[r*6+3], rny=all_info[r*6+4], rnz=all_info[r*6+5];
          int idx = 0;
          for (int kk=0; kk<rnz; kk++)
            for (int jj=0; jj<rny; jj++)
              for (int ii=0; ii<rnx; ii++) {
                const size_t gn = (size_t)(ox+ii) + (size_t)Nx*((size_t)(oy+jj) + (size_t)Ny*(oz+kk));
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
      fprintf(stdout, "CoupLB: VTK written at step %ld -> %s\n", step, fname);
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
      char vti_name[512];
      snprintf(vti_name, sizeof(vti_name), "%s_%06ld.vti",
               vtk_prefix.c_str(), steps[i]);
      // Extract basename (strip directory path if present)
      const char* base = strrchr(vti_name, '/');
      base = base ? base + 1 : vti_name;

      const double time = steps[i] * dt;
      fprintf(fp, "    <DataSet timestep=\"%.8e\" file=\"%s\"/>\n",
              time, base);
    }

    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
  }

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