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
                        double vel_scale = 1.0, double force_scale = 1.0)
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

    // Pack local interior data into flat arrays
    // Layout: x fastest, then y, then z (VTK ImageData default)
    std::vector<double> l_rho(nlocal), l_ux(nlocal), l_uy(nlocal), l_uz(nlocal);
    std::vector<double> l_fx(nlocal), l_fy(nlocal), l_fz(nlocal);
    std::vector<int>    l_type(nlocal);

    int c = 0;
    for (int k = 0; k < nlz; k++)
      for (int j = 0; j < nly; j++)
        for (int i = 0; i < nlx; i++) {
          const int n = grid.lidx(i, j, k);
          l_rho[c]  = grid.rho[n];
          l_ux[c]   = grid.ux[n] * vel_scale;
          l_uy[c]   = grid.uy[n] * vel_scale;
          l_uz[c]   = grid.uz[n] * vel_scale;
          l_fx[c]   = grid.fx[n] * force_scale;
          l_fy[c]   = grid.fy[n] * force_scale;
          l_fz[c]   = grid.fz[n] * force_scale;
          l_type[c] = grid.type[n];
          c++;
        }

    // Gather metadata: each rank sends (ox, oy, oz, nlx, nly, nlz)
    int local_info[6] = {grid.offset[0], grid.offset[1],
                         (Lattice::D == 3) ? grid.offset[2] : 0,
                         nlx, nly, nlz};
    std::vector<int> all_info(nprocs * 6);
    MPI_Gather(local_info, 6, MPI_INT, all_info.data(), 6, MPI_INT, 0, comm);

    // Gather local counts for variable-size Gatherv
    std::vector<int> counts(nprocs), displs(nprocs);
    MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
      displs[0] = 0;
      for (int r = 1; r < nprocs; r++)
        displs[r] = displs[r-1] + counts[r-1];
    }

    // Total gathered size (only meaningful on rank 0)
    int total_gathered = 0;
    if (rank == 0) {
      for (int r = 0; r < nprocs; r++) total_gathered += counts[r];
    }

    // Gather all fields to rank 0
    std::vector<double> g_rho, g_ux, g_uy, g_uz, g_fx, g_fy, g_fz;
    std::vector<int> g_type;
    if (rank == 0) {
      g_rho.resize(total_gathered);
      g_ux.resize(total_gathered);  g_uy.resize(total_gathered);  g_uz.resize(total_gathered);
      g_fx.resize(total_gathered);  g_fy.resize(total_gathered);  g_fz.resize(total_gathered);
      g_type.resize(total_gathered);
    }

    MPI_Gatherv(l_rho.data(), nlocal, MPI_DOUBLE, g_rho.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_ux.data(),  nlocal, MPI_DOUBLE, g_ux.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_uy.data(),  nlocal, MPI_DOUBLE, g_uy.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_uz.data(),  nlocal, MPI_DOUBLE, g_uz.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_fx.data(),  nlocal, MPI_DOUBLE, g_fx.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_fy.data(),  nlocal, MPI_DOUBLE, g_fy.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_fz.data(),  nlocal, MPI_DOUBLE, g_fz.data(),  counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(l_type.data(), nlocal, MPI_INT,   g_type.data(), counts.data(), displs.data(), MPI_INT,    0, comm);

    // Rank 0: reassemble gathered chunks into global arrays and write
    if (rank == 0) {
      std::vector<double> rho(ntot), ux(ntot), uy(ntot), uz(ntot);
      std::vector<double> fx(ntot), fy(ntot), fz(ntot);
      std::vector<int>    type(ntot);

      for (int r = 0; r < nprocs; r++) {
        const int ox = all_info[r*6+0], oy = all_info[r*6+1], oz = all_info[r*6+2];
        const int rnx = all_info[r*6+3], rny = all_info[r*6+4], rnz = all_info[r*6+5];
        const int base = displs[r];
        int idx = 0;
        for (int k = 0; k < rnz; k++)
          for (int j = 0; j < rny; j++)
            for (int i = 0; i < rnx; i++) {
              const int gi = ox + i, gj = oy + j, gk = oz + k;
              const size_t gn = (size_t)gi + (size_t)Nx * ((size_t)gj + (size_t)Ny * gk);
              rho[gn]  = g_rho[base + idx];
              ux[gn]   = g_ux[base + idx];
              uy[gn]   = g_uy[base + idx];
              uz[gn]   = g_uz[base + idx];
              fx[gn]   = g_fx[base + idx];
              fy[gn]   = g_fy[base + idx];
              fz[gn]   = g_fz[base + idx];
              type[gn] = g_type[base + idx];
              idx++;
            }
      }

      // Free gathered buffers
      g_rho.clear(); g_ux.clear(); g_uy.clear(); g_uz.clear();
      g_fx.clear();  g_fy.clear();  g_fz.clear();  g_type.clear();

      // Write single .vti file
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_%06ld.vti", prefix.c_str(), step);

      FILE* fp = fopen(fname, "w");
      if (!fp) {
        fprintf(stderr, "CoupLB IO: cannot open %s\n", fname);
        return;
      }

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

      // rho
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n");
      for (size_t n = 0; n < ntot; n++)
        fprintf(fp, "%.8e\n", rho[n]);
      fprintf(fp, "        </DataArray>\n");

      // velocity (physical units)
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity_phys\" "
                  "NumberOfComponents=\"3\" format=\"ascii\">\n");
      for (size_t n = 0; n < ntot; n++)
        fprintf(fp, "%.8e %.8e %.8e\n", ux[n], uy[n], uz[n]);
      fprintf(fp, "        </DataArray>\n");

      // type
      fprintf(fp, "        <DataArray type=\"Int32\" Name=\"type\" format=\"ascii\">\n");
      for (size_t n = 0; n < ntot; n++)
        fprintf(fp, "%d\n", type[n]);
      fprintf(fp, "        </DataArray>\n");

      // force (physical units)
      fprintf(fp, "        <DataArray type=\"Float64\" Name=\"force_phys\" "
                  "NumberOfComponents=\"3\" format=\"ascii\">\n");
      for (size_t n = 0; n < ntot; n++)
        fprintf(fp, "%.8e %.8e %.8e\n", fx[n], fy[n], fz[n]);
      fprintf(fp, "        </DataArray>\n");

      fprintf(fp, "      </CellData>\n");
      fprintf(fp, "    </Piece>\n");
      fprintf(fp, "  </ImageData>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);

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