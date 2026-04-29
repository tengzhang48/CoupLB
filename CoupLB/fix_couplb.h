// clang-format off
#ifdef FIX_CLASS
FixStyle(couplb,FixCoupLB);
#else

#ifndef FIX_COUPLB_H
#define FIX_COUPLB_H

#include "fix.h"
#include "couplb_lattice.h"
#include "couplb_collision.h"
#include "couplb_streaming.h"
#include "couplb_boundary.h"
#include "couplb_ibm.h"
#include "couplb_io.h"
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixCoupLB : public Fix {
public:
  FixCoupLB(class LAMMPS *, int, char **);
  ~FixCoupLB() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void end_of_step() override;

private:
  enum { NONE_G, CONSTANT_G, EQUAL_G };

  int Nx, Ny, Nz;
  double dx, inv_dx;
  double nu;
  double rho;
  double nu_lb;
  static constexpr double dx_lb = 1.0;
  static constexpr double rho_lb = 1.0;
  double tau;
  int md_per_lb;
  int md_sub_count;
  double xi_ibm;

  double dt_LBM;
  double force_scale;
  double vel_scale;

  double gx_ext, gy_ext, gz_ext;
  char *gxstr, *gystr, *gzstr;
  int gxvar, gyvar, gzvar;
  int gxstyle, gystyle, gzstyle;

  double wall_vel[3];
  int wall_xlo, wall_xhi, wall_ylo, wall_yhi, wall_zlo, wall_zhi;
  double domain_lo[3], domain_hi[3];

  int output_every;
  std::string output_file;
  bigint next_output;

  int check_every;
  bigint next_check;
  bigint lbm_step_count;

  int vtk_every;
  std::string vtk_prefix;
  std::string vtk_pvd_file;
  std::vector<long> vtk_steps;
  bigint next_vtk;

  // Solid VTK output (VTK PolyData .vtp)
  bool vtk_solid_on;
  std::vector<std::string> vtk_solid_attrs;
  std::string vtk_solid_pvd_file;

  // VTK region clipping (physical coordinates)
  bool has_vtk_region;
  double vtk_rlo[3], vtk_rhi[3];   // user-specified physical bounds
  int vtk_region[6];                // node indices: {ilo, ihi, jlo, jhi, klo, khi}

  int checkpoint_every;
  std::string checkpoint_prefix;
  bigint next_checkpoint;

  std::string restart_prefix;
  bool do_restart;

  CoupLB::DeltaKernel ibm_kernel;
  bool is3d;

  std::unique_ptr<CoupLB::Grid<CoupLB::D3Q19>>      grid3d;
  std::unique_ptr<CoupLB::BGK<CoupLB::D3Q19>>       bgk3d;
  std::unique_ptr<CoupLB::Streaming<CoupLB::D3Q19>> stream3d;

  std::unique_ptr<CoupLB::Grid<CoupLB::D2Q9>>       grid2d;
  std::unique_ptr<CoupLB::BGK<CoupLB::D2Q9>>        bgk2d;
  std::unique_ptr<CoupLB::Streaming<CoupLB::D2Q9>>  stream2d;

  MPI_Comm ycomm;
  bool ycomm_valid;

  bool ibm_has_particles;

  void setup_grid();
  void setup_boundaries(const double wv_lb[3]);
  void setup_ycomm();
  void lbm_step();
  void update_gravity_variables();
  void ibm_sub_coupling();

  template <typename L> void do_lbm_step(CoupLB::Grid<L>&, CoupLB::BGK<L>&, CoupLB::Streaming<L>&);
  template <typename L> void do_ibm_sub_coupling(CoupLB::Grid<L>&, CoupLB::Streaming<L>&, bool first);
  template <typename L> void apply_external_force(CoupLB::Grid<L>&);
  template <typename L> void enforce_wall_ghost_fields(CoupLB::Grid<L>&);
  template <typename L> void check_stability(CoupLB::Grid<L>&);
  template <typename L> void check_stability_precomputed(CoupLB::Grid<L>&);
  template <typename L> void write_profile(CoupLB::Grid<L>&, bigint);
};

} // namespace LAMMPS_NS

#endif
#endif
