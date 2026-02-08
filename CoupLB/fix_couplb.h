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
#include <memory>
#include <string>

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
  int Nx, Ny, Nz;
  double dx, tau, rho0, nu;
  int nsub;
  double dx_phys, dt_lbm, force_scale, vel_scale;
  double gx_ext, gy_ext, gz_ext;
  int wall_ylo, wall_yhi, wall_zlo, wall_zhi;
  double wall_vel[3];
  double domain_lo[3], domain_hi[3];

  int output_every;
  std::string output_file;
  bigint next_output;

  int check_every;
  bigint next_check;
  bigint lbm_step_count;

  CoupLB::DeltaKernel ibm_kernel;
  bool is3d;

  std::unique_ptr<CoupLB::Grid<CoupLB::D3Q19>>      grid3d;
  std::unique_ptr<CoupLB::BGK<CoupLB::D3Q19>>       bgk3d;
  std::unique_ptr<CoupLB::Streaming<CoupLB::D3Q19>>  stream3d;

  std::unique_ptr<CoupLB::Grid<CoupLB::D2Q9>>       grid2d;
  std::unique_ptr<CoupLB::BGK<CoupLB::D2Q9>>        bgk2d;
  std::unique_ptr<CoupLB::Streaming<CoupLB::D2Q9>>   stream2d;

  MPI_Comm ycomm;
  bool ycomm_valid;

  void setup_grid();
  void setup_boundaries();
  void setup_ycomm();
  void lbm_step();
  void ibm_coupling();

  template <typename L> void do_lbm_step(CoupLB::Grid<L>&, CoupLB::BGK<L>&, CoupLB::Streaming<L>&);
  template <typename L> void do_ibm_coupling(CoupLB::Grid<L>&, CoupLB::Streaming<L>&);
  template <typename L> void apply_external_force(CoupLB::Grid<L>&);
  template <typename L> void enforce_wall_ghost_fields(CoupLB::Grid<L>&);
  template <typename L> void check_stability(CoupLB::Grid<L>&);
  template <typename L> void write_profile(CoupLB::Grid<L>&, bigint);
};

} // namespace LAMMPS_NS

#endif // FIX_COUPLB_H
#endif // FIX_CLASS
