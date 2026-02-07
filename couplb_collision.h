#ifndef COUPLB_COLLISION_H
#define COUPLB_COLLISION_H

#include "couplb_lattice.h"

namespace LAMMPS_NS {
namespace CoupLB {

template <typename Lattice>
class BGK {
public:
  double tau, omega;

  BGK() : tau(1.0), omega(1.0) {}
  void set_tau(double t) { tau = t; omega = 1.0 / t; }
  void set_viscosity(double nu) { tau = nu / Lattice::cs2 + 0.5; omega = 1.0 / tau; }

  inline double guo_source(int q,
                           double ux_, double uy_, double uz_,
                           double fx_, double fy_, double fz_) const {
    constexpr double cs2 = Lattice::cs2;
    constexpr double cs4 = cs2 * cs2;
    const int ex = Lattice::e[q][0], ey = Lattice::e[q][1], ez = Lattice::e[q][2];
    const double eu = ex*ux_ + ey*uy_ + ez*uz_;
    const double ef = ex*fx_ + ey*fy_ + ez*fz_;
    const double uf = ux_*fx_ + uy_*fy_ + uz_*fz_;
    return Lattice::w[q] * (1.0 - 0.5 * omega) * ((ef - uf) / cs2 + eu * ef / cs4);
  }

  void collide(Grid<Lattice>& grid) {
    constexpr int Q = Lattice::Q;
    const int klo = (Lattice::D == 3) ? 1 : 0;
    const int khi = (Lattice::D == 3) ? (grid.gz - 2) : 0;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int k = klo; k <= khi; k++) {
      for (int j = 1; j <= grid.gy - 2; j++) {
        for (int i = 1; i <= grid.gx - 2; i++) {
          const int n = grid.idx(i, j, k);
          if (grid.type[n] != 0) continue;
          const double r  = grid.rho[n];
          const double u0 = grid.ux[n], u1 = grid.uy[n], u2 = grid.uz[n];
          const double f0 = grid.fx[n], f1 = grid.fy[n], f2 = grid.fz[n];
          for (int q = 0; q < Q; q++) {
            const double feq = grid.feq(q, r, u0, u1, u2);
            const double Si  = guo_source(q, u0, u1, u2, f0, f1, f2);
            grid.fi(q, n) += -omega * (grid.fi(q, n) - feq) + Si;
          }
        }
      }
    }
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_COLLISION_H
