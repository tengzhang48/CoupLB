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

  void collide(Grid<Lattice>& grid) {
    constexpr int Q = Lattice::Q;
    constexpr double cs2 = Lattice::cs2;
    constexpr double cs4 = cs2 * cs2;
    const double om = omega;
    const double half_om = 1.0 - 0.5 * om;
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
          const double uu = u0*u0 + u1*u1 + u2*u2;
          const double uf = u0*f0 + u1*f1 + u2*f2;

          for (int q = 0; q < Q; q++) {
            const double cx = Lattice::e[q][0];
            const double cy = Lattice::e[q][1];
            const double cz = Lattice::e[q][2];
            const double w  = Lattice::w[q];

            // Dot products computed once per direction
            const double eu = cx*u0 + cy*u1 + cz*u2;
            const double ef = cx*f0 + cy*f1 + cz*f2;

            // Equilibrium
            const double feq = w * r * (1.0 + eu/cs2 + 0.5*eu*eu/cs4 - 0.5*uu/cs2);

            // Guo source (inlined)
            const double Si = w * half_om * ((ef - uf)/cs2 + eu*ef/cs4);

            grid.fi(q, n) += -om * (grid.fi(q, n) - feq) + Si;
          }
        }
      }
    }
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_COLLISION_H