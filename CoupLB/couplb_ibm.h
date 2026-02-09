#ifndef COUPLB_IBM_H
#define COUPLB_IBM_H

#include "couplb_lattice.h"
#include <cmath>
#include <array>

namespace LAMMPS_NS {
namespace CoupLB {

inline double delta_roma(double r) {
  const double ar = std::fabs(r);
  if (ar <= 0.5) {
    const double arg = std::max(1.0 - 3.0*r*r, 0.0);
    return (1.0 + std::sqrt(arg)) / 3.0;
  }
  if (ar <= 1.5) {
    const double arg = std::max(-2.0 + 6.0*ar - 3.0*r*r, 0.0);
    return (5.0 - 3.0*ar - std::sqrt(arg)) / 6.0;
  }
  return 0.0;
}

inline double delta_peskin4(double r) {
  const double ar = std::fabs(r);
  if (ar <= 1.0) {
    const double arg = std::max(1.0 + 4.0*ar - 4.0*ar*ar, 0.0);
    return (3.0 - 2.0*ar + std::sqrt(arg)) / 8.0;
  }
  if (ar <= 2.0) {
    const double arg = std::max(-7.0 + 12.0*ar - 4.0*ar*ar, 0.0);
    return (5.0 - 2.0*ar - std::sqrt(arg)) / 8.0;
  }
  return 0.0;
}

enum class DeltaKernel { ROMA, PESKIN4 };

template <typename Lattice>
class IBM {
public:
  static inline double delta_product(DeltaKernel k, double rx, double ry, double rz) {
    auto d = (k == DeltaKernel::ROMA) ? delta_roma : delta_peskin4;
    return d(rx) * d(ry) * ((Lattice::D == 3) ? d(rz) : 1.0);
  }

  // Convert physical coordinates to ghost-inclusive local grid indices.
  // dx_phys is the PHYSICAL grid spacing (not lattice dx=1).
  static inline void phys_to_local(const Grid<Lattice>& g, double xp, double yp, double zp,
                                   const double lo[3], double dx_phys,
                                   double& lx, double& ly, double& lz) {
    lx = (xp-lo[0])/dx_phys - g.offset[0] + 1.0;
    ly = (yp-lo[1])/dx_phys - g.offset[1] + 1.0;
    lz = (Lattice::D==3) ? ((zp-lo[2])/dx_phys - g.offset[2] + 1.0) : 0.0;
  }

  static std::array<double,3> interpolate(const Grid<Lattice>& g,
      double xp, double yp, double zp, const double lo[3],
      double dx_phys,
      DeltaKernel kern = DeltaKernel::ROMA)
  {
    double lx, ly, lz;
    phys_to_local(g, xp, yp, zp, lo, dx_phys, lx, ly, lz);
    const int i0=(int)std::floor(lx), j0=(int)std::floor(ly);
    const int k0=(Lattice::D==3)?(int)std::floor(lz):0;

    double ui=0, vi=0, wi=0, ws=0;
    for (int k=((Lattice::D==3)?k0-1:0); k<=((Lattice::D==3)?k0+2:0); k++) {
      if (Lattice::D==3 && (k<0||k>=g.gz)) continue;
      for (int j=j0-1; j<=j0+2; j++) {
        if (j<0||j>=g.gy) continue;
        for (int i=i0-1; i<=i0+2; i++) {
          if (i<0||i>=g.gx) continue;
          const int n = g.idx(i,j,k);
          const int nt = g.type[n];
          const double d = delta_product(kern, lx-i, ly-j, lz-k);
          if (nt==0) { ui+=g.ux[n]*d; vi+=g.uy[n]*d; wi+=g.uz[n]*d; ws+=d; }
          else if (nt==1) { ws+=d; }
          else if (nt==2) { ui+=g.bc_ux[n]*d; vi+=g.bc_uy[n]*d; wi+=g.bc_uz[n]*d; ws+=d; }
        }
      }
    }
    if (ws > Constants::DELTA_TOL) { const double iw=1.0/ws; return {ui*iw, vi*iw, wi*iw}; }
    return {0.0, 0.0, 0.0};
  }

  static void spread(Grid<Lattice>& g, double xp, double yp, double zp,
      double fxl, double fyl, double fzl, double dvl, const double lo[3],
      double dx_phys,
      DeltaKernel kern = DeltaKernel::ROMA)
  {
    double lx, ly, lz;
    phys_to_local(g, xp, yp, zp, lo, dx_phys, lx, ly, lz);
    const int i0=(int)std::floor(lx), j0=(int)std::floor(ly);
    const int k0=(Lattice::D==3)?(int)std::floor(lz):0;

    // Volume ratio: dvl is Lagrangian volume, dx_phys^D is Eulerian cell volume
    const double cell_vol = dx_phys * dx_phys * ((Lattice::D==3) ? dx_phys : 1.0);
    const double dvr = dvl / cell_vol;

    double ws = 0.0;
    for (int k=((Lattice::D==3)?k0-1:0); k<=((Lattice::D==3)?k0+2:0); k++) {
      if (Lattice::D==3 && (k<0||k>=g.gz)) continue;
      for (int j=j0-1;j<=j0+2;j++) { if(j<0||j>=g.gy) continue;
        for (int i=i0-1;i<=i0+2;i++) { if(i<0||i>=g.gx) continue;
          if (g.type[g.idx(i,j,k)]!=0) continue;
          ws += delta_product(kern, lx-i, ly-j, lz-k);
        }
      }
    }
    if (ws < Constants::DELTA_TOL) return;

    const double sc = dvr / ws;
    for (int k=((Lattice::D==3)?k0-1:0); k<=((Lattice::D==3)?k0+2:0); k++) {
      if (Lattice::D==3 && (k<0||k>=g.gz)) continue;
      for (int j=j0-1;j<=j0+2;j++) { if(j<0||j>=g.gy) continue;
        for (int i=i0-1;i<=i0+2;i++) { if(i<0||i>=g.gx) continue;
          const int n = g.idx(i,j,k);
          if (g.type[n]!=0) continue;
          const double w = delta_product(kern, lx-i, ly-j, lz-k) * sc;
          g.fx[n] += fxl*w; g.fy[n] += fyl*w; g.fz[n] += fzl*w;
        }
      }
    }
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_IBM_H