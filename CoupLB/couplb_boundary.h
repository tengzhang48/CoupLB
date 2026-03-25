#ifndef COUPLB_BOUNDARY_H
#define COUPLB_BOUNDARY_H

#include "couplb_lattice.h"

namespace LAMMPS_NS {
namespace CoupLB {

template <typename Lattice>
class Boundary {
public:
  static void set_walls_x(Grid<Lattice>& g, bool lo, bool hi, int tlo, int thi) {
    if (lo) for(int k=0;k<g.gz;k++) for(int j=0;j<g.gy;j++) {
      int n=g.idx(0,j,k); g.type[n]=tlo; g.wall_dim[n]=0;
    }
    if (hi) for(int k=0;k<g.gz;k++) for(int j=0;j<g.gy;j++) {
      int n=g.idx(g.gx-1,j,k); g.type[n]=thi; g.wall_dim[n]=0;
    }
  }
  static void set_walls_y(Grid<Lattice>& g, bool lo, bool hi, int tlo, int thi) {
    if (lo) for(int k=0;k<g.gz;k++) for(int i=0;i<g.gx;i++) {
      int n=g.idx(i,0,k); g.type[n]=tlo; g.wall_dim[n]=1;
    }
    if (hi) for(int k=0;k<g.gz;k++) for(int i=0;i<g.gx;i++) {
      int n=g.idx(i,g.gy-1,k); g.type[n]=thi; g.wall_dim[n]=1;
    }
  }
  static void set_walls_z(Grid<Lattice>& g, bool lo, bool hi, int tlo, int thi) {
    if (lo) for(int j=0;j<g.gy;j++) for(int i=0;i<g.gx;i++) {
      int n=g.idx(i,j,0); g.type[n]=tlo; g.wall_dim[n]=2;
    }
    if (hi) for(int j=0;j<g.gy;j++) for(int i=0;i<g.gx;i++) {
      int n=g.idx(i,j,g.gz-1); g.type[n]=thi; g.wall_dim[n]=2;
    }
  }
  static void set_wall_velocity(Grid<Lattice>& g, int wt, double vx, double vy, double vz) {
    for(int n=0;n<g.ntotal;n++) if(g.type[n]==wt) { g.bc_ux[n]=vx; g.bc_uy[n]=vy; g.bc_uz[n]=vz; }
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_BOUNDARY_H
