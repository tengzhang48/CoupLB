#ifndef COUPLB_STREAMING_H
#define COUPLB_STREAMING_H

#include "couplb_lattice.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cassert>

namespace LAMMPS_NS {
namespace CoupLB {

namespace Tags {
  constexpr int DIST_BASE  = 5000;
  constexpr int VEL_BASE   = 6000;
  constexpr int FORCE_BASE = 7000;   // reverse force exchange
}

template <typename Lattice>
class Streaming {
public:
  static constexpr int NVEL_FIELDS = 4;

  int neigh[3][2];
  bool periodic[3];
  int nprocs[3];
  MPI_Comm comm;
  double rho0;

  std::vector<double> send_lo, send_hi, recv_lo, recv_hi;
  std::vector<double> vsend_lo, vsend_hi, vrecv_lo, vrecv_hi;
  size_t max_dist_buf, max_vel_buf;

  Streaming() : comm(MPI_COMM_WORLD), rho0(1.0), max_dist_buf(0), max_vel_buf(0), max_force_buf(0) {
    for (int d = 0; d < 3; d++) {
      neigh[d][0] = neigh[d][1] = MPI_PROC_NULL;
      periodic[d] = false; nprocs[d] = 1;
      wall_face[d][0] = wall_face[d][1] = false;
    }
  }

  void set_comm(MPI_Comm c) { comm = c; }
  void set_neighbors(int pn[3][2]) {
    for (int d=0;d<3;d++) { neigh[d][0]=pn[d][0]; neigh[d][1]=pn[d][1]; }
  }
  void set_periodic(bool px, bool py, bool pz) { periodic[0]=px; periodic[1]=py; periodic[2]=pz; }
  void set_nprocs(int nx, int ny, int nz) { nprocs[0]=nx; nprocs[1]=ny; nprocs[2]=nz; }

  // Precomputed wall flags: wall_face[dim][0]=lo ghost, wall_face[dim][1]=hi ghost
  bool wall_face[3][2];

  // Force exchange buffers (3 fields: fx, fy, fz)
  static constexpr int NFORCE_FIELDS = 3;
  std::vector<double> fsend_lo, fsend_hi, frecv_lo, frecv_hi;
  size_t max_force_buf;

  void allocate_buffers(const Grid<Lattice>& grid, double rho0_ref) {
    rho0 = rho0_ref;
    size_t mf = 0;
    const int nd = (Lattice::D == 3) ? 3 : 2;
    for (int d=0;d<nd;d++) { size_t fs = face_size(grid,d); if (fs>mf) mf=fs; }
    max_dist_buf  = static_cast<size_t>(Lattice::Q) * mf;
    max_vel_buf   = static_cast<size_t>(NVEL_FIELDS) * mf;
    max_force_buf = static_cast<size_t>(NFORCE_FIELDS) * mf;
    send_lo.resize(max_dist_buf); send_hi.resize(max_dist_buf);
    recv_lo.resize(max_dist_buf); recv_hi.resize(max_dist_buf);
    vsend_lo.resize(max_vel_buf); vsend_hi.resize(max_vel_buf);
    vrecv_lo.resize(max_vel_buf); vrecv_hi.resize(max_vel_buf);
    fsend_lo.resize(max_force_buf); fsend_hi.resize(max_force_buf);
    frecv_lo.resize(max_force_buf); frecv_hi.resize(max_force_buf);
  }

  // Precompute wall face flags. Call once after setup_boundaries().
  void precompute_wall_flags(const Grid<Lattice>& grid) {
    const int nd = (Lattice::D == 3) ? 3 : 2;
    for (int d=0;d<nd;d++) {
      wall_face[d][0] = has_wall_on_face(grid, d, 0);
      wall_face[d][1] = has_wall_on_face(grid, d, dim_ext(grid,d)-1);
    }
  }

  void stream(Grid<Lattice>& grid) {
    constexpr int Q = Lattice::Q;
    constexpr double cs2 = Lattice::cs2;
    const int gx = grid.gx, gy = grid.gy, gz = grid.gz;
    const double rho_floor = std::max(Constants::RHO_CLAMP_FRAC * rho0, Constants::RHO_MIN);
    const int klo = (Lattice::D == 3) ? 1 : 0;
    const int khi = (Lattice::D == 3) ? (gz - 2) : 0;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int k = klo; k <= khi; k++) {
      for (int j = 1; j <= gy - 2; j++) {
        for (int i = 1; i <= gx - 2; i++) {
          const int n = grid.idx(i, j, k);
          if (grid.type[n] != 0) continue;
          for (int q = 0; q < Q; q++) {
            const int si = i - Lattice::e[q][0];
            const int sj = j - Lattice::e[q][1];
            const int sk = k - Lattice::e[q][2];
            if (si<0||si>=gx||sj<0||sj>=gy||((Lattice::D==3)&&(sk<0||sk>=gz))) {
              grid.fi_buf(q, n) = grid.fi(q, n); continue;
            }
            const int sn = grid.idx(si, sj, sk);
            const int st = grid.type[sn];
            if (st == 0) {
              grid.fi_buf(q, n) = grid.fi(q, sn);
            } else if (st == 1) {
              grid.fi_buf(q, n) = grid.fi(Lattice::opp[q], n);
            } else if (st == 2) {
              const int qo = Lattice::opp[q];
              const double eu = Lattice::e[q][0]*grid.bc_ux[sn]
                              + Lattice::e[q][1]*grid.bc_uy[sn]
                              + Lattice::e[q][2]*grid.bc_uz[sn];
              const double rho_floor = std::max(Constants::RHO_CLAMP_FRAC * rho0, Constants::RHO_MIN);
              const double rw = std::max(grid.rho[n], rho_floor);
              
              grid.fi_buf(q, n) = grid.fi(qo, n) + 2.0*Lattice::w[qo]*rw*eu/cs2;              
            } else if (st == 3) {
              // Free-slip: specular reflection (flip wall-normal component only)
              const int qr = Lattice::reflect[grid.wall_dim[sn]][q];
              grid.fi_buf(q, n) = grid.fi(qr, n);
            } else if (st == 4) {
              // Open boundary: zero-gradient extrapolation.
              // Unknown incoming population copied from current interior node.
              // Lets momentum leave the domain (not conserved).
              grid.fi_buf(q, n) = grid.fi(q, n);
            }
          }
        }
      }
    }
    grid.f.swap(grid.f_buf);
  }

  void exchange(Grid<Lattice>& g) {
    exchange_dim(g,0); exchange_dim(g,1);
    if (Lattice::D==3) exchange_dim(g,2);
  }
  void exchange_velocity(Grid<Lattice>& g) {
    exchange_vel_dim(g,0); exchange_vel_dim(g,1);
    if (Lattice::D==3) exchange_vel_dim(g,2);
  }

  // Reverse force exchange: accumulate forces from ghost cells back
  // to the owning proc's interior cells. This is needed because IBM
  // spreading can deposit forces on ghost cells when a particle's
  // delta stencil extends across a subdomain boundary. Without this,
  // those forces are silently lost.
  //
  // Communication pattern is the REVERSE of the velocity exchange:
  //   - Pack ghost layer forces
  //   - Send ghost_lo to neighbor_lo, ghost_hi to neighbor_hi
  //   - Unpack by ADDING (not overwriting) to interior face
  void exchange_forces(Grid<Lattice>& g) {
    exchange_force_dim(g,0); exchange_force_dim(g,1);
    if (Lattice::D==3) exchange_force_dim(g,2);
  }

private:
  bool is_self_periodic(int d) const { return periodic[d] && nprocs[d]==1; }

  bool has_wall_on_face(const Grid<Lattice>& g, int dim, int pos) const {
    const int gx=g.gx, gy=g.gy, gz=g.gz;
    if (dim==0) { for(int k=0;k<gz;k++) for(int j=0;j<gy;j++) if(g.type[g.idx(pos,j,k)]==0) return false; }
    else if (dim==1) { for(int k=0;k<gz;k++) for(int i=0;i<gx;i++) if(g.type[g.idx(i,pos,k)]==0) return false; }
    else { for(int j=0;j<gy;j++) for(int i=0;i<gx;i++) if(g.type[g.idx(i,j,pos)]==0) return false; }
    return true;
  }

  size_t face_size(const Grid<Lattice>& g, int d) const {
    if(d==0) return size_t(g.gy)*g.gz; if(d==1) return size_t(g.gx)*g.gz; return size_t(g.gx)*g.gy;
  }
  int dim_ext(const Grid<Lattice>& g, int d) const { return d==0?g.gx:d==1?g.gy:g.gz; }

  void exchange_dim(Grid<Lattice>& g, int d) {
    const int ds=dim_ext(g,d), lf=1, hf=ds-2, gl=0, gh=ds-1;
    if (is_self_periodic(d) && ds>=4) {
      if (!wall_face[d][0]) copy_face_f(g,d,hf,gl);
      if (!wall_face[d][1]) copy_face_f(g,d,lf,gh);
      return;
    }
    if (neigh[d][0]==MPI_PROC_NULL && neigh[d][1]==MPI_PROC_NULL) return;
    const size_t bs = size_t(Lattice::Q)*face_size(g,d);
    if (!bs) return; assert(bs<=max_dist_buf);
    const int c = int(bs);
    pack_face_f(g,d,lf,send_lo); pack_face_f(g,d,hf,send_hi);
    MPI_Sendrecv(send_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::DIST_BASE+d,
                 recv_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::DIST_BASE+d,comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::DIST_BASE+3+d,
                 recv_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::DIST_BASE+3+d,comm,MPI_STATUS_IGNORE);
    if (neigh[d][0]!=MPI_PROC_NULL && !wall_face[d][0]) unpack_face_f(g,d,gl,recv_lo);
    if (neigh[d][1]!=MPI_PROC_NULL && !wall_face[d][1]) unpack_face_f(g,d,gh,recv_hi);
  }

  void exchange_vel_dim(Grid<Lattice>& g, int d) {
    const int ds=dim_ext(g,d), lf=1, hf=ds-2, gl=0, gh=ds-1;
    if (is_self_periodic(d) && ds>=4) {
      if (!wall_face[d][0]) copy_face_vel(g,d,hf,gl);
      if (!wall_face[d][1]) copy_face_vel(g,d,lf,gh);
      return;
    }
    if (neigh[d][0]==MPI_PROC_NULL && neigh[d][1]==MPI_PROC_NULL) return;
    const size_t bs = size_t(NVEL_FIELDS)*face_size(g,d);
    if (!bs) return; assert(bs<=max_vel_buf);
    const int c = int(bs);
    pack_face_vel(g,d,lf,vsend_lo); pack_face_vel(g,d,hf,vsend_hi);
    MPI_Sendrecv(vsend_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::VEL_BASE+d,
                 vrecv_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::VEL_BASE+d,comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(vsend_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::VEL_BASE+3+d,
                 vrecv_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::VEL_BASE+3+d,comm,MPI_STATUS_IGNORE);
    if (neigh[d][0]!=MPI_PROC_NULL && !wall_face[d][0]) unpack_face_vel(g,d,gl,vrecv_lo);
    if (neigh[d][1]!=MPI_PROC_NULL && !wall_face[d][1]) unpack_face_vel(g,d,gh,vrecv_hi);
  }

  // Reverse force exchange: ghost --> interior (additive)
  void exchange_force_dim(Grid<Lattice>& g, int d) {
    const int ds=dim_ext(g,d), lf=1, hf=ds-2, gl=0, gh=ds-1;

    // Self-periodic: ghost forces accumulate onto opposite interior face
    if (is_self_periodic(d) && ds>=4) {
      accum_face_force(g, d, gl, lf);   // ghost_lo --> interior lo_face
      accum_face_force(g, d, gh, hf);   // ghost_hi --> interior hi_face
      return;
    }
    if (neigh[d][0]==MPI_PROC_NULL && neigh[d][1]==MPI_PROC_NULL) return;
    const size_t bs = size_t(NFORCE_FIELDS)*face_size(g,d);
    if (!bs) return; assert(bs<=max_force_buf);
    const int c = int(bs);

    // Pack ghost layers (reverse direction: send ghosts, receive into interior)
    pack_face_force(g,d,gl,fsend_lo);   // pack ghost_lo
    pack_face_force(g,d,gh,fsend_hi);   // pack ghost_hi

    // Reverse: send ghost_lo to neighbor_lo (who owns those cells as interior)
    //          recv from neighbor_hi (their ghost_hi = our interior hi_face)
    MPI_Sendrecv(fsend_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::FORCE_BASE+d,
                 frecv_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::FORCE_BASE+d,comm,MPI_STATUS_IGNORE);
    MPI_Sendrecv(fsend_hi.data(),c,MPI_DOUBLE,neigh[d][1],Tags::FORCE_BASE+3+d,
                 frecv_lo.data(),c,MPI_DOUBLE,neigh[d][0],Tags::FORCE_BASE+3+d,comm,MPI_STATUS_IGNORE);

    // Accumulate (add, not overwrite) into interior faces
    if (neigh[d][0]!=MPI_PROC_NULL) accum_unpack_force(g,d,lf,frecv_lo);
    if (neigh[d][1]!=MPI_PROC_NULL) accum_unpack_force(g,d,hf,frecv_hi);
  }

  // Pack / unpack / copy helpers
  void pack_face_f(const Grid<Lattice>& g, int d, int p, std::vector<double>& b) const {
    size_t c=0; iter(g,d,p,[&](int n){ for(int q=0;q<Lattice::Q;q++) b[c++]=g.fi(q,n); });
  }
  void unpack_face_f(Grid<Lattice>& g, int d, int p, const std::vector<double>& b) {
    size_t c=0; iter(g,d,p,[&](int n){ for(int q=0;q<Lattice::Q;q++) g.fi(q,n)=b[c++]; });
  }
  void copy_face_f(Grid<Lattice>& g, int d, int s, int t) {
    iter2(g,d,s,t,[&](int sn,int dn){ for(int q=0;q<Lattice::Q;q++) g.fi(q,dn)=g.fi(q,sn); });
  }
  void pack_face_vel(const Grid<Lattice>& g, int d, int p, std::vector<double>& b) const {
    size_t c=0; iter(g,d,p,[&](int n){ b[c++]=g.rho[n]; b[c++]=g.ux[n]; b[c++]=g.uy[n]; b[c++]=g.uz[n]; });
  }
  void unpack_face_vel(Grid<Lattice>& g, int d, int p, const std::vector<double>& b) {
    size_t c=0; iter(g,d,p,[&](int n){ g.rho[n]=b[c++]; g.ux[n]=b[c++]; g.uy[n]=b[c++]; g.uz[n]=b[c++]; });
  }
  void copy_face_vel(Grid<Lattice>& g, int d, int s, int t) {
    iter2(g,d,s,t,[&](int sn,int dn){ g.rho[dn]=g.rho[sn]; g.ux[dn]=g.ux[sn]; g.uy[dn]=g.uy[sn]; g.uz[dn]=g.uz[sn]; });
  }

  // Force helpers - reverse communication (ghost --> interior, additive)
  void pack_face_force(const Grid<Lattice>& g, int d, int p, std::vector<double>& b) const {
    size_t c=0; iter(g,d,p,[&](int n){ b[c++]=g.fx[n]; b[c++]=g.fy[n]; b[c++]=g.fz[n]; });
  }
  // Additive unpack: ACCUMULATE received forces onto interior face
  void accum_unpack_force(Grid<Lattice>& g, int d, int p, const std::vector<double>& b) {
    size_t c=0; iter(g,d,p,[&](int n){ g.fx[n]+=b[c++]; g.fy[n]+=b[c++]; g.fz[n]+=b[c++]; });
  }
  // Self-periodic: accumulate forces from ghost face onto interior face
  void accum_face_force(Grid<Lattice>& g, int d, int src, int dst) {
    iter2(g,d,src,dst,[&](int sn,int dn){ g.fx[dn]+=g.fx[sn]; g.fy[dn]+=g.fy[sn]; g.fz[dn]+=g.fz[sn]; });
  }

  // Face iteration (full ghost-inclusive extent of other dims for corner consistency)
  template<typename F> void iter(const Grid<Lattice>& g, int d, int p, F fn) const {
    assert(d>=0&&d<3&&p>=0);
    assert((d==0&&p<g.gx)||(d==1&&p<g.gy)||(d==2&&p<g.gz));
    if(d==0) { for(int k=0;k<g.gz;k++) for(int j=0;j<g.gy;j++) fn(g.idx(p,j,k)); }
    else if(d==1) { for(int k=0;k<g.gz;k++) for(int i=0;i<g.gx;i++) fn(g.idx(i,p,k)); }
    else { for(int j=0;j<g.gy;j++) for(int i=0;i<g.gx;i++) fn(g.idx(i,j,p)); }
  }
  template<typename F> void iter2(const Grid<Lattice>& g, int d, int s, int t, F fn) const {
    assert(d>=0&&d<3);
    if(d==0) { assert(s>=0&&s<g.gx&&t>=0&&t<g.gx); for(int k=0;k<g.gz;k++) for(int j=0;j<g.gy;j++) fn(g.idx(s,j,k),g.idx(t,j,k)); }
    else if(d==1) { assert(s>=0&&s<g.gy&&t>=0&&t<g.gy); for(int k=0;k<g.gz;k++) for(int i=0;i<g.gx;i++) fn(g.idx(i,s,k),g.idx(i,t,k)); }
    else { assert(s>=0&&s<g.gz&&t>=0&&t<g.gz); for(int j=0;j<g.gy;j++) for(int i=0;i<g.gx;i++) fn(g.idx(i,j,s),g.idx(i,j,t)); }
  }
};

} // namespace CoupLB
} // namespace LAMMPS_NS

#endif // COUPLB_STREAMING_H
