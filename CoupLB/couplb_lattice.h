#ifndef COUPLB_LATTICE_H
#define COUPLB_LATTICE_H
#include <vector>
#include <array>
#include <cstring>
#include <cmath>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace LAMMPS_NS {
namespace CoupLB {
namespace Constants {
constexpr double RHO_MIN        = 1e-10;
constexpr double DELTA_TOL      = 1e-12;
constexpr double ZERO_TOL       = 1e-15;
constexpr double RHO_CLAMP_FRAC = 0.1;
constexpr double MA_WARN        = 0.3;
constexpr double MA_ERROR       = 0.5;
}
// ============================================================
// Lattice descriptors
// ============================================================
struct D2Q9 {
static constexpr int D = 2;
static constexpr int Q = 9;
static constexpr double cs2 = 1.0 / 3.0;
static constexpr int e[9][3] = {
{ 0, 0, 0},
{ 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0},
{ 1, 1, 0}, {-1,-1, 0}, { 1,-1, 0}, {-1, 1, 0}
};
static constexpr double w[9] = {
4.0/9.0,
1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
static constexpr int opp[9] = { 0, 2,1, 4,3, 6,5, 8,7 };
};
struct D3Q19 {
static constexpr int D = 3;
static constexpr int Q = 19;
static constexpr double cs2 = 1.0 / 3.0;
static constexpr int e[19][3] = {
{ 0, 0, 0},
{ 1, 0, 0}, {-1, 0, 0},
{ 0, 1, 0}, { 0,-1, 0},
{ 0, 0, 1}, { 0, 0,-1},
{ 1, 1, 0}, {-1,-1, 0}, { 1,-1, 0}, {-1, 1, 0},
{ 1, 0, 1}, {-1, 0,-1}, { 1, 0,-1}, {-1, 0, 1},
{ 0, 1, 1}, { 0,-1,-1}, { 0, 1,-1}, { 0,-1, 1}
};
static constexpr double w[19] = {
1.0/3.0,
1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
1.0/18.0, 1.0/18.0,
1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
static constexpr int opp[19] = {
0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17
};
};
// ============================================================
// Grid — local processor subdomain with ghost layers
//
// SoA layout: f[q * ntotal + node_index]
// Ghost-inclusive: gx=nx+2, gy=ny+2, gz=nz+2 (3D) or 1 (2D)
// ============================================================
template <typename Lattice>
class Grid {
public:
static constexpr int D = Lattice::D;
static constexpr int Q = Lattice::Q;
int nx, ny, nz;
int gx, gy, gz;
int ntotal;
double dx;
int offset[3];
int Nx, Ny, Nz;
std::vector<double> f, f_buf;
std::vector<double> rho, ux, uy, uz;
std::vector<double> fx, fy, fz;
std::vector<int> type;
std::vector<double> bc_ux, bc_uy, bc_uz;
// Diagnostic: number of density clamps in last compute_macroscopic call
int rho_clamp_count;
Grid() : nx(0), ny(0), nz(0), dx(1.0), ntotal(0),
gx(0), gy(0), gz(0), Nx(0), Ny(0), Nz(0), rho_clamp_count(0) {
offset[0] = offset[1] = offset[2] = 0;
}
void allocate(int nx_, int ny_, int nz_, double dx_,
int ox, int oy, int oz,
int Nx_, int Ny_, int Nz_) {
nx = nx_; ny = ny_; nz = nz_; dx = dx_;
offset[0] = ox; offset[1] = oy; offset[2] = oz;
Nx = Nx_; Ny = Ny_; Nz = Nz_;
gx = nx + 2;
gy = ny + 2;
gz = (D == 3) ? (nz + 2) : 1;
ntotal = gx * gy * gz;
f.assign(Q * ntotal, 0.0);
f_buf.assign(Q * ntotal, 0.0);
rho.assign(ntotal, 1.0);
ux.assign(ntotal, 0.0);  uy.assign(ntotal, 0.0);  uz.assign(ntotal, 0.0);
fx.assign(ntotal, 0.0);  fy.assign(ntotal, 0.0);  fz.assign(ntotal, 0.0);
type.assign(ntotal, 0);
bc_ux.assign(ntotal, 0.0);  bc_uy.assign(ntotal, 0.0);  bc_uz.assign(ntotal, 0.0);
}
inline int idx(int i, int j, int k = 0) const {
assert(i >= 0 && i < gx);
assert(j >= 0 && j < gy);
assert(k >= 0 && k < gz);
return i + gx * (j + gy * k);
}
inline int lidx(int li, int lj, int lk = 0) const {
return idx(li + 1, lj + 1, (D == 3) ? (lk + 1) : 0);
}
inline double& fi(int q, int n)       { return f[q * ntotal + n]; }
inline double  fi(int q, int n) const { return f[q * ntotal + n]; }
inline double& fi_buf(int q, int n)       { return f_buf[q * ntotal + n]; }
inline double  fi_buf(int q, int n) const { return f_buf[q * ntotal + n]; }
inline double feq(int q, double r, double u0, double u1, double u2) const {
const double eu = Lattice::e[q][0]*u0 + Lattice::e[q][1]*u1
+ Lattice::e[q][2]*u2;
const double uu = u0*u0 + u1*u1 + u2*u2;
constexpr double cs2 = Lattice::cs2;
constexpr double cs4 = cs2 * cs2;
return Lattice::w[q] * r * (1.0 + eu/cs2 + 0.5*eu*eu/cs4 - 0.5*uu/cs2);
}
void init_equilibrium(double rho0, double ux0, double uy0, double uz0) {
for (int n = 0; n < ntotal; n++) {
rho[n] = rho0;
ux[n] = ux0; uy[n] = uy0; uz[n] = uz0;
fx[n] = fy[n] = fz[n] = 0.0;
for (int q = 0; q < Q; q++)
fi(q, n) = feq(q, rho0, ux0, uy0, uz0);
}
f_buf = f;  // Ensure swap buffer matches initial state
}
void compute_macroscopic(bool guo_correction = true) {
const int klo = (D == 3) ? 1 : 0;
const int khi = (D == 3) ? (gz - 2) : 0;
int local_clamp_count = 0;  // Thread-local counter for density clamps
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) reduction(+:local_clamp_count)
#endif
for (int k = klo; k <= khi; k++) {
for (int j = 1; j <= gy - 2; j++) {
for (int i = 1; i <= gx - 2; i++) {
const int n = idx(i, j, k);
if (type[n] != 0) continue;
double r = 0.0, su = 0.0, sv = 0.0, sw = 0.0;
for (int q = 0; q < Q; q++) {
const double fval = fi(q, n);
r  += fval;
su += Lattice::e[q][0] * fval;
sv += Lattice::e[q][1] * fval;
sw += Lattice::e[q][2] * fval;
}
// Clamp rho[n] itself, not just r_safe. Collision, streaming,
// and external force all read rho[n] directly. Storing a
// negative density would poison downstream computations.
if (r <= Constants::RHO_MIN) {
    r = Constants::RHO_MIN;
    local_clamp_count++;
}
// Safety net for corrupted values (should never trigger if physics is correct)
if (!std::isfinite(r)) {
    r = Constants::RHO_MIN;
    local_clamp_count++;  // Count as clamp event
}
rho[n] = r;
const double inv_r = 1.0 / r;  // Now guaranteed safe
if (guo_correction) {
ux[n] = (su + 0.5 * fx[n]) * inv_r;
uy[n] = (sv + 0.5 * fy[n]) * inv_r;
uz[n] = (sw + 0.5 * fz[n]) * inv_r;
} else {
ux[n] = su * inv_r;
uy[n] = sv * inv_r;
uz[n] = sw * inv_r;
}
}
}
}
rho_clamp_count = local_clamp_count;  // STORE FINAL COUNT TO MEMBER
}
double max_velocity() const {
double umax2 = 0.0;
const int klo = (D == 3) ? 1 : 0;
const int khi = (D == 3) ? (gz - 2) : 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) reduction(max:umax2)
#endif
for (int k = klo; k <= khi; k++) {
for (int j = 1; j <= gy - 2; j++) {
for (int i = 1; i <= gx - 2; i++) {
const int n = idx(i, j, k);
if (type[n] != 0) continue;
const double u2 = ux[n]*ux[n] + uy[n]*uy[n] + uz[n]*uz[n];
if (u2 > umax2) umax2 = u2;
}
}
}
return std::sqrt(umax2);
}
void compute_diagnostics(double& mass, double& momx, double& momy, double& momz) const {
mass = momx = momy = momz = 0.0;
const int klo = (D == 3) ? 1 : 0;
const int khi = (D == 3) ? (gz - 2) : 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) \
reduction(+:mass,momx,momy,momz)
#endif
for (int k = klo; k <= khi; k++) {
for (int j = 1; j <= gy - 2; j++) {
for (int i = 1; i <= gx - 2; i++) {
const int n = idx(i, j, k);
if (type[n] != 0) continue;
mass += rho[n];
momx += rho[n] * ux[n];
momy += rho[n] * uy[n];
momz += rho[n] * uz[n];
}
}
}
}
void clear_forces() {
std::memset(fx.data(), 0, ntotal * sizeof(double));
std::memset(fy.data(), 0, ntotal * sizeof(double));
std::memset(fz.data(), 0, ntotal * sizeof(double));
}
};
} // namespace CoupLB
} // namespace LAMMPS_NS
#endif // COUPLB_LATTICE_H
