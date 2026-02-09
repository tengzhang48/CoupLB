#include "fix_couplb.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "error.h"
#include "modify.h"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ------------------------------------------------------------------
   fix ID group couplb Nx Ny Nz viscosity rho0 keyword value ...

   Keywords:
     nsub N              LBM sub-steps per MD step (default 1)
     gravity gx gy gz    body force acceleration (lattice units)
     wall_y lo hi        y-boundary: 0=periodic 1=wall 2=moving
     wall_z lo hi        z-boundary (3D only)
     wall_vel vx vy vz   velocity for type-2 walls (lattice units)
     dx value            physical grid spacing (default Lx/Nx)
     output N file       write velocity profile every N steps
     check_every N       stability check frequency (default 0 = at output)
     kernel {roma|peskin4}  IBM delta function (default roma)
     vtk N prefix        write VTK field every N steps
     checkpoint N prefix write checkpoint every N steps
     restart prefix      load checkpoint on init
------------------------------------------------------------------ */

FixCoupLB::FixCoupLB(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),
  ycomm(MPI_COMM_NULL), ycomm_valid(false)
{
  if (narg < 8)
    error->all(FLERR, "Illegal fix couplb command: need Nx Ny Nz nu rho0");

  Nx = std::stoi(arg[3]); Ny = std::stoi(arg[4]); Nz = std::stoi(arg[5]);
  nu = std::stod(arg[6]); rho0 = std::stod(arg[7]);

  nsub = 1;
  gx_ext = gy_ext = gz_ext = 0.0;
  wall_ylo = wall_yhi = wall_zlo = wall_zhi = 0;
  wall_vel[0] = wall_vel[1] = wall_vel[2] = 0.0;
  dx_phys = 0.0;
  output_every = 0; output_file = "couplb_profile.dat";
  check_every = 0; lbm_step_count = 0;
  vtk_every = 0; vtk_prefix = "couplb_vtk"; vtk_pvd_file = "";
  checkpoint_every = 0; checkpoint_prefix = "couplb_ckpt";
  do_restart = false;
  ibm_kernel = CoupLB::DeltaKernel::ROMA;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nsub")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb nsub: need 1 value");
      nsub = std::stoi(arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg],"gravity")==0) {
      if (iarg+4>narg) error->all(FLERR,"fix couplb gravity: need 3 values");
      gx_ext=std::stod(arg[iarg+1]); gy_ext=std::stod(arg[iarg+2]); gz_ext=std::stod(arg[iarg+3]); iarg+=4;
    } else if (strcmp(arg[iarg],"wall_y")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb wall_y: need 2 values");
      wall_ylo=std::stoi(arg[iarg+1]); wall_yhi=std::stoi(arg[iarg+2]); iarg+=3;
    } else if (strcmp(arg[iarg],"wall_z")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb wall_z: need 2 values");
      wall_zlo=std::stoi(arg[iarg+1]); wall_zhi=std::stoi(arg[iarg+2]); iarg+=3;
    } else if (strcmp(arg[iarg],"wall_vel")==0) {
      if (iarg+4>narg) error->all(FLERR,"fix couplb wall_vel: need 3 values");
      wall_vel[0]=std::stod(arg[iarg+1]); wall_vel[1]=std::stod(arg[iarg+2]); wall_vel[2]=std::stod(arg[iarg+3]); iarg+=4;
    } else if (strcmp(arg[iarg],"dx")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb dx: need 1 value");
      dx_phys=std::stod(arg[iarg+1]); iarg+=2;
    } else if (strcmp(arg[iarg],"output")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb output: need 2 values");
      output_every=std::stoi(arg[iarg+1]); output_file=arg[iarg+2]; iarg+=3;
    } else if (strcmp(arg[iarg],"check_every")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb check_every: need 1 value");
      check_every=std::stoi(arg[iarg+1]); iarg+=2;
    } else if (strcmp(arg[iarg],"kernel")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb kernel: need 1 value");
      if (strcmp(arg[iarg+1],"roma")==0) ibm_kernel = CoupLB::DeltaKernel::ROMA;
      else if (strcmp(arg[iarg+1],"peskin4")==0) ibm_kernel = CoupLB::DeltaKernel::PESKIN4;
      else error->all(FLERR,"fix couplb kernel: use roma or peskin4");
      iarg+=2;
    } else if (strcmp(arg[iarg],"vtk")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb vtk: need N prefix");
      vtk_every=std::stoi(arg[iarg+1]); vtk_prefix=arg[iarg+2];
      vtk_pvd_file = vtk_prefix + ".pvd";
      iarg+=3;
    } else if (strcmp(arg[iarg],"checkpoint")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb checkpoint: need N prefix");
      checkpoint_every=std::stoi(arg[iarg+1]); checkpoint_prefix=arg[iarg+2]; iarg+=3;
    } else if (strcmp(arg[iarg],"restart")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb restart: need prefix");
      restart_prefix=arg[iarg+1]; do_restart=true; iarg+=2;
    } else {
      error->all(FLERR,"Unknown fix couplb keyword");
    }
  }

  is3d = (domain->dimension == 3);
  if (!is3d && Nz != 1) error->all(FLERR,"fix couplb: Nz must be 1 for 2D");
}

FixCoupLB::~FixCoupLB() {
  if (ycomm_valid && ycomm != MPI_COMM_NULL) MPI_Comm_free(&ycomm);
}

int FixCoupLB::setmask() {
  int mask = 0;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

void FixCoupLB::init()
{
  for (int d=0;d<3;d++) { domain_lo[d]=domain->boxlo[d]; domain_hi[d]=domain->boxhi[d]; }
  if (dx_phys<=0) dx_phys = (domain_hi[0]-domain_lo[0]) / Nx;
  dx = 1.0;
  tau = nu / (1.0/3.0) + 0.5;

  if (tau <= 0.5) error->all(FLERR,"fix couplb: tau <= 0.5 (unstable)");
  if (tau < 0.505) error->warning(FLERR,"fix couplb: tau very close to 0.5");

  dt_lbm = update->dt / nsub;
  vel_scale = dx_phys / dt_lbm;

  const int dim = is3d ? 3 : 2;
  double dp = 1.0;
  for (int d=0;d<dim+1;d++) dp *= dx_phys;
  force_scale = rho0 * dp / (dt_lbm * dt_lbm);

  if (force_scale <= 0.0 || !std::isfinite(force_scale))
    error->all(FLERR, "fix couplb: force_scale is non-positive or non-finite "
               "(check rho0, dx_phys, dt_lbm)");

  // Wall velocity Ma check
  const double cs = std::sqrt(1.0/3.0);
  const double uw = std::sqrt(wall_vel[0]*wall_vel[0]+wall_vel[1]*wall_vel[1]+wall_vel[2]*wall_vel[2]);
  if (uw > CoupLB::Constants::MA_ERROR*cs)
    error->all(FLERR,"fix couplb: wall Ma > 0.5");
  if (uw > CoupLB::Constants::MA_WARN*cs && comm->me==0 && screen)
    fprintf(screen,"CoupLB WARNING: wall Ma=%.3f > 0.3\n", uw/cs);

  // Gravity-driven Ma estimate
  const double gm = std::sqrt(gx_ext*gx_ext+gy_ext*gy_ext+gz_ext*gz_ext);
  if (gm > CoupLB::Constants::ZERO_TOL) {
    const int H = (wall_ylo>0&&wall_yhi>0) ? Ny : (wall_zlo>0&&wall_zhi>0) ? Nz : 0;
    if (H > 0) {
      const double ue = gm*H*H/(8.0*nu);
      if (ue/cs > CoupLB::Constants::MA_ERROR)
        error->all(FLERR,"fix couplb: estimated Poiseuille Ma > 0.5");
      if (ue/cs > CoupLB::Constants::MA_WARN && comm->me==0 && screen)
        fprintf(screen,"CoupLB WARNING: estimated Poiseuille Ma=%.3f\n", ue/cs);
    }
  }

  if (output_every>0) next_output = update->ntimestep;
  next_check = (check_every>0) ? update->ntimestep+check_every : -1;
  next_vtk = (vtk_every>0) ? update->ntimestep : -1;
  next_checkpoint = (checkpoint_every>0) ? update->ntimestep+checkpoint_every : -1;
  lbm_step_count = 0;

  setup_grid();
  setup_boundaries();

  // Precompute wall face flags for fast exchange checks
  if (is3d) stream3d->precompute_wall_flags(*grid3d);
  else      stream2d->precompute_wall_flags(*grid2d);

  // Load checkpoint if restart requested (after grid + boundaries are set)
  if (do_restart) {
    long ckpt_step = 0;
    bool ok;
    if (is3d) ok = CoupLB::IO<CoupLB::D3Q19>::read_checkpoint(*grid3d, world, restart_prefix, ckpt_step);
    else      ok = CoupLB::IO<CoupLB::D2Q9>::read_checkpoint(*grid2d, world, restart_prefix, ckpt_step);
    if (!ok) error->all(FLERR, "fix couplb: failed to read checkpoint");
    if (comm->me==0 && screen)
      fprintf(screen, "CoupLB: restarted from checkpoint step %ld\n", ckpt_step);
    do_restart = false;  // Only load once
  }

  setup_ycomm();

  if (nsub > 1) {
    int ng=0; for(int i=0;i<atom->nlocal;i++) if(atom->mask[i]&groupbit) ng++;
    int nga=0; MPI_Allreduce(&ng,&nga,1,MPI_INT,MPI_SUM,world);
    if (nga>0) error->all(FLERR,"fix couplb: nsub>1 with IBM particles not supported");
  }

  if (comm->me==0 && screen) {
    fprintf(screen,"CoupLB: %dD grid %dx%dx%d tau=%.6f nu=%.6f rho0=%.4f\n", dim,Nx,Ny,Nz,tau,nu,rho0);
    fprintf(screen,"CoupLB: dx_phys=%.4e dt_lbm=%.4e force_scale=%.4e vel_scale=%.4e\n", dx_phys,dt_lbm,force_scale,vel_scale);
    fprintf(screen,"CoupLB: kernel=%s\n", ibm_kernel==CoupLB::DeltaKernel::ROMA?"roma":"peskin4");
    if (gm>CoupLB::Constants::ZERO_TOL) fprintf(screen,"CoupLB: gravity=(%.2e,%.2e,%.2e)\n",gx_ext,gy_ext,gz_ext);
    if (wall_ylo||wall_yhi) fprintf(screen,"CoupLB: y-walls lo=%d hi=%d\n",wall_ylo,wall_yhi);
    if (wall_zlo||wall_zhi) fprintf(screen,"CoupLB: z-walls lo=%d hi=%d\n",wall_zlo,wall_zhi);
    if (output_every>0) fprintf(screen,"CoupLB: output every %d -> %s\n",output_every,output_file.c_str());
    if (check_every>0) fprintf(screen,"CoupLB: stability check every %d steps\n",check_every);
    if (nsub>1) fprintf(screen,"CoupLB: nsub=%d (pure-fluid)\n",nsub);
    if (vtk_every>0) fprintf(screen,"CoupLB: VTK output every %d -> %s_*.vti (series: %s)\n",vtk_every,vtk_prefix.c_str(),vtk_pvd_file.c_str());
    if (checkpoint_every>0) fprintf(screen,"CoupLB: checkpoint every %d -> %s.*.clbk\n",checkpoint_every,checkpoint_prefix.c_str());
    fprintf(screen,"CoupLB: periodicity=(%d,%d,%d) procgrid=(%d,%d,%d)\n",
      domain->periodicity[0],domain->periodicity[1],domain->periodicity[2],
      comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);
  }
}

void FixCoupLB::setup_grid()
{
  const double *sl=domain->sublo, *sh=domain->subhi;
  constexpr double eps=1e-12;
  int x0=int(std::floor((sl[0]-domain_lo[0])/dx_phys+eps));
  int y0=int(std::floor((sl[1]-domain_lo[1])/dx_phys+eps));
  int z0=int(std::floor((sl[2]-domain_lo[2])/dx_phys+eps));
  int x1=int(std::floor((sh[0]-domain_lo[0])/dx_phys+eps));
  int y1=int(std::floor((sh[1]-domain_lo[1])/dx_phys+eps));
  int z1=int(std::floor((sh[2]-domain_lo[2])/dx_phys+eps));
  int nlx=x1-x0, nly=y1-y0, nlz=is3d?(z1-z0):1;
  if (nlx<=0||nly<=0||(is3d&&nlz<=0)) error->one(FLERR,"fix couplb: local grid empty");
  if (screen) fprintf(screen,"CoupLB rank %d: %dx%dx%d offset(%d,%d,%d)\n",comm->me,nlx,nly,nlz,x0,y0,z0);

  int pn[3][2]; for(int d=0;d<3;d++) { pn[d][0]=comm->procneigh[d][0]; pn[d][1]=comm->procneigh[d][1]; }
  bool px=domain->periodicity[0], py=domain->periodicity[1], pz=domain->periodicity[2];

  if (is3d) {
    grid3d=std::make_unique<CoupLB::Grid<CoupLB::D3Q19>>();
    grid3d->allocate(nlx,nly,nlz,dx,x0,y0,z0,Nx,Ny,Nz);
    grid3d->init_equilibrium(rho0,0,0,0);
    bgk3d=std::make_unique<CoupLB::BGK<CoupLB::D3Q19>>(); bgk3d->set_tau(tau);
    stream3d=std::make_unique<CoupLB::Streaming<CoupLB::D3Q19>>();
    stream3d->set_comm(world); stream3d->set_neighbors(pn);
    stream3d->set_periodic(px,py,pz); stream3d->set_nprocs(comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);
    stream3d->allocate_buffers(*grid3d,rho0);
  } else {
    grid2d=std::make_unique<CoupLB::Grid<CoupLB::D2Q9>>();
    grid2d->allocate(nlx,nly,1,dx,x0,y0,0,Nx,Ny,1);
    grid2d->init_equilibrium(rho0,0,0,0);
    bgk2d=std::make_unique<CoupLB::BGK<CoupLB::D2Q9>>(); bgk2d->set_tau(tau);
    stream2d=std::make_unique<CoupLB::Streaming<CoupLB::D2Q9>>();
    stream2d->set_comm(world); stream2d->set_neighbors(pn);
    stream2d->set_periodic(px,py,false); stream2d->set_nprocs(comm->procgrid[0],comm->procgrid[1],1);
    stream2d->allocate_buffers(*grid2d,rho0);
  }
}

void FixCoupLB::setup_boundaries()
{
  bool aylo=comm->myloc[1]==0, ayhi=comm->myloc[1]==comm->procgrid[1]-1;
  bool azlo=comm->myloc[2]==0, azhi=comm->myloc[2]==comm->procgrid[2]-1;
  if (is3d) {
    auto& g=*grid3d;
    if (wall_ylo||wall_yhi) CoupLB::Boundary<CoupLB::D3Q19>::set_walls_y(g,aylo&&wall_ylo>0,ayhi&&wall_yhi>0,wall_ylo,wall_yhi);
    if (wall_zlo||wall_zhi) CoupLB::Boundary<CoupLB::D3Q19>::set_walls_z(g,azlo&&wall_zlo>0,azhi&&wall_zhi>0,wall_zlo,wall_zhi);
    CoupLB::Boundary<CoupLB::D3Q19>::set_wall_velocity(g,2,wall_vel[0],wall_vel[1],wall_vel[2]);
  } else {
    auto& g=*grid2d;
    if (wall_ylo||wall_yhi) CoupLB::Boundary<CoupLB::D2Q9>::set_walls_y(g,aylo&&wall_ylo>0,ayhi&&wall_yhi>0,wall_ylo,wall_yhi);
    CoupLB::Boundary<CoupLB::D2Q9>::set_wall_velocity(g,2,wall_vel[0],wall_vel[1],0);
  }
}

void FixCoupLB::setup_ycomm() {
  if (ycomm_valid && ycomm!=MPI_COMM_NULL) MPI_Comm_free(&ycomm);
  ycomm_valid = false;
  if (comm->procgrid[1]<=1) return;
  int color = comm->myloc[0] + comm->procgrid[0]*comm->myloc[2];
  MPI_Comm_split(world, color, comm->myloc[1], &ycomm);
  ycomm_valid = true;
}

void FixCoupLB::setup(int vflag) { post_force(vflag); }

void FixCoupLB::post_force(int) {
  for (int s=0;s<nsub;s++) { lbm_step(); lbm_step_count++; }
  ibm_coupling();
}

void FixCoupLB::lbm_step() {
  if (is3d) do_lbm_step(*grid3d,*bgk3d,*stream3d);
  else      do_lbm_step(*grid2d,*bgk2d,*stream2d);
}

template<typename L>
void FixCoupLB::do_lbm_step(CoupLB::Grid<L>& g, CoupLB::BGK<L>& b, CoupLB::Streaming<L>& s) {
  apply_external_force(g);
  g.compute_macroscopic(true);
  b.collide(g);
  s.exchange(g);
  s.stream(g);
  enforce_wall_ghost_fields(g);
  // IMPORTANT: clear_forces() MUST stay at the END of lbm_step, not the start.
  // IBM reaction forces are spread during do_ibm_coupling() AFTER this function
  // returns. Those forces must survive until the NEXT call to do_lbm_step(),
  // where they enter via Guo forcing in collide(). Moving clear to the start
  // would destroy them before collision uses them.
  g.clear_forces();
}

template<typename L>
void FixCoupLB::apply_external_force(CoupLB::Grid<L>& g) {
  if (std::fabs(gx_ext)<CoupLB::Constants::ZERO_TOL &&
      std::fabs(gy_ext)<CoupLB::Constants::ZERO_TOL &&
      std::fabs(gz_ext)<CoupLB::Constants::ZERO_TOL) return;
  const int klo=(L::D==3)?1:0, khi=(L::D==3)?(g.gz-2):0;
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int k=klo;k<=khi;k++)
    for (int j=1;j<=g.gy-2;j++)
      for (int i=1;i<=g.gx-2;i++) {
        const int n=g.idx(i,j,k);
        if (g.type[n]!=0) continue;
        g.fx[n]+=gx_ext*g.rho[n]; g.fy[n]+=gy_ext*g.rho[n]; g.fz[n]+=gz_ext*g.rho[n];
      }
}

void FixCoupLB::ibm_coupling() {
  if (is3d) do_ibm_coupling(*grid3d,*stream3d);
  else      do_ibm_coupling(*grid2d,*stream2d);
}

template<typename L>
void FixCoupLB::do_ibm_coupling(CoupLB::Grid<L>& g, CoupLB::Streaming<L>& s) {
  double **x=atom->x, **v=atom->v, **f=atom->f;
  const int nl=atom->nlocal; const int *mk=atom->mask;

  enforce_wall_ghost_fields(g);

  // Global check: only skip IBM entirely if NO rank has particles.
  // MPI exchanges are collective — every rank must participate.
  int ng=0; for(int i=0;i<nl;i++) if(mk[i]&groupbit) ng++;
  int ng_global=0;
  MPI_Allreduce(&ng,&ng_global,1,MPI_INT,MPI_SUM,world);
  if (!ng_global) return;

  g.compute_macroscopic(false);
  s.exchange_velocity(g);

  // Local IBM work — only ranks with particles do interpolation/spreading
  for (int i=0;i<nl;i++) {
    if (!(mk[i]&groupbit)) continue;
    auto uf = CoupLB::IBM<L>::interpolate(g, x[i][0], x[i][1], x[i][2], domain_lo, dx_phys, ibm_kernel);
    const double ux=uf[0]*vel_scale, uy=uf[1]*vel_scale, uz=uf[2]*vel_scale;
    const double dt=update->dt;
    const double m = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    const double fx=m*(ux-v[i][0])/dt, fy=m*(uy-v[i][1])/dt, fz=m*(uz-v[i][2])/dt;
    f[i][0]+=fx; f[i][1]+=fy; f[i][2]+=fz;
    constexpr double dv=1.0;
    CoupLB::IBM<L>::spread(g, x[i][0],x[i][1],x[i][2], -fx/force_scale,-fy/force_scale,-fz/force_scale, dv, domain_lo, dx_phys, ibm_kernel);
  }

  // Reverse force exchange: accumulate forces from ghost cells back
  // to owning proc's interior cells. IBM stencils near subdomain
  // boundaries may have spread force to ghost cells; without this
  // step those forces would be silently lost.
  s.exchange_forces(g);
  // NOTE: Do NOT clear_forces() here. The spread IBM reaction forces
  // must survive until the next lbm_step()'s collision, where they
  // enter via Guo forcing. clear_forces() at the end of do_lbm_step
  // handles the reset after they've been consumed.
}

template<typename L>
void FixCoupLB::enforce_wall_ghost_fields(CoupLB::Grid<L>& g) {
  for (int n=0;n<g.ntotal;n++) {
    if (g.type[n]==1) { g.rho[n]=rho0; g.ux[n]=g.uy[n]=g.uz[n]=0; }
    else if (g.type[n]==2) { g.rho[n]=rho0; g.ux[n]=g.bc_ux[n]; g.uy[n]=g.bc_uy[n]; g.uz[n]=g.bc_uz[n]; }
  }
}

template<typename L>
void FixCoupLB::check_stability(CoupLB::Grid<L>& g) {
  g.compute_macroscopic(false);
  check_stability_precomputed(g);
}

template<typename L>
void FixCoupLB::check_stability_precomputed(CoupLB::Grid<L>& g) {
  double lu=g.max_velocity(), gu=0;
  MPI_Allreduce(&lu,&gu,1,MPI_DOUBLE,MPI_MAX,world);
  const double cs=std::sqrt(L::cs2), ma=gu/cs;
  if (ma>CoupLB::Constants::MA_ERROR) error->all(FLERR,"fix couplb: Ma>0.5, simulation invalid");
  if (ma>CoupLB::Constants::MA_WARN && comm->me==0 && screen)
    fprintf(screen,"CoupLB WARNING: Ma=%.4f at step %ld\n",ma,(long)update->ntimestep);

  double lm,lpx,lpy,lpz; g.compute_diagnostics(lm,lpx,lpy,lpz);
  double gm,gpx,gpy,gpz;
  MPI_Reduce(&lm,&gm,1,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(&lpx,&gpx,1,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(&lpy,&gpy,1,MPI_DOUBLE,MPI_SUM,0,world);
  MPI_Reduce(&lpz,&gpz,1,MPI_DOUBLE,MPI_SUM,0,world);
  if (comm->me==0 && screen)
    fprintf(screen,"CoupLB step %ld: Ma=%.4f mass=%.8e mom=(%.4e,%.4e,%.4e)\n",
      (long)update->ntimestep,ma,gm,gpx,gpy,gpz);

  int local_clamps = g.rho_clamp_count;
  int global_clamps = 0;
  MPI_Reduce(&local_clamps, &global_clamps, 1, MPI_INT, MPI_SUM, 0, world);
  if (global_clamps > 0 && comm->me==0 && screen)
    fprintf(screen,"CoupLB WARNING: %d nodes had density clamped at step %ld\n",
      global_clamps, (long)update->ntimestep);
}

void FixCoupLB::end_of_step() {
  const bigint step = update->ntimestep;
  bool chk=false, out=false, vtk=false, ckpt=false;
  if (check_every>0 && step>=next_check) { chk=true; next_check=step+check_every; }
  if (output_every>0 && step>=next_output) { out=true; chk=true; next_output+=output_every; }
  if (vtk_every>0 && step>=next_vtk) { vtk=true; next_vtk=step+vtk_every; }
  if (checkpoint_every>0 && step>=next_checkpoint) { ckpt=true; next_checkpoint=step+checkpoint_every; }
  if (!chk && !out && !vtk && !ckpt) return;

  // Compute macroscopic fields once if any output needs them.
  // check_stability, write_profile, and write_vtk all read rho/ux/uy/uz.
  // checkpoint only needs f[] so it doesn't require this.
  const bool need_macro = chk || out || vtk;

  if (is3d) {
    if (need_macro) grid3d->compute_macroscopic(false);
    if (chk) check_stability_precomputed(*grid3d);
    if (out) write_profile(*grid3d, step);
    if (vtk) {
      CoupLB::IO<CoupLB::D3Q19>::write_vtk(*grid3d, world, (long)step, vtk_prefix, domain_lo, dx_phys, vel_scale, force_scale);
      vtk_steps.push_back((long)step);
      if (comm->me==0) CoupLB::IO<CoupLB::D3Q19>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_lbm);
    }
    if (ckpt) CoupLB::IO<CoupLB::D3Q19>::write_checkpoint(*grid3d, world, (long)step, checkpoint_prefix);
  } else {
    if (need_macro) grid2d->compute_macroscopic(false);
    if (chk) check_stability_precomputed(*grid2d);
    if (out) write_profile(*grid2d, step);
    if (vtk) {
      CoupLB::IO<CoupLB::D2Q9>::write_vtk(*grid2d, world, (long)step, vtk_prefix, domain_lo, dx_phys, vel_scale, force_scale);
      vtk_steps.push_back((long)step);
      if (comm->me==0) CoupLB::IO<CoupLB::D2Q9>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_lbm);
    }
    if (ckpt) CoupLB::IO<CoupLB::D2Q9>::write_checkpoint(*grid2d, world, (long)step, checkpoint_prefix);
  }
}

template<typename L>
void FixCoupLB::write_profile(CoupLB::Grid<L>& g, bigint step) {
  const int nyl=g.ny, nx=g.nx, nz=(L::D==3)?g.nz:1, cxz=nx*nz;
  std::vector<double> ul(nyl,0), vl(nyl,0), rl(nyl,0);
  for(int k=0;k<nz;k++) for(int j=0;j<nyl;j++) for(int i=0;i<nx;i++) {
    const int n=g.lidx(i,j,k); ul[j]+=g.ux[n]; vl[j]+=g.uy[n]; rl[j]+=g.rho[n];
  }
  for(int j=0;j<nyl;j++) { ul[j]/=cxz; vl[j]/=cxz; rl[j]/=cxz; }

  int me; MPI_Comm_rank(world,&me);
  const char* kn = ibm_kernel==CoupLB::DeltaKernel::ROMA ? "roma" : "peskin4";

  auto write_data = [&](FILE* fp, int ny_tot, const double* ux_d, const double* uy_d, const double* rho_d, int yo) {
    if (step <= (bigint)output_every) {
      fprintf(fp,"# CoupLB velocity profile\n# Columns: step j y_lat rho ux uy\n");
      fprintf(fp,"# H=Ny=%d tau=%.6f nu=%.6f rho0=%.4f kernel=%s\n#\n",Ny,tau,nu,rho0,kn);
    }
    fprintf(fp,"# step = %ld\n",(long)step);
    for(int j=0;j<ny_tot;j++)
      fprintf(fp,"%ld %d %.1f %.8f %.10e %.10e\n",(long)step,j,(double)(yo+j),rho_d[j],ux_d[j],uy_d[j]);
    fprintf(fp,"\n");
  };

  if (!ycomm_valid) {
    if (me==0) {
      FILE* fp=fopen(output_file.c_str(), step<=(bigint)output_every?"w":"a");
      if (!fp) { if(screen) fprintf(screen,"CoupLB WARNING: cannot open %s\n",output_file.c_str()); return; }
      write_data(fp,nyl,ul.data(),vl.data(),rl.data(),g.offset[1]); fclose(fp);
    }
  } else {
    int yr,ys; MPI_Comm_rank(ycomm,&yr); MPI_Comm_size(ycomm,&ys);
    std::vector<int> cnt(ys), dsp(ys);
    MPI_Gather(&nyl,1,MPI_INT,cnt.data(),1,MPI_INT,0,ycomm);
    int nyt=0; if(yr==0) for(int i=0;i<ys;i++){dsp[i]=nyt;nyt+=cnt[i];}
    std::vector<double> ua,va,ra;
    if(yr==0){ua.resize(nyt);va.resize(nyt);ra.resize(nyt);}
    MPI_Gatherv(ul.data(),nyl,MPI_DOUBLE,ua.data(),cnt.data(),dsp.data(),MPI_DOUBLE,0,ycomm);
    MPI_Gatherv(vl.data(),nyl,MPI_DOUBLE,va.data(),cnt.data(),dsp.data(),MPI_DOUBLE,0,ycomm);
    MPI_Gatherv(rl.data(),nyl,MPI_DOUBLE,ra.data(),cnt.data(),dsp.data(),MPI_DOUBLE,0,ycomm);
    if(me==0&&yr==0) {
      FILE* fp=fopen(output_file.c_str(), step<=(bigint)output_every?"w":"a");
      if(!fp){if(screen)fprintf(screen,"CoupLB WARNING: cannot open %s\n",output_file.c_str());}
      else{write_data(fp,nyt,ua.data(),va.data(),ra.data(),0); fclose(fp);}
    }
  }
}