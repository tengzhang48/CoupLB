#include "fix_couplb.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "error.h"
#include "modify.h"
#include "input.h"
#include "variable.h"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ------------------------------------------------------------------
   fix ID group couplb Nx Ny Nz nu rho keyword value ...

   All input parameters are in LAMMPS units.
   The fix converts to lattice units internally.

   Required:
     Nx Ny Nz            LBM grid dimensions
     nu                  kinematic viscosity
     rho                 fluid density

   Keywords:
     md_per_lb N         MD steps per LBM step (default 1)
     xi_ibm value        IBM penalty relaxation factor, 0 < xi <= 1
     gravity gx gy gz    body force acceleration;
                         each component can be a constant or v_varname
     wall_x lo hi        x-boundary: 0=periodic 1=no-slip 2=moving 3=free-slip 4=open
     wall_y lo hi        y-boundary: 0=periodic 1=no-slip 2=moving 3=free-slip 4=open
     wall_z lo hi        z-boundary (3D only): 0=periodic 1=no-slip 2=moving 3=free-slip 4=open
     wall_vel vx vy vz   velocity for type-2 walls
     dx value            grid spacing (default Lx/Nx)
     output N file       write velocity profile every N steps
     check_every N       stability check frequency
     kernel {roma|peskin4}  IBM delta function (default roma)
     vtk N prefix        write VTK field every N steps
     checkpoint N prefix write checkpoint every N steps
     restart prefix      load checkpoint on init
------------------------------------------------------------------ */

FixCoupLB::FixCoupLB(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg),
  ycomm(MPI_COMM_NULL), ycomm_valid(false),
  gxstr(nullptr), gystr(nullptr), gzstr(nullptr)
{
  if (narg < 8)
    error->all(FLERR, "Illegal fix couplb command: need Nx Ny Nz nu rho");

  Nx = std::stoi(arg[3]); Ny = std::stoi(arg[4]); Nz = std::stoi(arg[5]);
  nu = std::stod(arg[6]); rho = std::stod(arg[7]);

  md_per_lb = 1;
  md_sub_count = 0;
  xi_ibm = 1.0;
  gx_ext = gy_ext = gz_ext = 0.0;
  gxstyle = gystyle = gzstyle = NONE_G;
  gxvar = gyvar = gzvar = -1;
  wall_xlo = wall_xhi = wall_ylo = wall_yhi = wall_zlo = wall_zhi = 0;
  wall_vel[0] = wall_vel[1] = wall_vel[2] = 0.0;
  dx = 0.0;
  output_every = 0; output_file = "couplb_profile.dat";
  check_every = 0; lbm_step_count = 0;
  vtk_every = 0; vtk_prefix = "couplb_vtk"; vtk_pvd_file = "";
  checkpoint_every = 0; checkpoint_prefix = "couplb_ckpt";
  do_restart = false;
  ibm_kernel = CoupLB::DeltaKernel::ROMA;
  ibm_has_particles = false;

  int iarg = 8;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"md_per_lb")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb md_per_lb: need 1 value");
      md_per_lb = std::stoi(arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg],"xi_ibm")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb xi_ibm: need 1 value");
      xi_ibm = std::stod(arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg],"gravity")==0) {
      if (iarg+4>narg) error->all(FLERR,"fix couplb gravity: need 3 values");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        gxstr = utils::strdup(arg[iarg+1]+2);
        gxstyle = EQUAL_G;
      } else {
        gx_ext = std::stod(arg[iarg+1]);
        gxstyle = CONSTANT_G;
      }
      if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
        gystr = utils::strdup(arg[iarg+2]+2);
        gystyle = EQUAL_G;
      } else {
        gy_ext = std::stod(arg[iarg+2]);
        gystyle = CONSTANT_G;
      }
      if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
        gzstr = utils::strdup(arg[iarg+3]+2);
        gzstyle = EQUAL_G;
      } else {
        gz_ext = std::stod(arg[iarg+3]);
        gzstyle = CONSTANT_G;
      }
      iarg+=4;
    } else if (strcmp(arg[iarg],"wall_x")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb wall_x: need 2 values");
      wall_xlo=std::stoi(arg[iarg+1]); wall_xhi=std::stoi(arg[iarg+2]); iarg+=3;
    } else if (strcmp(arg[iarg],"wall_y")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb wall_y: need 2 values");
      wall_ylo=std::stoi(arg[iarg+1]); wall_yhi=std::stoi(arg[iarg+2]); iarg+=3;
    } else if (strcmp(arg[iarg],"wall_z")==0) {
      if (iarg+3>narg) error->all(FLERR,"fix couplb wall_z: need 2 values");
      wall_zlo=std::stoi(arg[iarg+1]); wall_zhi=std::stoi(arg[iarg+2]); iarg+=3;
    } else if (strcmp(arg[iarg],"wall_vel")==0) {
      if (iarg+4>narg) error->all(FLERR,"fix couplb wall_vel: need 3 values");
      wall_vel[0]=std::stod(arg[iarg+1]);
      wall_vel[1]=std::stod(arg[iarg+2]);
      wall_vel[2]=std::stod(arg[iarg+3]);
      iarg+=4;
    } else if (strcmp(arg[iarg],"dx")==0) {
      if (iarg+2>narg) error->all(FLERR,"fix couplb dx: need 1 value");
      dx=std::stod(arg[iarg+1]); iarg+=2;
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

FixCoupLB::~FixCoupLB()
{
  if (ycomm_valid && ycomm != MPI_COMM_NULL) MPI_Comm_free(&ycomm);
  delete[] gxstr;
  delete[] gystr;
  delete[] gzstr;
}

int FixCoupLB::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

void FixCoupLB::init()
{
  if (md_per_lb < 1)
    error->all(FLERR, "fix couplb: md_per_lb must be >= 1");
  if (xi_ibm <= 0.0 || xi_ibm > 1.0)
    error->all(FLERR, "fix couplb: xi_ibm must be in (0, 1]");

  if (gxstr) {
    gxvar = input->variable->find(gxstr);
    if (gxvar < 0) error->all(FLERR, "Variable {} for gravity x in fix couplb does not exist", gxstr);
    if (!input->variable->equalstyle(gxvar))
      error->all(FLERR, "Variable {} for gravity x in fix couplb must be equal-style", gxstr);
  }
  if (gystr) {
    gyvar = input->variable->find(gystr);
    if (gyvar < 0) error->all(FLERR, "Variable {} for gravity y in fix couplb does not exist", gystr);
    if (!input->variable->equalstyle(gyvar))
      error->all(FLERR, "Variable {} for gravity y in fix couplb must be equal-style", gystr);
  }
  if (gzstr) {
    gzvar = input->variable->find(gzstr);
    if (gzvar < 0) error->all(FLERR, "Variable {} for gravity z in fix couplb does not exist", gzstr);
    if (!input->variable->equalstyle(gzvar))
      error->all(FLERR, "Variable {} for gravity z in fix couplb must be equal-style", gzstr);
  }

  for (int d=0;d<3;d++) { domain_lo[d]=domain->boxlo[d]; domain_hi[d]=domain->boxhi[d]; }
  if (dx<=0) dx = (domain_hi[0]-domain_lo[0]) / Nx;
  inv_dx = 1.0 / dx;

  dt_LBM = update->dt * md_per_lb;
  vel_scale = dx / dt_LBM;

  nu_lb = nu * dt_LBM / (dx * dx);
  tau = nu_lb / (1.0/3.0) + 0.5;

  if (tau <= 0.5) error->all(FLERR,"fix couplb: tau <= 0.5 (unstable)");
  if (tau < 0.505) error->warning(FLERR,"fix couplb: tau very close to 0.5");

  const int dim = is3d ? 3 : 2;
  double dp = 1.0;
  for (int d=0;d<dim+1;d++) dp *= dx;
  force_scale = rho * dp / (dt_LBM * dt_LBM);

  if (force_scale <= 0.0 || !std::isfinite(force_scale))
    error->all(FLERR, "fix couplb: force_scale is non-positive or non-finite "
               "(check rho, dx, dt_LBM)");

  const double inv_vel = 1.0 / vel_scale;
  double wv_lb[3] = { wall_vel[0]*inv_vel, wall_vel[1]*inv_vel, wall_vel[2]*inv_vel };

  const double cs = std::sqrt(1.0/3.0);
  const double uw = std::sqrt(wv_lb[0]*wv_lb[0]+wv_lb[1]*wv_lb[1]+wv_lb[2]*wv_lb[2]);
  if (uw > CoupLB::Constants::MA_ERROR*cs)
    error->all(FLERR,"fix couplb: wall Ma > 0.5");
  if (uw > CoupLB::Constants::MA_WARN*cs && comm->me==0 && screen)
    fprintf(screen,"CoupLB WARNING: wall Ma=%.3f > 0.3\n", uw/cs);

  if (gxstyle == CONSTANT_G && gystyle == CONSTANT_G && gzstyle == CONSTANT_G) {
    const double accel_conv = dt_LBM * dt_LBM / dx;
    const double gx_lb = gx_ext * accel_conv;
    const double gy_lb = gy_ext * accel_conv;
    const double gz_lb = gz_ext * accel_conv;
    const double gm = std::sqrt(gx_lb*gx_lb+gy_lb*gy_lb+gz_lb*gz_lb);
    if (gm > CoupLB::Constants::ZERO_TOL) {
      const int H = (wall_ylo>0&&wall_yhi>0) ? Ny : (wall_zlo>0&&wall_zhi>0) ? Nz : 0;
      if (H > 0) {
        const double ue = gm*H*H/(8.0*nu_lb);
        if (ue/cs > CoupLB::Constants::MA_ERROR)
          error->all(FLERR,"fix couplb: estimated Poiseuille Ma > 0.5");
        if (ue/cs > CoupLB::Constants::MA_WARN && comm->me==0 && screen)
          fprintf(screen,"CoupLB WARNING: estimated Poiseuille Ma=%.3f\n", ue/cs);
      }
    }
  }

  // Step 0 output is written directly in setup()
  if (output_every>0) next_output = update->ntimestep + output_every;
  next_check = (check_every>0) ? update->ntimestep+check_every : -1;
  next_vtk = (vtk_every>0) ? update->ntimestep + vtk_every : -1;
  next_checkpoint = (checkpoint_every>0) ? update->ntimestep + checkpoint_every : -1;
  lbm_step_count = 0;
  md_sub_count = 0;

  setup_grid();
  setup_boundaries(wv_lb);

  if (is3d) stream3d->precompute_wall_flags(*grid3d);
  else      stream2d->precompute_wall_flags(*grid2d);

  if (do_restart) {
    long ckpt_step = 0;
    bool ok;
    if (is3d) ok = CoupLB::IO<CoupLB::D3Q19>::read_checkpoint(*grid3d, world, restart_prefix, ckpt_step);
    else      ok = CoupLB::IO<CoupLB::D2Q9>::read_checkpoint(*grid2d, world, restart_prefix, ckpt_step);
    if (!ok) error->all(FLERR, "fix couplb: failed to read checkpoint");
    if (comm->me==0 && screen)
      fprintf(screen, "CoupLB: restarted from checkpoint step %ld\n", ckpt_step);
    do_restart = false;
  }

  setup_ycomm();

  if (comm->me==0 && screen) {
    fprintf(screen,"CoupLB: %dD grid %dx%dx%d tau=%.6f nu=%.6f rho=%.4f\n", dim,Nx,Ny,Nz,tau,nu,rho);
    fprintf(screen,"CoupLB: nu=%.4e -> nu_lb=%.6f\n", nu, nu_lb);
    fprintf(screen,"CoupLB: dx=%.4e dt_LBM=%.4e vel_scale=%.4e force_scale=%.4e\n", dx,dt_LBM,vel_scale,force_scale);
    fprintf(screen,"CoupLB: kernel=%s xi_ibm=%.4f\n", ibm_kernel==CoupLB::DeltaKernel::ROMA?"roma":"peskin4", xi_ibm);
    if (md_per_lb>1)
      fprintf(screen,"CoupLB: md_per_lb=%d (dt_LBM = %.4e, dt_MD = %.4e)\n",
              md_per_lb, dt_LBM, update->dt);
    if (gxstyle==EQUAL_G || gystyle==EQUAL_G || gzstyle==EQUAL_G) {
      fprintf(screen,"CoupLB: gravity (variable): (%s, %s, %s)\n",
              gxstr ? gxstr : "const", gystr ? gystr : "const", gzstr ? gzstr : "const");
    } else if (gxstyle != NONE_G || gystyle != NONE_G || gzstyle != NONE_G) {
      const double accel_conv = dt_LBM * dt_LBM / dx;
      fprintf(screen,"CoupLB: gravity=(%.2e,%.2e,%.2e) -> lattice=(%.2e,%.2e,%.2e)\n",
              gx_ext,gy_ext,gz_ext, gx_ext*accel_conv,gy_ext*accel_conv,gz_ext*accel_conv);
    }
    if (wall_xlo||wall_xhi) fprintf(screen,"CoupLB: x-walls lo=%d hi=%d\n",wall_xlo,wall_xhi);
    if (wall_ylo||wall_yhi) fprintf(screen,"CoupLB: y-walls lo=%d hi=%d\n",wall_ylo,wall_yhi);
    if (wall_zlo||wall_zhi) fprintf(screen,"CoupLB: z-walls lo=%d hi=%d\n",wall_zlo,wall_zhi);
    if (uw > CoupLB::Constants::ZERO_TOL)
      fprintf(screen,"CoupLB: wall_vel=(%.4e,%.4e,%.4e) -> lattice=(%.4e,%.4e,%.4e)\n",
              wall_vel[0],wall_vel[1],wall_vel[2], wv_lb[0],wv_lb[1],wv_lb[2]);
    if (output_every>0) fprintf(screen,"CoupLB: output every %d -> %s\n",output_every,output_file.c_str());
    if (check_every>0) fprintf(screen,"CoupLB: stability check every %d steps\n",check_every);
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
  int x0=int(std::floor((sl[0]-domain_lo[0])/dx+eps));
  int y0=int(std::floor((sl[1]-domain_lo[1])/dx+eps));
  int z0=int(std::floor((sl[2]-domain_lo[2])/dx+eps));
  int x1=int(std::floor((sh[0]-domain_lo[0])/dx+eps));
  int y1=int(std::floor((sh[1]-domain_lo[1])/dx+eps));
  int z1=int(std::floor((sh[2]-domain_lo[2])/dx+eps));
  int nlx=x1-x0, nly=y1-y0, nlz=is3d?(z1-z0):1;
  if (nlx<=0||nly<=0||(is3d&&nlz<=0)) error->one(FLERR,"fix couplb: local grid empty");
  if (screen) fprintf(screen,"CoupLB rank %d: %dx%dx%d offset(%d,%d,%d)\n",comm->me,nlx,nly,nlz,x0,y0,z0);

  int pn[3][2]; for(int d=0;d<3;d++) { pn[d][0]=comm->procneigh[d][0]; pn[d][1]=comm->procneigh[d][1]; }
  bool px=domain->periodicity[0], py=domain->periodicity[1], pz=domain->periodicity[2];

  if (is3d) {
    grid3d=std::make_unique<CoupLB::Grid<CoupLB::D3Q19>>();
    grid3d->allocate(nlx,nly,nlz,dx_lb,x0,y0,z0,Nx,Ny,Nz);
    grid3d->init_equilibrium(rho_lb,0,0,0);
    bgk3d=std::make_unique<CoupLB::BGK<CoupLB::D3Q19>>(); bgk3d->set_tau(tau);
    stream3d=std::make_unique<CoupLB::Streaming<CoupLB::D3Q19>>();
    stream3d->set_comm(world); stream3d->set_neighbors(pn);
    stream3d->set_periodic(px,py,pz); stream3d->set_nprocs(comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);
    stream3d->allocate_buffers(*grid3d,rho_lb);
  } else {
    grid2d=std::make_unique<CoupLB::Grid<CoupLB::D2Q9>>();
    grid2d->allocate(nlx,nly,1,dx_lb,x0,y0,0,Nx,Ny,1);
    grid2d->init_equilibrium(rho_lb,0,0,0);
    bgk2d=std::make_unique<CoupLB::BGK<CoupLB::D2Q9>>(); bgk2d->set_tau(tau);
    stream2d=std::make_unique<CoupLB::Streaming<CoupLB::D2Q9>>();
    stream2d->set_comm(world); stream2d->set_neighbors(pn);
    stream2d->set_periodic(px,py,false); stream2d->set_nprocs(comm->procgrid[0],comm->procgrid[1],1);
    stream2d->allocate_buffers(*grid2d,rho_lb);
  }
}

void FixCoupLB::setup_boundaries(const double wv_lb[3])
{
  bool axlo=comm->myloc[0]==0, axhi=comm->myloc[0]==comm->procgrid[0]-1;
  bool aylo=comm->myloc[1]==0, ayhi=comm->myloc[1]==comm->procgrid[1]-1;
  bool azlo=comm->myloc[2]==0, azhi=comm->myloc[2]==comm->procgrid[2]-1;
  if (is3d) {
    auto& g=*grid3d;
    if (wall_xlo||wall_xhi) CoupLB::Boundary<CoupLB::D3Q19>::set_walls_x(g,axlo&&wall_xlo>0,axhi&&wall_xhi>0,wall_xlo,wall_xhi);
    if (wall_ylo||wall_yhi) CoupLB::Boundary<CoupLB::D3Q19>::set_walls_y(g,aylo&&wall_ylo>0,ayhi&&wall_yhi>0,wall_ylo,wall_yhi);
    if (wall_zlo||wall_zhi) CoupLB::Boundary<CoupLB::D3Q19>::set_walls_z(g,azlo&&wall_zlo>0,azhi&&wall_zhi>0,wall_zlo,wall_zhi);
    CoupLB::Boundary<CoupLB::D3Q19>::set_wall_velocity(g,2,wv_lb[0],wv_lb[1],wv_lb[2]);
  } else {
    auto& g=*grid2d;
    if (wall_xlo||wall_xhi) CoupLB::Boundary<CoupLB::D2Q9>::set_walls_x(g,axlo&&wall_xlo>0,axhi&&wall_xhi>0,wall_xlo,wall_xhi);
    if (wall_ylo||wall_yhi) CoupLB::Boundary<CoupLB::D2Q9>::set_walls_y(g,aylo&&wall_ylo>0,ayhi&&wall_yhi>0,wall_ylo,wall_yhi);
    CoupLB::Boundary<CoupLB::D2Q9>::set_wall_velocity(g,2,wv_lb[0],wv_lb[1],0);
  }
}

void FixCoupLB::setup_ycomm()
{
  if (ycomm_valid && ycomm!=MPI_COMM_NULL) MPI_Comm_free(&ycomm);
  ycomm_valid = false;
  if (comm->procgrid[1]<=1) return;
  int color = comm->myloc[0] + comm->procgrid[0]*comm->myloc[2];
  MPI_Comm_split(world, color, comm->myloc[1], &ycomm);
  ycomm_valid = true;
}

void FixCoupLB::setup(int vflag)
{
  post_force(vflag);

  // Flush any IBM ghost forces from the setup substep
  if (md_per_lb > 1) {
    if (is3d) stream3d->exchange_forces(*grid3d);
    else      stream2d->exchange_forces(*grid2d);
  }
  md_sub_count = 0;

  // Write step-0 output (end_of_step is not called during LAMMPS setup)
  const long step = (long)update->ntimestep;
  const bool need_macro = (output_every > 0 || vtk_every > 0);

  if (is3d) {
    if (need_macro) grid3d->compute_macroscopic(false);
    if (output_every > 0) {
      if (comm->me == 0) {
        const char* kn = ibm_kernel==CoupLB::DeltaKernel::ROMA ? "roma" : "peskin4";
        FILE* fp = fopen(output_file.c_str(), "w");
        if (fp) {
          fprintf(fp,"# H=Ny=%d tau=%.6f nu=%.4e nu_lb=%.6f rho=%.4f kernel=%s\n",Ny,tau,nu,nu_lb,rho,kn);
          fprintf(fp,"# CoupLB velocity profile\n# Columns: step j y rho ux uy\n#\n");
          fclose(fp);
        }
      }
      write_profile(*grid3d, update->ntimestep);
    }
    if (vtk_every > 0) {
      CoupLB::IO<CoupLB::D3Q19>::write_vtk(*grid3d, world, step, vtk_prefix, domain_lo, dx, vel_scale, force_scale);
      vtk_steps.push_back(step);
      if (comm->me == 0) CoupLB::IO<CoupLB::D3Q19>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_LBM);
    }
  } else {
    if (need_macro) grid2d->compute_macroscopic(false);
    if (output_every > 0) {
      if (comm->me == 0) {
        const char* kn = ibm_kernel==CoupLB::DeltaKernel::ROMA ? "roma" : "peskin4";
        FILE* fp = fopen(output_file.c_str(), "w");
        if (fp) {
          fprintf(fp,"# H=Ny=%d tau=%.6f nu=%.4e nu_lb=%.6f rho=%.4f kernel=%s\n",Ny,tau,nu,nu_lb,rho,kn);
          fprintf(fp,"# CoupLB velocity profile\n# Columns: step j y rho ux uy\n#\n");
          fclose(fp);
        }
      }
      write_profile(*grid2d, update->ntimestep);
    }
    if (vtk_every > 0) {
      CoupLB::IO<CoupLB::D2Q9>::write_vtk(*grid2d, world, step, vtk_prefix, domain_lo, dx, vel_scale, force_scale);
      vtk_steps.push_back(step);
      if (comm->me == 0) CoupLB::IO<CoupLB::D2Q9>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_LBM);
    }
  }
}

void FixCoupLB::post_force(int)
{
  const bool first = (md_sub_count == 0);

  if (first) {
    lbm_step();
    lbm_step_count++;
  }

  ibm_sub_coupling();
  md_sub_count = (md_sub_count + 1) % md_per_lb;
}

void FixCoupLB::lbm_step()
{
  if (is3d) do_lbm_step(*grid3d,*bgk3d,*stream3d);
  else      do_lbm_step(*grid2d,*bgk2d,*stream2d);
}

template<typename L>
void FixCoupLB::do_lbm_step(CoupLB::Grid<L>& g, CoupLB::BGK<L>& b, CoupLB::Streaming<L>& s)
{
  apply_external_force(g);
  g.compute_macroscopic(true);
  b.collide(g);
  s.exchange(g);
  s.stream(g);
  enforce_wall_ghost_fields(g);
  g.clear_forces();
}

void FixCoupLB::update_gravity_variables()
{
  modify->clearstep_compute();
  if (gxstyle == EQUAL_G) gx_ext = input->variable->compute_equal(gxvar);
  if (gystyle == EQUAL_G) gy_ext = input->variable->compute_equal(gyvar);
  if (gzstyle == EQUAL_G) gz_ext = input->variable->compute_equal(gzvar);
  modify->addstep_compute(update->ntimestep + md_per_lb);
}

template<typename L>
void FixCoupLB::apply_external_force(CoupLB::Grid<L>& g)
{
  if (gxstyle == EQUAL_G || gystyle == EQUAL_G || gzstyle == EQUAL_G)
    update_gravity_variables();

  if (gxstyle == NONE_G && gystyle == NONE_G && gzstyle == NONE_G) return;

  const double accel_conv = dt_LBM * dt_LBM / dx;
  const double gx = gx_ext * accel_conv;
  const double gy = gy_ext * accel_conv;
  const double gz = gz_ext * accel_conv;

  if (std::fabs(gx)<CoupLB::Constants::ZERO_TOL &&
      std::fabs(gy)<CoupLB::Constants::ZERO_TOL &&
      std::fabs(gz)<CoupLB::Constants::ZERO_TOL) return;

  const int klo=(L::D==3)?1:0, khi=(L::D==3)?(g.gz-2):0;
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int k=klo;k<=khi;k++)
    for (int j=1;j<=g.gy-2;j++)
      for (int i=1;i<=g.gx-2;i++) {
        const int n=g.idx(i,j,k);
        if (g.type[n]!=0) continue;
        g.fx[n]+=gx*g.rho[n]; g.fy[n]+=gy*g.rho[n]; g.fz[n]+=gz*g.rho[n];
      }
}

/* ------------------------------------------------------------------
   IBM coupling with per-substep recomputation
   
   The subcycling structure is adapted from the "Variant A" scheme of:
   
   Tretyakov, N., & Dünweg, B. (2017). Comput. Phys. Commun., 216, 102-108.

   The fluid velocity field is computed ONCE per LBM cycle (first
   substep only), but the penalty force is recomputed every substep
   using the particle's CURRENT velocity and position.


------------------------------------------------------------------ */
void FixCoupLB::ibm_sub_coupling()
{
  const bool first = (md_sub_count == 0);
  if (is3d) do_ibm_sub_coupling(*grid3d, *stream3d, first);
  else      do_ibm_sub_coupling(*grid2d, *stream2d, first);
}

template<typename L>
void FixCoupLB::do_ibm_sub_coupling(CoupLB::Grid<L>& g, CoupLB::Streaming<L>& s, bool first)
{
  double **x = atom->x, **v = atom->v, **f = atom->f;
  const int nl = atom->nlocal;
  const int *mk = atom->mask;

  enforce_wall_ghost_fields(g);

  if (first) {
    int ng = 0;
    for (int i = 0; i < nl; i++) if (mk[i] & groupbit) ng++;
    int ng_global = 0;
    MPI_Allreduce(&ng, &ng_global, 1, MPI_INT, MPI_SUM, world);
    ibm_has_particles = (ng_global > 0);
    if (!ibm_has_particles) return;

    g.compute_macroscopic(false);
    s.exchange_velocity(g);
  }

  if (!ibm_has_particles) return;

  const double xi_over_dtLBM = xi_ibm / dt_LBM;
  const double inv_N = 1.0 / static_cast<double>(md_per_lb);
  
  // Lagrangian volume for penalty IBM: always dx^D.
  // F_IBM is already a force (not force density), 
  // independent of particle type, size, or marker spacing.
  const double dv = is3d ? (dx*dx*dx) : (dx*dx);

  for (int i = 0; i < nl; i++) {
    if (!(mk[i] & groupbit)) continue;

    auto uf = CoupLB::IBM<L>::interpolate(g, x[i][0], x[i][1], x[i][2],
                                          domain_lo, inv_dx, ibm_kernel);

    const double m = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];

    const double fx_ibm = m * xi_over_dtLBM * (uf[0]*vel_scale - v[i][0]);
    const double fy_ibm = m * xi_over_dtLBM * (uf[1]*vel_scale - v[i][1]);
    const double fz_ibm = m * xi_over_dtLBM * (uf[2]*vel_scale - v[i][2]);

    f[i][0] += fx_ibm;
    f[i][1] += fy_ibm;
    f[i][2] += fz_ibm;

    // Impulse on the particle is f * dt_MD, so must scale by 1/N = dt_MD / dt_LBM
    CoupLB::IBM<L>::spread(g, x[i][0], x[i][1], x[i][2],
                           -fx_ibm * inv_N / force_scale,
                           -fy_ibm * inv_N / force_scale,
                           -fz_ibm * inv_N / force_scale,
                           dv, domain_lo, inv_dx, dx, ibm_kernel);
  }

  if (md_sub_count == md_per_lb - 1) {
    s.exchange_forces(g);
  }
}

template<typename L>
void FixCoupLB::enforce_wall_ghost_fields(CoupLB::Grid<L>& g)
{
  for (int n=0;n<g.ntotal;n++) {
    if (g.type[n]==1) { g.rho[n]=rho_lb; g.ux[n]=g.uy[n]=g.uz[n]=0; }
    else if (g.type[n]==2) { g.rho[n]=rho_lb; g.ux[n]=g.bc_ux[n]; g.uy[n]=g.bc_uy[n]; g.uz[n]=g.bc_uz[n]; }
    else if (g.type[n]==3) { g.rho[n]=rho_lb; g.ux[n]=g.uy[n]=g.uz[n]=0; }
    // type 4 (open): do nothing — fields from zero-gradient streaming
  }
}

template<typename L>
void FixCoupLB::check_stability(CoupLB::Grid<L>& g)
{
  g.compute_macroscopic(false);
  check_stability_precomputed(g);
}

template<typename L>
void FixCoupLB::check_stability_precomputed(CoupLB::Grid<L>& g)
{
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

void FixCoupLB::end_of_step()
{
  const bigint step = update->ntimestep;
  bool chk=false, out=false, vtk=false, ckpt=false;
  if (check_every>0 && step>=next_check) { chk=true; next_check=step+check_every; }
  if (output_every>0 && step>=next_output) { out=true; chk=true; next_output+=output_every; }
  if (vtk_every>0 && step>=next_vtk) { vtk=true; next_vtk=step+vtk_every; }
  if (checkpoint_every>0 && step>=next_checkpoint) { ckpt=true; next_checkpoint=step+checkpoint_every; }
  if (!chk && !out && !vtk && !ckpt) return;

  const bool need_macro = chk || out || vtk;

  if (is3d) {
    if (need_macro) grid3d->compute_macroscopic(false);
    if (chk) check_stability_precomputed(*grid3d);
    if (out) write_profile(*grid3d, step);
    if (vtk) {
      CoupLB::IO<CoupLB::D3Q19>::write_vtk(*grid3d, world, (long)step, vtk_prefix, domain_lo, dx, vel_scale, force_scale);
      vtk_steps.push_back((long)step);
      if (comm->me==0) CoupLB::IO<CoupLB::D3Q19>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_LBM);
    }
    if (ckpt) CoupLB::IO<CoupLB::D3Q19>::write_checkpoint(*grid3d, world, (long)step, checkpoint_prefix);
  } else {
    if (need_macro) grid2d->compute_macroscopic(false);
    if (chk) check_stability_precomputed(*grid2d);
    if (out) write_profile(*grid2d, step);
    if (vtk) {
      CoupLB::IO<CoupLB::D2Q9>::write_vtk(*grid2d, world, (long)step, vtk_prefix, domain_lo, dx, vel_scale, force_scale);
      vtk_steps.push_back((long)step);
      if (comm->me==0) CoupLB::IO<CoupLB::D2Q9>::write_pvd(vtk_pvd_file, vtk_prefix, vtk_steps, dt_LBM);
    }
    if (ckpt) CoupLB::IO<CoupLB::D2Q9>::write_checkpoint(*grid2d, world, (long)step, checkpoint_prefix);
  }
}

template<typename L>
void FixCoupLB::write_profile(CoupLB::Grid<L>& g, bigint step)
{
  const int nyl=g.ny, nx=g.nx, nz=(L::D==3)?g.nz:1, cxz=nx*nz;
  std::vector<double> ul(nyl,0), vl(nyl,0), rl(nyl,0);
  for(int k=0;k<nz;k++) for(int j=0;j<nyl;j++) for(int i=0;i<nx;i++) {
    const int n=g.lidx(i,j,k); ul[j]+=g.ux[n]; vl[j]+=g.uy[n]; rl[j]+=g.rho[n];
  }
  for(int j=0;j<nyl;j++) { ul[j]/=cxz; vl[j]/=cxz; rl[j]/=cxz; }

  int me; MPI_Comm_rank(world,&me);

  auto write_data = [&](FILE* fp, int ny_tot, const double* ux_d, const double* uy_d, const double* rho_d, int yo) {
    fprintf(fp,"# step = %ld\n",(long)step);
    for(int j=0;j<ny_tot;j++)
      fprintf(fp,"%ld %d %.1f %.8f %.10e %.10e\n",(long)step,j,(double)(yo+j)*dx,rho*rho_d[j],vel_scale*ux_d[j],vel_scale*uy_d[j]);
    fprintf(fp,"\n");
  };

  if (!ycomm_valid) {
    if (me==0) {
      FILE* fp=fopen(output_file.c_str(), "a");
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
      FILE* fp=fopen(output_file.c_str(), "a");
      if(!fp){if(screen)fprintf(screen,"CoupLB WARNING: cannot open %s\n",output_file.c_str());}
      else{write_data(fp,nyt,ua.data(),va.data(),ra.data(),0); fclose(fp);}
    }
  }
}
