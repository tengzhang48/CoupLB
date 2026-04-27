"""
Create LAMMPS data file for swimming robot in 2D.
Based on Hu et al. (2018) https://doi.org/10.1038/nature25443

Units: CGS (EMU)

LBM parameter selection: choose dx_LB, then U*
"""

import numpy as np

#===============================================================================
# PHYSICAL PARAMETERS (CGS)
#===============================================================================

# --- Robot geometry ---
L      = 0.37      # cm, total length
w_phys = 0.15      # cm, physical width (for reference only)
h      = 0.0185    # cm, thickness

# --- Material ---
G_mod = 33e4       # dyn/cm^2 (Ba), shear modulus
K_mod = 20*G_mod   # dyn/cm^2, bulk modulus
rho_robot = 1.86   # g/cm^3 (physical)

E  = 9*K_mod*G_mod / (3*K_mod + G_mod)
nu_poisson = (3*K_mod - 2*G_mod) / (2*(3*K_mod + G_mod))

# --- Magnetization ---
M    = 62.         # magnetization parameter [1e5 kA/m in SI]
phi0 = np.pi/4     # initial magnetization phase angle

# --- Actuation ---
B_max = 170.       # Gauss, peak applied field
freq  = 25.0       # Hz, oscillation frequency

# --- Fluid (water) ---
nu_fluid  = 0.01   # cm^2/s, kinematic viscosity
rho_fluid = 1.0    # g/cm^3 (physical)

# --- Gravity ---
g_SI  = 981.0      # cm/s^2
g_eff = g_SI * (rho_robot - rho_fluid) / rho_robot  # buoyancy-corrected

# --- Domain ---
x_length = 3.2
y_length = 4.8

#===============================================================================
# DISCRETIZATION & LBM GRID
#===============================================================================

Na    = 38                # atoms
dx_BP = L / (Na - 1)      # BPM segment length
dx_LB = 0.005             # cm, LBM grid spacing

# In 2D LBM, the simulated slice has depth = dx_LB (one lattice cell).
w = dx_LB

Nx = int(x_length / dx_LB)
Ny = int(y_length / dx_LB)
Gr = dx_BP / dx_LB

# Cross-section properties
A_cs = w * h
I_bm = w * h**3 / 12
J_pm = w * h**3 / 3

# Bond coefficients (bpm/rotational)
d0 = dx_BP
Kr = E * A_cs / d0
Ks = 12 * E * I_bm / d0**3
Kt = G_mod * J_pm / d0    # irrelevant in 2D
Kb = E * I_bm / d0

# BPM damping coefficients
bpm_damp = 1e-6

# Per-atom mass
m_atom = rho_robot * A_cs * dx_BP

#===============================================================================
# LBM PARAMETERS
#===============================================================================

u_char = 30.0    # cm/s, conservative upper bound on max fluid velocity
U_star = 0.1     # target lattice velocity

cs  = 1.0 / np.sqrt(3)
cs2 = 1.0 / 3.0

dt_LB     = U_star * dx_LB / u_char
nu_LB     = nu_fluid * dt_LB / dx_LB**2
tau       = 0.5 + nu_LB / cs2
Ma        = U_star / cs
Re_est    = u_char * L / nu_fluid

#===============================================================================
# TIMESTEP & SUBSTEPPING
#===============================================================================

# --- Frequency estimates (informational only) ---
I_seg  = m_atom * d0**2
# Using "disc" keyword in fix nve/bpm/sphere --> 2D disk inertia: I = (1/2)*m*r^2
I_disc = 0.5 * m_atom * (h/2)**2

omega_r = np.sqrt(Kr / m_atom)
omega_s = np.sqrt(Ks / m_atom)
omega_b = np.sqrt(Kb / I_disc)

mu_segment = M * w * h * dx_BP
omega_mag  = np.sqrt(mu_segment * B_max / I_disc) if (mu_segment * B_max > 0) else 0.0
omega_max  = max(omega_r, omega_s, omega_b, omega_mag)

# Magnetic actuation on tiny, low-mass segments generates large torques, making
# the stable time step much smaller than the BPM and LBM time steps.
# Note that `fix couplb` doesn't damp rotations.
dt_lammps = 1e-7  # s (determined a posteriori)

md_per_lb = max(1, int(np.round(dt_LB / dt_lammps)))   

#===============================================================================
# SIMULATION TIME
#===============================================================================
n_ramp_factor = 1
n_swim_cycles = 10
period = 1.0 / freq

ramp_time  = n_ramp_factor * period
swim_time  = n_swim_cycles * period
total_time = ramp_time + swim_time

n_ramp_steps      = int(np.ceil(ramp_time / dt_lammps))
n_steps_per_cycle = int(np.round(period / dt_lammps))
n_steps_half      = n_steps_per_cycle // 2
n_total_steps     = int(np.ceil(total_time / dt_lammps))
n_lb_steps        = n_total_steps // md_per_lb

#===============================================================================
# PRINT SUMMARY
#===============================================================================

print("=" * 70)
print("SWIMMING ROBOT - PARAMETER SUMMARY")
print("=" * 70)

print(f"\nUNIT SYSTEM: CGS (EMU)")

print(f"\nROBOT GEOMETRY:")
print(f"  L = {L} cm   w_phys = {w_phys} cm   h = {h} cm")
print(f"  w_eff = dx_LB = {w} cm  (2D: out-of-plane depth = 1 lattice cell)")
print(f"  N = {Na} atoms   dx_BP = {dx_BP:.6f} cm   dx_BP/dx_LB = {Gr:.1f}")

print(f"\nMATERIAL:")
print(f"  E = {E:.4e} dyn/cm^2   G = {G_mod:.4e} dyn/cm^2")
print(f"  nu_poisson = {nu_poisson:.4f}   rho_phys = {rho_robot} g/cm^3")

print(f"\nBOND COEFFICIENTS (bpm/rotational):")
print(f"  d0 = {d0:.6f} cm")
print(f"  Kr = {Kr:.4e}   Ks = {Ks:.4e}   Kt = {Kt:.4e}   Kb = {Kb:.4e}")
print(f"  damping = {bpm_damp}")
print(f"  m_atom = {m_atom:.4e} g")

print(f"\nACTUATION:")
print(f"  B_max = {B_max} G   freq = {freq} Hz   phi0 = pi/4")
print(f"  mu_segment = {mu_segment:.4e}")

print(f"\nFLUID (water):")
print(f"  nu = {nu_fluid} cm^2/s   rho_sim = {rho_fluid} g/cm^3")

print(f"\nGRAVITY:")
print(f"  g = {g_SI} cm/s^2   g_eff = {g_eff:.4f} cm/s^2")

print(f"\nDOMAIN (all periodic):")
print(f"  {x_length} x {y_length} cm   ({x_length/L:.1f}L x {y_length/L:.1f}L)")

print(f"\n{'=' * 70}")
print("LATTICE BOLTZMANN PARAMETERS")
print(f"{'=' * 70}")

print(f"\n  dx_LB  = {dx_LB:.6f} cm      Grid: {Nx} x {Ny}")
print(f"  U*     = {U_star:.6f}         u_char = {u_char} cm/s")
print(f"  dt_LB  = {dt_LB:.6e} s")
print(f"  nu_LB  = {nu_LB:.6f}")
print(f"  tau    = {tau:.6f}")
print(f"  Ma     = {Ma:.6f}")
print(f"  Re_est = {Re_est:.1f}")

if tau <= 0.505:
    print(f"\n  WARNING: tau = {tau:.4f} is very close to 0.5!")
if tau <= 0.5:
    print(f"  tau <= 0.5: UNSTABLE!")

print(f"\n{'=' * 70}")
print("TIMESTEP & SUBSTEPPING")
print(f"{'=' * 70}")

print(f"\n  dt_LAMMPS    = {dt_lammps:.6e} s")
print(f"  dt_LB        = {dt_LB:.6e} s")
print(f"  md_per_lb    = {md_per_lb}")

print(f"\n  Frequencies:")
print(f"    omega_r   = {omega_r:.2f}   omega_s = {omega_s:.2f}")
print(f"    omega_b   = {omega_b:.2f}   omega_mag = {omega_mag:.2f}")
print(f"    omega_max = {omega_max:.2f}")
print(f"    I_seg = {I_seg:.4e}     I_disc = {I_disc:.4e}")

dt_crit = np.pi / omega_max
print(f"    dt_crit    = {dt_crit:.4e} s")
print(f"    dt/dt_crit = {dt_lammps/dt_crit:.4f} {'OK' if dt_lammps/dt_crit<0.5 else 'X'}")

#===============================================================================
# CREATE ROBOT ATOMS
#===============================================================================

xc = x_length / 2
yc = y_length / 2

x_atoms = np.linspace(-L/2, L/2, Na) + xc
y_atoms = np.full(Na, yc)
z_atoms = np.zeros(Na)

x_local = np.linspace(-L/2, L/2, Na)
theta   = 2*np.pi*(x_local/L - 0.5) + phi0
mu_mag  = M * w * h * dx_BP
mux = mu_mag * np.cos(theta)
muy = mu_mag * np.sin(theta)
muz = np.zeros(Na)

# In 2D, bpm/sphere: mass = density * pi * (d/2)^2
density_eff_2d = m_atom / (np.pi * (h/2)**2)

print(f"\n{'=' * 70}")
print("ATOM DATA")
print(f"{'=' * 70}")
print(f"  N atoms = {Na}   N bonds = {Na-1}")
print(f"  diameter = {h:.6f} cm (= thickness)")
print(f"  density_eff (2D disk) = {density_eff_2d:.4e} g/cm^3")
print(f"  per-atom mass = {m_atom:.4e} g  (rho_sim * w * h * dx_BP)")
print(f"  dipole magnitude = {mu_mag:.4e}")

#===============================================================================
# WRITE DATA FILE
#===============================================================================

filename = 'robot.lam'
with open(filename, 'w') as f:
    f.write(f'LAMMPS data file - CGS\n\n')

    f.write(f'{Na} atoms\n')
    f.write(f'{Na-1} bonds\n\n')

    f.write(f'1 atom types\n')
    f.write(f'1 bond types\n\n')

    f.write(f'{0:.6f} {x_length:.6f} xlo xhi\n')
    f.write(f'{0:.6f} {y_length:.6f} ylo yhi\n')
    f.write(f'{-0.5*dx_LB:.6f} {0.5*dx_LB:.6f} zlo zhi\n\n')

    f.write('Atoms # hybrid \n\n')
    for i in range(Na):
        f.write(f'{i+1} 1 '
                f'{x_atoms[i]:.6f} {y_atoms[i]:.6f} {z_atoms[i]:.6f} '
                f'1 {h:.6f} {density_eff_2d:.6e} 0 '
                f'{mux[i]:.6e} {muy[i]:.6e} {muz[i]:.6e}\n')

    f.write('\nBonds\n\n')
    for i in range(Na-1):
        f.write(f'{i+1} 1 {i+1} {i+2}\n')

print(f"\nWritten: {filename}")

#===============================================================================
# SIMULATION TIME
#===============================================================================

print(f"\n{'=' * 70}")
print("SIMULATION TIME")
print(f"{'=' * 70}")
print(f"  Ramp:      {ramp_time:.4f} s  ({n_ramp_steps:,} LAMMPS steps)")
print(f"  Swimming:  {n_swim_cycles} cycles = {swim_time:.4f} s")
print(f"  Total:     {total_time:.4f} s  ({n_total_steps:,} LAMMPS steps, {n_lb_steps:,} LB steps)")
print(f"  Steps/cycle = {n_steps_per_cycle:,}   Steps/half = {n_steps_half:,}")