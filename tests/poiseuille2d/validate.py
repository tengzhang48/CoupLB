#!/usr/bin/env python3
"""
validate.py -- Compare CoupLB output against analytical solutions.

Reads Ny, nu, nu_lb from the profile header. Test-specific parameters
(body force, wall velocity) are passed on the command line.

Usage:
  python validate.py poiseuille <file> <gravity>
  python validate.py couette    <file> <wall_vel>

Examples:
  python validate.py poiseuille couplb_poiseuille.dat 1e-5
  python validate.py couette    couplb_couette.dat 0.05
"""
import sys
import re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def parse_header(filename):
    """Read Ny, nu, nu_lb, rho from the profile header line."""
    params = {}
    with open(filename) as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith('#'):
                break
            # Match: # H=Ny=32 tau=0.800000 nu=1.0000e-01 nu_lb=0.100000 rho=1.0000 kernel=roma
            m = re.search(r'H=Ny=(\d+)', s)
            if m:
                params['Ny'] = int(m.group(1))
            for key in ['tau', 'nu', 'nu_lb', 'rho']:
                m = re.search(key + r'=([\d.eE+\-]+)', s)
                if m:
                    params[key] = float(m.group(1))
    return params


def read_last_block(filename):
    """Read the last timestep block from a CoupLB profile file."""
    steps = {}
    current = None
    with open(filename) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith('# step ='):
                current = int(s.split('=')[1])
                steps[current] = []
                continue
            if s.startswith('#') or current is None:
                continue
            parts = s.split()
            if len(parts) >= 6:
                steps[current].append([float(x) for x in parts])
    if not steps:
        print(f"ERROR: no data in {filename}")
        sys.exit(1)
    last = max(steps.keys())
    data = np.array(steps[last])
    print(f"Read step {last}, {len(data)} nodes")
    return last, data


def validate_poiseuille(fname, g_phys):
    """Validate Poiseuille flow against analytical solution.

    Analytical (half-way bounce-back):
      u(y) = g/(2*nu) * (y + 0.5*dx) * (H - 0.5*dx - y)
    where H = Ny*dx, y is physical coordinate from profile.
    """
    params = parse_header(fname)
    Ny = params.get('Ny')
    nu = params.get('nu')
    if Ny is None or nu is None:
        print("ERROR: could not read Ny or nu from header")
        sys.exit(1)

    step, data = read_last_block(fname)
    y, ux = data[:, 2], data[:, 4]

    # Derive dx from y-column spacing
    if len(y) > 1:
        dx = y[1] - y[0]
    else:
        dx = 1.0
    H = Ny * dx

    ux_exact = g_phys / (2 * nu) * (y + 0.5 * dx) * (H - 0.5 * dx - y)

    err = np.abs(ux - ux_exact)
    ux_max = np.max(ux_exact)
    rel = np.max(err) / ux_max * 100 if ux_max > 0 else float('inf')

    print(f"\n{'=' * 55}")
    print(f"  POISEUILLE (step {step})")
    print(f"{'=' * 55}")
    print(f"  Ny={Ny}  nu={nu:.4e}  g={g_phys:.4e}  dx={dx:.4f}")
    print(f"  u_max exact={ux_max:.6e}  LBM={np.max(ux):.6e}")
    print(f"  max|err|={np.max(err):.6e}  rel={rel:.4f}%  L2={np.sqrt(np.mean(err**2)):.6e}")
    status = 'PASS' if rel < 1 else 'MARGINAL' if rel < 5 else 'FAIL'
    print(f"  -> {status}")
    print(f"{'=' * 55}\n")

    if HAS_PLT:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot(ux_exact, y, 'b-', lw=2, label='Exact')
        a1.plot(ux, y, 'ro', ms=4, label='CoupLB')
        a1.set_xlabel('$u_x$')
        a1.set_ylabel('$y$')
        a1.legend()
        a1.grid(alpha=0.3)
        a2.plot(y, err, 'k.-')
        a2.set_xlabel('$y$')
        a2.set_ylabel('|error|')
        a2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('poiseuille_validation.png', dpi=150)
        print("  Plot -> poiseuille_validation.png\n")


def validate_couette(fname, Uw):
    """Validate Couette flow against analytical solution.

    Analytical (half-way bounce-back):
      u(y) = Uw * (y + 0.5*dx) / (Ny*dx)
    """
    params = parse_header(fname)
    Ny = params.get('Ny')
    if Ny is None:
        print("ERROR: could not read Ny from header")
        sys.exit(1)

    step, data = read_last_block(fname)
    y, ux = data[:, 2], data[:, 4]

    if len(y) > 1:
        dx = y[1] - y[0]
    else:
        dx = 1.0
    H = Ny * dx

    ux_exact = Uw * (y + 0.5 * dx) / H

    err = np.abs(ux - ux_exact)
    rel = np.max(err) / abs(Uw) * 100

    print(f"\n{'=' * 55}")
    print(f"  COUETTE (step {step})")
    print(f"{'=' * 55}")
    print(f"  Ny={Ny}  Uw={Uw:.4e}  dx={dx:.4f}")
    print(f"  u_max exact={np.max(ux_exact):.6e}  LBM={np.max(ux):.6e}")
    print(f"  max|err|={np.max(err):.6e}  rel={rel:.4f}%  L2={np.sqrt(np.mean(err**2)):.6e}")
    status = 'PASS' if rel < 1 else 'MARGINAL' if rel < 5 else 'FAIL'
    print(f"  -> {status}")
    print(f"{'=' * 55}\n")

    if HAS_PLT:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot(ux_exact, y, 'b-', lw=2, label='Exact')
        a1.plot(ux, y, 'ro', ms=4, label='CoupLB')
        a1.set_xlabel('$u_x$')
        a1.set_ylabel('$y$')
        a1.legend()
        a1.grid(alpha=0.3)
        a2.plot(y, err, 'k.-')
        a2.set_xlabel('$y$')
        a2.set_ylabel('|error|')
        a2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('couette_validation.png', dpi=150)
        print("  Plot -> couette_validation.png\n")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python validate.py poiseuille <file> <gravity>")
        print("  python validate.py couette    <file> <wall_vel>")
        sys.exit(1)

    test = sys.argv[1].lower()
    fname = sys.argv[2]
    param = float(sys.argv[3])

    if test == 'poiseuille':
        validate_poiseuille(fname, param)
    elif test == 'couette':
        validate_couette(fname, param)
    else:
        print(f"Unknown test: {test}")
        sys.exit(1)