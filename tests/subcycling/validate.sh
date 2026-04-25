#!/usr/bin/env bash

#SBATCH --job-name=ibm-sub
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=%x.%N.%j.out

set -uo pipefail

# IBM subcycling unit test (bash + bc + awk, no Python)
# Validates md_per_lb=4 and md_per_lb=10 match baseline (md_per_lb=1)
#
# Run:
#   sbatch validate.sh
#   bash validate.sh
#   NPROCS=4 LAMMPS_BIN=lmp_couplb bash validate.sh

LAMMPS_BIN="$HOME/coupLB/lammps-10Dec2025/lmp_couplb"
NPROCS=${SLURM_NTASKS}
LMP="mpirun -n ${NPROCS} ${LAMMPS_BIN}"

DX=0.625
NX=96
NY=64
NU=0.1
RHO=1.26
G0=2e-4
DT_LB=0.08333
PMASS=0.161
V0X=-0.3

XLEN=$(echo "${NX} * ${DX}" | bc -l)
YLEN=$(echo "${NY} * ${DX}" | bc -l)
XC=$(echo "${XLEN} / 2" | bc -l)
YC=$(echo "${YLEN} / 2" | bc -l)

DT_B=$(echo "${DT_LB} / 4" | bc -l)
DT_C=$(echo "${DT_LB} / 10" | bc -l)

SPINUP_STEPS=10000
LBM_RUN=2000
LBM_PRINT=10

NSTEPS_A=$(( LBM_RUN * 1 ))
NSTEPS_B=$(( LBM_RUN * 4 ))
NSTEPS_C=$(( LBM_RUN * 10 ))

PFREQ_A=$(( LBM_PRINT * 1 ))
PFREQ_B=$(( LBM_PRINT * 4 ))
PFREQ_C=$(( LBM_PRINT * 10 ))

CFREQ_A=$(( NSTEPS_A / 2 ))
CFREQ_B=$(( NSTEPS_B / 2 ))
CFREQ_C=$(( NSTEPS_C / 2 ))

THERMO_A=$(( NSTEPS_A / 5 ))
THERMO_B=$(( NSTEPS_B / 5 ))
THERMO_C=$(( NSTEPS_C / 5 ))

NU_LB=$(echo "${NU} * ${DT_LB} / (${DX} * ${DX})" | bc -l)
TAU=$(echo "0.5 + 3.0 * ${NU_LB}" | bc -l)

RUN_OK=true

echo "============================================================"
echo " IBM Subcycling Unit Test"
echo " Grid: ${NX}x${NY}  dx=${DX}  nu=${NU}  tau=${TAU}"
echo " dt_LB=${DT_LB}  G0=${G0}  v0x=${V0X}"
echo " LAMMPS: ${LAMMPS_BIN}  ranks: ${NPROCS}"
echo "============================================================"
echo ""

# ============================================================
# Phase 1: Spin up Poiseuille flow
# ============================================================
cat > in.spinup <<EOF
units           si
dimension       2
boundary        p f p
atom_style      atomic
atom_modify     map array

region          box block 0.0 ${XLEN} 0.0 ${YLEN} -0.5 0.5
create_box      1 box

create_atoms    1 single ${XC} ${YC} 0.0
mass            1 ${PMASS}
pair_style      none
comm_modify     cutoff 2.0

group           fluid_only empty

timestep        ${DT_LB}

fix flow fluid_only couplb ${NX} ${NY} 1 ${NU} ${RHO} &
    dx ${DX} &
    wall_y 1 1 &
    gravity ${G0} 0.0 0.0 &
    kernel roma &
    check_every 5000 &
    checkpoint ${SPINUP_STEPS} ckpt_pois

thermo          2000
run             ${SPINUP_STEPS}
print "=== Spinup complete ==="
EOF

echo "--- Phase 1: Spinning up Poiseuille flow (${SPINUP_STEPS} steps) ---"
${LMP} -in in.spinup -log log.spinup 2>&1 | tail -20
RC=$?
if [ $RC -ne 0 ]; then echo "*** SPINUP FAILED (exit $RC) ***"; RUN_OK=false; fi
echo ""

# ============================================================
# Phase 2A: Baseline  md_per_lb=1
# ============================================================
cat > in.track_base <<EOF
units           si
dimension       2
boundary        p f p
atom_style      atomic
atom_modify     map array

region          box block 0.0 ${XLEN} 0.0 ${YLEN} -0.5 0.5
create_box      1 box

create_atoms    1 single ${XC} ${YC} 0.0
mass            1 ${PMASS}
pair_style      none
comm_modify     cutoff 2.0
neigh_modify    every 1 delay 0 check no

velocity        all set ${V0X} 0.0 0.0
timestep        ${DT_LB}

fix nve all nve
fix flow all couplb ${NX} ${NY} 1 ${NU} ${RHO} &
    dx ${DX} &
    wall_y 1 1 &
    gravity ${G0} 0.0 0.0 &
    kernel roma &
    restart ckpt_pois &
    check_every ${CFREQ_A}

variable        t   equal time
variable        px  equal x[1]
variable        py  equal y[1]
variable        vxp equal vx[1]
variable        vyp equal vy[1]
variable        fxp equal fx[1]
variable        fyp equal fy[1]

fix prt all print ${PFREQ_A} &
    "\${t} \${px} \${py} \${vxp} \${vyp} \${fxp} \${fyp}" &
    file track_base.dat screen no &
    title "# t x y vx vy fx fy"

thermo          ${THERMO_A}
run             ${NSTEPS_A}
EOF

echo "--- Phase 2A: Baseline  md_per_lb=1  dt=${DT_LB} ---"
${LMP} -in in.track_base -log log.track_base 2>&1 | tail -8
RC=$?
if [ $RC -ne 0 ]; then echo "*** BASELINE FAILED (exit $RC) ***"; RUN_OK=false; fi
echo ""

# ============================================================
# Phase 2B: md_per_lb=4
# ============================================================
cat > in.track_sub4 <<EOF
units           si
dimension       2
boundary        p f p
atom_style      atomic
atom_modify     map array

region          box block 0.0 ${XLEN} 0.0 ${YLEN} -0.5 0.5
create_box      1 box

create_atoms    1 single ${XC} ${YC} 0.0
mass            1 ${PMASS}
pair_style      none
comm_modify     cutoff 2.0
neigh_modify    every 4 delay 0 check no

velocity        all set ${V0X} 0.0 0.0
timestep        ${DT_B}

fix nve all nve
fix flow all couplb ${NX} ${NY} 1 ${NU} ${RHO} &
    dx ${DX} &
    md_per_lb 4 &
    wall_y 1 1 &
    gravity ${G0} 0.0 0.0 &
    kernel roma &
    restart ckpt_pois &
    check_every ${CFREQ_B}

variable        t   equal time
variable        px  equal x[1]
variable        py  equal y[1]
variable        vxp equal vx[1]
variable        vyp equal vy[1]
variable        fxp equal fx[1]
variable        fyp equal fy[1]

fix prt all print ${PFREQ_B} &
    "\${t} \${px} \${py} \${vxp} \${vyp} \${fxp} \${fyp}" &
    file track_sub4.dat screen no &
    title "# t x y vx vy fx fy"

thermo          ${THERMO_B}
run             ${NSTEPS_B}
EOF

echo "--- Phase 2B: md_per_lb=4   dt=${DT_B} ---"
${LMP} -in in.track_sub4 -log log.track_sub4 2>&1 | tail -8
RC=$?
if [ $RC -ne 0 ]; then echo "*** SUB4 FAILED (exit $RC) ***"; RUN_OK=false; fi
echo ""

# ============================================================
# Phase 2C: md_per_lb=10
# ============================================================
cat > in.track_sub10 <<EOF
units           si
dimension       2
boundary        p f p
atom_style      atomic
atom_modify     map array

region          box block 0.0 ${XLEN} 0.0 ${YLEN} -0.5 0.5
create_box      1 box

create_atoms    1 single ${XC} ${YC} 0.0
mass            1 ${PMASS}
pair_style      none
comm_modify     cutoff 2.0
neigh_modify    every 10 delay 0 check no

velocity        all set ${V0X} 0.0 0.0
timestep        ${DT_C}

fix nve all nve
fix flow all couplb ${NX} ${NY} 1 ${NU} ${RHO} &
    dx ${DX} &
    md_per_lb 10 &
    wall_y 1 1 &
    gravity ${G0} 0.0 0.0 &
    kernel roma &
    restart ckpt_pois &
    check_every ${CFREQ_C}

variable        t   equal time
variable        px  equal x[1]
variable        py  equal y[1]
variable        vxp equal vx[1]
variable        vyp equal vy[1]
variable        fxp equal fx[1]
variable        fyp equal fy[1]

fix prt all print ${PFREQ_C} &
    "\${t} \${px} \${py} \${vxp} \${vyp} \${fxp} \${fyp}" &
    file track_sub10.dat screen no &
    title "# t x y vx vy fx fy"

thermo          ${THERMO_C}
run             ${NSTEPS_C}
EOF

echo "--- Phase 2C: md_per_lb=10  dt=${DT_C} ---"
${LMP} -in in.track_sub10 -log log.track_sub10 2>&1 | tail -8
RC=$?
if [ $RC -ne 0 ]; then echo "*** SUB10 FAILED (exit $RC) ***"; RUN_OK=false; fi
echo ""

# ============================================================
# Phase 3: Analysis
# ============================================================
echo "============================================================"
echo " RESULTS"
echo "============================================================"
echo ""

if [ "$RUN_OK" = false ]; then
    echo "WARNING: One or more LAMMPS runs failed."
    echo ""
fi

PASSED=0
FAILED=0

count_lines() {
    if [ -f "$1" ]; then
        awk '!/^#/ && NF>=6' "$1" | wc -l
    else
        echo 0
    fi
}

NB=$(count_lines track_base.dat)
N4=$(count_lines track_sub4.dat)
N10=$(count_lines track_sub10.dat)
echo "Data points:  base=${NB}  sub4=${N4}  sub10=${N10}"
echo ""

# Test 1: particle moves rightward
echo "TEST 1: vx_final > 0"
if [ "$NB" -eq 0 ]; then
    echo "  SKIP"; FAILED=$((FAILED+1))
else
    VXF=$(awk '!/^#/ && NF>=6 { vx=$4 } END { printf "%.6f", vx }' track_base.dat)
    echo "  vx final: ${VXF}"
    T1=$(awk -v v="${VXF}" 'BEGIN { print (v+0 > 0) ? "PASS" : "FAIL" }')
    if [ "$T1" = "PASS" ]; then echo "  -> PASS"; PASSED=$((PASSED+1))
    else echo "  -> FAIL"; FAILED=$((FAILED+1)); fi
fi
echo ""

# Test 2: subcycling matches baseline
TOLERANCE=0.02
echo "TEST 2: Subcycling matches baseline (tol $(echo "${TOLERANCE} * 100" | bc)%)"
echo ""

compare_run() {
    local LABEL=$1 TFILE=$2 NPTS=$3
    if [ "$NB" -lt 10 ] || [ "$NPTS" -lt 10 ]; then
        echo "  ${LABEL}: insufficient data -> SKIP"
        FAILED=$((FAILED+1))
        return
    fi

    RESULT=$(awk -v tol="${TOLERANCE}" '
    BEGIN { maxerr=0; skip=5; n=0 }
    FNR==NR {
        if (/^#/ || NF<6) next
        n++; vxb[n] = $4; next
    }
    {
        if (/^#/ || NF<6) next
        n2++; vxt = $4
        if (n2 > skip && vxb[n2]+0 != 0) {
            vr = vxb[n2]+0; vt = vxt+0
            if (vr < 0) vr = -vr
            if (vr < 0.01) next
            err = (vt - (vxb[n2]+0))
            if (err < 0) err = -err
            err = err / vr
            if (err > maxerr) maxerr = err
        }
    }
    END {
        nn = (n < n2) ? n : n2
        vr = vxb[nn]+0; vt = vxt+0
        ratio = (vr != 0) ? vt/vr : 0
        ok = (maxerr < tol) ? 1 : 0
        printf "%.6e %.6f %.6f %.6f %d", maxerr, vr, vt, ratio, ok
    }' track_base.dat "${TFILE}")

    MAXERR=$(echo "${RESULT}" | awk '{print $1}')
    VXR=$(echo "${RESULT}" | awk '{print $2}')
    VXT=$(echo "${RESULT}" | awk '{print $3}')
    RATIO=$(echo "${RESULT}" | awk '{print $4}')
    OK=$(echo "${RESULT}" | awk '{print $5}')

    echo "  ${LABEL}:"
    echo "    max |dvx/vx|: ${MAXERR}  (tol ${TOLERANCE})"
    echo "    final: vx_ref=${VXR}  vx_test=${VXT}  ratio=${RATIO}"
    if [ "$OK" -eq 1 ]; then echo "    -> PASS"; PASSED=$((PASSED+1))
    else echo "    -> FAIL"; FAILED=$((FAILED+1)); fi
    echo ""
}

compare_run "md_per_lb=4  vs baseline" track_sub4.dat "$N4"
compare_run "md_per_lb=10 vs baseline" track_sub10.dat "$N10"

# Test 3: no vertical drift
echo "TEST 3: No vertical drift"
if [ "$NB" -eq 0 ]; then
    echo "  SKIP"; FAILED=$((FAILED+1))
else
    VYMAX=$(awk '!/^#/ && NF>=6 { v=$5; if(v<0)v=-v; if(v>m)m=v } END { printf "%.2e", m }' track_base.dat)
    echo "  max |vy| = ${VYMAX}"
    T3=$(awk -v v="${VYMAX}" 'BEGIN { print (v+0 < 0.01) ? "PASS" : "FAIL" }')
    if [ "$T3" = "PASS" ]; then echo "  -> PASS"; PASSED=$((PASSED+1))
    else echo "  -> FAIL"; FAILED=$((FAILED+1)); fi
fi
echo ""

# Test 4: force magnitude sanity
echo "TEST 4: Initial force magnitude"
if [ "$NB" -eq 0 ]; then
    echo "  SKIP"; FAILED=$((FAILED+1))
else
    FX0=$(awk '!/^#/ && NF>=6 { printf "%.4f", $6; exit }' track_base.dat)
    echo "  fx[0] = ${FX0}  (expect O(1))"
    T4=$(awk -v f="${FX0}" 'BEGIN { af=f; if(af<0)af=-af; print (af>0.1 && af<20) ? "PASS" : "FAIL" }')
    if [ "$T4" = "PASS" ]; then echo "  -> PASS"; PASSED=$((PASSED+1))
    else echo "  -> FAIL"; FAILED=$((FAILED+1)); fi
fi
echo ""

# Trajectory table
echo "--- Velocity comparison (every 20th point) ---"
printf "%10s  %11s  %11s  %11s  %8s  %8s\n" t vx_base vx_sub4 vx_sub10 r4 r10
echo "-----------------------------------------------------------------------"

if [ -f track_base.dat ] && [ -f track_sub4.dat ] && [ -f track_sub10.dat ]; then
    paste <(awk '!/^#/ && NF>=6 { print $1, $4 }' track_base.dat) \
          <(awk '!/^#/ && NF>=6 { print $4 }' track_sub4.dat) \
          <(awk '!/^#/ && NF>=6 { print $4 }' track_sub10.dat) \
    | awk 'NR%20==1 || NR==1 {
        t=$1; vb=$2; v4=$3; v10=$4
        r4  = (vb+0 != 0) ? v4/vb : 0
        r10 = (vb+0 != 0) ? v10/vb : 0
        printf "%10.2f  %11.6f  %11.6f  %11.6f  %8.4f  %8.4f\n", t, vb, v4, v10, r4, r10
    }'
fi
echo ""

# Summary
TOTAL=$((PASSED + FAILED))
echo "============================================================"
echo "  ${PASSED}/${TOTAL} TESTS PASSED"
if [ $FAILED -eq 0 ] && [ "$RUN_OK" = true ]; then
    echo "  Subcycling OK."
else
    echo "  SOME TESTS FAILED -- see above"
fi
echo "============================================================"
echo ""
echo "Files:"
echo "  Logs:  log.spinup log.track_base log.track_sub4 log.track_sub10"
echo "  Data:  track_base.dat track_sub4.dat track_sub10.dat"

[ $FAILED -eq 0 ] && [ "$RUN_OK" = true ] && exit 0 || exit 1