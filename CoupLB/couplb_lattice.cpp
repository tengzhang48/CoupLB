// Out-of-class definitions for static constexpr member arrays.
//
// Required for C++14: static constexpr members are only "declared"
// inside the class. Array element access (e.g. Lattice::e[q][0])
// constitutes an ODR-use requiring a definition in one translation unit.
//
// C++17 makes static constexpr implicitly inline, so this file is
// unnecessary with -std=c++17 but harmless to include.

#include "couplb_lattice.h"

namespace LAMMPS_NS {
namespace CoupLB {

constexpr int    D2Q9::D;
constexpr int    D2Q9::Q;
constexpr double D2Q9::cs2;
constexpr int    D2Q9::e[9][3];
constexpr double D2Q9::w[9];
constexpr int    D2Q9::opp[9];
constexpr int    D2Q9::reflect[2][9];

constexpr int    D3Q19::D;
constexpr int    D3Q19::Q;
constexpr double D3Q19::cs2;
constexpr int    D3Q19::e[19][3];
constexpr double D3Q19::w[19];
constexpr int    D3Q19::opp[19];
constexpr int    D3Q19::reflect[3][19];

} // namespace CoupLB
} // namespace LAMMPS_NS
