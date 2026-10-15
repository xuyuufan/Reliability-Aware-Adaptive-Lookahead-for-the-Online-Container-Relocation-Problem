# Preserved Minmax source provenance

- Repository: Marco Caserta, [`marcocaserta/BRP`](https://github.com/marcocaserta/BRP)
- Immutable commit: [`2f4c88001e5edbe0a782672f7815e7c96f2e738c`](https://github.com/marcocaserta/BRP/tree/2f4c88001e5edbe0a782672f7815e7c96f2e738c)
- Original path: [`src/heuristic.cpp`](https://github.com/marcocaserta/BRP/blob/2f4c88001e5edbe0a782672f7815e7c96f2e738c/src/heuristic.cpp)
- Preserved local path: [`vendor/heuristic.cpp`](vendor/heuristic.cpp)
- Upstream source file was fetched at the immutable commit without edits. Its header identifies GNU GPL version 2 or, at the recipient's option, a later version. The commit tree has no `LICENSE`, `COPYING`, or `LICENSE.txt`; [`COPYING`](COPYING) is the complete official GNU GPL-2 text from <https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt>.
- The source has a reusable `min_el_i` helper (lines 128–136) and `max_in_choosestack` min-safe/otherwise-max score core (lines 139–165). The Python module ports those helpers with attribution. The C++ `block_heuristic` wrapper (lines 195–203) sends any available empty stack first, before calling that score core. PLAN-052 feeds empty-stack value `N+1` into the ported score core, following paper Eq. (11), and does not copy the wrapper's empty-first action.
- The retained upstream C++ file is preserved unchanged as source/provenance material; the solver ports only the scoring core to Python and does not compile or execute C++.
- SHA256 `vendor/heuristic.cpp`: `595c9ded28fee2bbbf8c379fc3887fe0410d3c943456f00ddfc718e5c1b35cf9`.
- SHA256 `COPYING`: `edaef632cbb643e4e7a221717a6c441a4c1a7c918e6e4d56debc3d8739b233f6`.
