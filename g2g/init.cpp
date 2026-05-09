/* headers */
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <fenv.h>
#include <signal.h>
#include <cassert>
#include <cstdint>
#include "hardware_topo.h"
#include "common.h"
#include "init.h"
#include "timer.h"
#include "partition.h"
#include "matrix.h"

#if USE_LIBXC
#include "libxc/libxcproxy.h"
#endif

//#include "qmmm_forces.h"
using std::cout;
using std::endl;
using std::boolalpha;
using std::runtime_error;
using std::ifstream;
using std::string;
using namespace G2G;

Partition partition;

/* global variables */
namespace G2G {
  FortranVars fortran_vars;
  int cpu_threads=0;
  int gpu_threads=0;
  int recommended_blas_threads=0;
}

/* methods */
//===========================================================================================
extern "C" void g2g_init_(void) {
  int max_threads = 1;
  if (getenv("OMP_NUM_THREADS")) {
    max_threads = omp_get_max_threads();
  }

#if GPU_KERNELS
  if (verbose > 3) cout << "G2G initialisation." << endl;
  cuInit(0);
  int devcount = cudaGetGPUCount();
  int devnum = -1;
  cudaDeviceProp devprop;
  for (int i = 0; i < devcount; i++) {
    if (cudaGetDeviceProperties(&devprop, i) != cudaSuccess)
      throw runtime_error("Could not get device propierties!");
    if (verbose > 2) cout << "  GPU Device used: " << devprop.name << endl;
  }
  G2G::gpu_threads = devcount;

  // Keep the default stream-ordered memory pool from returning freed blocks
  // to the driver between allocations. Without this, cudaFreeAsync trims
  // back to the driver on every call and we lose the pooling benefit.
  for (int i = 0; i < devcount; i++) {
    cudaMemPool_t mempool;
    if (cudaDeviceGetDefaultMemPool(&mempool, i) == cudaSuccess) {
      uint64_t threshold = UINT64_MAX;
      cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold,
                              &threshold);
    }
  }
#endif
  // Auto-tune: set recommended_blas_threads from hardware topology.
  // When overlap is requested and OMP_NUM_THREADS is not explicitly set,
  // cap OMP threads to leave headroom for the concurrent int3lu BLAS section.
  // Formula calibrated on 5800X3D (8 phys / 16 logical): OMP=6, BLAS=4 optimal.
  {
    int phys = detect_physical_cores();
    G2G::recommended_blas_threads = ::recommended_blas_threads(phys);
    const char* ov = getenv("LIO_OVERLAP_INT3LU_G2G");
    bool overlap_on = (ov && ov[0] == '1' && ov[1] == '\0');
    if (overlap_on && getenv("OMP_NUM_THREADS") == nullptr) {
      int n_omp = ::recommended_omp_threads(phys);
      omp_set_num_threads(n_omp);
      if (verbose > 1)
        printf("  [overlap] auto OMP_NUM_THREADS=%d BLAS=%d (phys_cores=%d)\n",
               n_omp, G2G::recommended_blas_threads, phys);
    }
  }
#if CPU_KERNELS
  G2G::cpu_threads = max_threads - G2G::gpu_threads;
  if (G2G::cpu_threads < 0) G2G::cpu_threads = 0;
#endif
  if (G2G::gpu_threads == 0 && G2G::cpu_threads == 0)
    throw runtime_error(
        "  ERROR: Either a gpu or a cpu thread is needed to run G2G");
  if (verbose > 2) cout << "  Using " << G2G::cpu_threads << " CPU Threads and "
       << G2G::gpu_threads << " GPU Threads." << endl;

}
extern "C" int g2g_recommended_blas_threads_(void) {
  return G2G::recommended_blas_threads;
}
//==========================================================================================
namespace G2G {
void gpu_set_variables(void);
template <class T>
void gpu_set_atom_positions(const HostMatrix<T>& m);
}
//==========================================================================================
extern "C" void g2g_parameter_init_(
    const unsigned int& norm, const unsigned int& natom,
    const unsigned int& max_atoms,
    const unsigned int& ngaussians,  // const unsigned int& ngaussiansd,
    double* r, double* Rm, const unsigned int* Iz, const unsigned int* Nr,
    const unsigned int* Nr2, unsigned int* Nuc, const unsigned int& M,
    unsigned int* ncont, const unsigned int* nshell, double* c, double* a,
    double* rho_vec, double* fock_vec, double* fockb_vec,
    double* rhoalpha, double* rhobeta,
    const unsigned int& nco, bool& OPEN, const unsigned int& nunp,
    const unsigned int& nopt, const unsigned int& Iexch, double* e, double* e2,
    double* e3, double* wang, double* wang2, double* wang3,
    bool& use_libxc, const unsigned int& ex_functional_id, 
    const unsigned int& ec_functional_id, bool& becke){
  fortran_vars.atoms = natom;
  fortran_vars.max_atoms = max_atoms;
  fortran_vars.gaussians = ngaussians;

  fortran_vars.do_forces = (nopt == 2);
  fortran_vars.normalize = norm;
  fortran_vars.normalization_factor =
      (fortran_vars.normalize ? (1.0 / sqrt(3)) : 1.0);

#ifdef _DEBUG
  // trap floating point exceptions on debug
  signal(SIGFPE, SIG_DFL);
  //feenableexcept(FE_INVALID);
  // This line interferes with Lapack routines on floating point error catching.
  // Commented out until a better solution is found.
#endif

  /* MO BASIS SET */
  fortran_vars.s_funcs = nshell[0];
  fortran_vars.p_funcs = nshell[1] / 3;
  fortran_vars.d_funcs = nshell[2] / 6;
  fortran_vars.spd_funcs =
      fortran_vars.s_funcs + fortran_vars.p_funcs + fortran_vars.d_funcs;
  // M =	# of contractions
  fortran_vars.m = M;
  fortran_vars.nco = nco;
  fortran_vars.iexch = Iexch;
  if (Iexch == 4 || Iexch == 5)
    cout << "WARNING: Iexch=4,5 not working properly." << endl;
  fortran_vars.lda = (Iexch <= 3);
  fortran_vars.gga = !fortran_vars.lda;
  assert(0 < Iexch && Iexch <= 9);

  fortran_vars.atom_positions_pointer =
      FortranMatrix<double>(r, fortran_vars.atoms, 3, max_atoms);
  fortran_vars.atom_types.resize(fortran_vars.atoms);
  fortran_vars.atom_Z.resize(fortran_vars.atoms);
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    fortran_vars.atom_types(i) = Iz[i] - 1;
    fortran_vars.atom_Z(i)     = Iz[i];
  }

  fortran_vars.shells1.resize(fortran_vars.atoms);
  fortran_vars.shells2.resize(fortran_vars.atoms);
  fortran_vars.rm.resize(fortran_vars.atoms);
  fortran_vars.rm_base.resize(fortran_vars.atoms);

  /* ignore the 0th element on these */
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    fortran_vars.shells1(i) = Nr[Iz[i]];
  }
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    fortran_vars.shells2(i) = Nr2[Iz[i]];
  }
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    fortran_vars.rm_base(i) = Rm[Iz[i]];
  }

  /* MO BASIS SET */
  fortran_vars.nucleii = FortranMatrix<uint>(Nuc, fortran_vars.m, 1, 1);
  fortran_vars.contractions = FortranMatrix<uint>(ncont, fortran_vars.m, 1, 1);
  for (uint i = 0; i < fortran_vars.m; i++) {
    if ((fortran_vars.contractions(i) - 1) > MAX_CONTRACTIONS)
      throw runtime_error("Maximum functions per contraction reached!");
  }
  fortran_vars.a_values =
      FortranMatrix<double>(a, fortran_vars.m, MAX_CONTRACTIONS, ngaussians);
  fortran_vars.c_values =
      FortranMatrix<double>(c, fortran_vars.m, MAX_CONTRACTIONS, ngaussians);

  // nco = number of Molecular orbitals ocupped
  fortran_vars.nco = nco;
  fortran_vars.OPEN = OPEN;

  if (verbose > 2) {
     cout << "  QM atoms: " << fortran_vars.atoms;
     cout << " - MM atoms: " << fortran_vars.max_atoms - fortran_vars.atoms << endl;
     cout << "  Total number of basis: " << fortran_vars.gaussians;
     cout << " (s: " << fortran_vars.s_funcs << " p: " << fortran_vars.p_funcs
          << " d: "<< fortran_vars.d_funcs << ")" << endl;
  }

  if (fortran_vars.OPEN) {
    fortran_vars.nunp = nunp;
    if (verbose > 2) cout << "  Open shell calculation - Occupied MO(UP): "
         << fortran_vars.nco << " - Occupied MO(DOWN): "
         << fortran_vars.nco + fortran_vars.nunp << endl;

    fortran_vars.rmm_dens_a = FortranMatrix<double>(
        rhoalpha, fortran_vars.m, fortran_vars.m, fortran_vars.m);
    fortran_vars.rmm_dens_b = FortranMatrix<double>(
        rhobeta, fortran_vars.m, fortran_vars.m, fortran_vars.m);
    fortran_vars.rmm_input_ndens1 = FortranMatrix<double>(
        rho_vec, fortran_vars.m, fortran_vars.m, fortran_vars.m);

    //  Matriz de fock
    fortran_vars.rmm_output_a = FortranMatrix<double>(
        fock_vec,  (fortran_vars.m * (fortran_vars.m + 1) / 2));
    fortran_vars.rmm_output_b = FortranMatrix<double>(
        fockb_vec, (fortran_vars.m * (fortran_vars.m + 1) / 2));
  } else {
    if (verbose > 2) cout << "  Closed shell calculation - Occupied MO: "
         << fortran_vars.nco << endl;
    // matriz densidad
    fortran_vars.rmm_input_ndens1 = FortranMatrix<double>(
        rho_vec, fortran_vars.m, fortran_vars.m, fortran_vars.m);
    // matriz de Fock
    fortran_vars.rmm_output = FortranMatrix<double>(
        fock_vec,  (fortran_vars.m * (fortran_vars.m + 1) / 2));
  }

  fortran_vars.e1 =
      FortranMatrix<double>(e, SMALL_GRID_SIZE, 3, SMALL_GRID_SIZE);
  fortran_vars.e2 =
      FortranMatrix<double>(e2, MEDIUM_GRID_SIZE, 3, MEDIUM_GRID_SIZE);
  fortran_vars.e3 = FortranMatrix<double>(e3, BIG_GRID_SIZE, 3, BIG_GRID_SIZE);
  fortran_vars.wang1 =
      FortranMatrix<double>(wang, SMALL_GRID_SIZE, 1, SMALL_GRID_SIZE);
  fortran_vars.wang2 =
      FortranMatrix<double>(wang2, MEDIUM_GRID_SIZE, 1, MEDIUM_GRID_SIZE);
  fortran_vars.wang3 =
      FortranMatrix<double>(wang3, BIG_GRID_SIZE, 1, BIG_GRID_SIZE);

  fortran_vars.atom_atom_dists =
      HostMatrix<double>(fortran_vars.atoms, fortran_vars.atoms);
  fortran_vars.nearest_neighbor_dists = HostMatrix<double>(fortran_vars.atoms);

  // For Becke partitions.
  fortran_vars.becke = becke;

/** Variables para configurar libxc **/
#if USE_LIBXC
    fortran_vars.use_libxc = use_libxc;
    fortran_vars.ex_functional_id = ex_functional_id;
    fortran_vars.ec_functional_id = ec_functional_id;
    if (fortran_vars.use_libxc) {
        cout << "*Using Libxc" << endl;
    }
#endif

#if GPU_KERNELS
  G2G::gpu_set_variables();

  // Pin the Fortran-allocated RMM density buffer in place so the per-iter
  // host->device upload becomes a fast pinned transfer instead of going
  // through the driver's pageable staging buffer. cudaHostRegister pins
  // existing pages without reallocating; the matching cudaHostUnregister is
  // in g2g_deinit_. Closed-shell only: open-shell density buffers
  // (rmm_dens_a/b) have a smaller Fortran-side allocation than the
  // FortranMatrix m*m dimensions imply and registering past the end caused
  // open-shell SCF to diverge to NaN.
  if (!fortran_vars.OPEN) {
    cudaHostRegister(fortran_vars.rmm_input_ndens1.data,
                     fortran_vars.m * fortran_vars.m * sizeof(double),
                     cudaHostRegisterDefault);
  }
#endif
}
//============================================================================================================
extern "C" void g2g_deinit_(void) {
  if (verbose > 3) cout << "G2G Deinitialisation." << endl;
#if GPU_KERNELS
  if (!fortran_vars.OPEN && fortran_vars.rmm_input_ndens1.data)
    cudaHostUnregister(fortran_vars.rmm_input_ndens1.data);
#endif
  partition.clear();
}
//============================================================================================================
void compute_new_grid(const unsigned int grid_type) {
  switch (grid_type) {
    case 0:
      fortran_vars.grid_type = SMALL_GRID;
      fortran_vars.grid_size = SMALL_GRID_SIZE;
      fortran_vars.e = fortran_vars.e1;
      fortran_vars.wang = fortran_vars.wang1;
      fortran_vars.shells = fortran_vars.shells1;
      for (int i = 0; i < fortran_vars.atoms; i++) {
        fortran_vars.rm(i) = fortran_vars.rm_base(i);
      }
      break;
    case 1:
      fortran_vars.grid_type = MEDIUM_GRID;
      fortran_vars.grid_size = MEDIUM_GRID_SIZE;
      fortran_vars.e = fortran_vars.e2;
      fortran_vars.wang = fortran_vars.wang2;
      fortran_vars.shells = fortran_vars.shells2;
      for (int i = 0; i < fortran_vars.atoms; i++) {
        fortran_vars.rm(i) = fortran_vars.rm_base(i);
      }
      break;
    case 2:
      fortran_vars.grid_type = BIG_GRID;
      fortran_vars.grid_size = BIG_GRID_SIZE;
      fortran_vars.e = fortran_vars.e3;
      fortran_vars.wang = fortran_vars.wang3;
      fortran_vars.shells = fortran_vars.shells2;
      for (int i = 0; i < fortran_vars.atoms; i++) {
        fortran_vars.rm(i) = fortran_vars.rm_base(i);
      }
      break;
    case 3:
      fortran_vars.grid_type = BIG_GRID;
      fortran_vars.grid_size = BIG_GRID_SIZE;
      fortran_vars.e = fortran_vars.e3;
      fortran_vars.wang = fortran_vars.wang3;
      fortran_vars.shells.resize(fortran_vars.atoms);
      for (int i = 0; i < fortran_vars.atoms; i++) {
        fortran_vars.shells(i) = fortran_vars.shells2(i) * 2;
        fortran_vars.rm(i)     = fortran_vars.rm_base(i) * 0.5;
      }
      break;
    case 4:
      fortran_vars.grid_type = BIG_GRID;
      fortran_vars.grid_size = BIG_GRID_SIZE;
      fortran_vars.e = fortran_vars.e3;
      fortran_vars.wang = fortran_vars.wang3;
      fortran_vars.shells.resize(fortran_vars.atoms);
      for (int i = 0; i < fortran_vars.atoms; i++) {
        fortran_vars.shells(i) = fortran_vars.shells2(i) * 4;
        fortran_vars.rm(i)     = fortran_vars.rm_base(i) * 0.25;
      }
      break;
    default:
      throw std::runtime_error("Error de grilla");
      break;
  }

  Timer t_grilla;
  t_grilla.start_and_sync();
  partition.regenerate();
  t_grilla.stop_and_sync();

#if CPU_KERNELS && !CPU_RECOMPUTE
  /** compute functions **/
  // if (fortran_vars.do_forces) cout << "<===== computing all functions
  // [forces] =======>" << endl;
  // else cout << "<===== computing all functions =======>" << endl;

  partition.compute_functions(fortran_vars.do_forces, fortran_vars.gga);

#endif
}
//==============================================================================================================
extern "C" void g2g_reload_atom_positions_(const unsigned int& grid_type, 
                                           unsigned int* atom_Z_in) {
  // IGRID indicates the grid type used.
  // atom_Z is updated in case Becke partition is desired.

  HostMatrixFloat3 atom_positions(fortran_vars.atoms);  // gpu version (float3)
  fortran_vars.atom_positions.resize(
      fortran_vars.atoms);  // cpu version (double3)
  for (uint i = 0; i < fortran_vars.atoms; i++) {
    double3 pos = make_double3(fortran_vars.atom_positions_pointer(i, 0),
                               fortran_vars.atom_positions_pointer(i, 1),
                               fortran_vars.atom_positions_pointer(i, 2));
    fortran_vars.atom_positions(i) = pos;
    fortran_vars.atom_Z(i)         = atom_Z_in[i];
    atom_positions(i) = make_float3(pos.x, pos.y, pos.z);
  }

#if GPU_KERNELS
#if FULL_DOUBLE
  G2G::gpu_set_atom_positions(fortran_vars.atom_positions);
#else
  G2G::gpu_set_atom_positions(atom_positions);
#endif
#endif

  compute_new_grid(grid_type);
}
//==============================================================================================================
extern "C" void g2g_new_grid_(const unsigned int& grid_type) {
  //	cout << "<======= GPU New Grid (" << grid_type << ")========>" << endl;
  if (grid_type == (uint)fortran_vars.grid_type)
    ;
  //		cout << "not loading, same grid as loaded" << endl;
  else
    compute_new_grid(grid_type);
}

template <bool compute_rmm, bool lda, bool compute_forces>
void g2g_iteration(bool compute_energy, double* fort_energy_ptr,
                   double* fort_forces_ptr) {
  Timers timers;
#ifdef _DEBUG 
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
  partition.solve(timers, compute_rmm, lda, compute_forces, compute_energy,
                  fort_energy_ptr, fort_forces_ptr, fortran_vars.OPEN);
#ifdef _DEBUG 
  fedisableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
}

//===============================================================================================================
extern "C" void g2g_solve_groups_(const uint& computation_type,
                                  double* fort_energy_ptr,
                                  double* fort_forces_ptr) {
  // COMPUTE_RMM             0
  // COMPUTE_ENERGY_ONLY     1
  // COMPUTE_ENERGY_FORCE    2
  // COMPUTE_FORCE_ONLY      3

  /*	cout << "<================ iteracion [";
   switch(computation_type) {
     case COMPUTE_RMM: cout << "rmm"; break;
     case COMPUTE_ENERGY_ONLY: cout << "energia"; break;
     case COMPUTE_ENERGY_FORCE: cout << "energia+fuerzas"; break;
     case COMPUTE_FORCE_ONLY: cout << "fuerzas"; break;
   }
         cout << "] ==========>" << endl;
 */
  bool compute_energy = (computation_type == COMPUTE_ENERGY_ONLY ||
                         computation_type == COMPUTE_ENERGY_FORCE);
  bool compute_forces = (computation_type == COMPUTE_FORCE_ONLY ||
                         computation_type == COMPUTE_ENERGY_FORCE);
  bool compute_rmm = (computation_type == COMPUTE_RMM);

  if (energy_all_iterations) compute_energy = true;

  if (compute_rmm) {
    if (fortran_vars.lda) {
      if (compute_forces)
        g2g_iteration<true, true, true>(compute_energy, fort_energy_ptr,
                                        fort_forces_ptr);
      else
        g2g_iteration<true, true, false>(compute_energy, fort_energy_ptr,
                                         fort_forces_ptr);
    } else {
      if (compute_forces)
        g2g_iteration<true, false, true>(compute_energy, fort_energy_ptr,
                                         fort_forces_ptr);
      else {
        g2g_iteration<true, false, false>(compute_energy, fort_energy_ptr,
                                          fort_forces_ptr);
      }
    }
  } else {
    if (fortran_vars.lda) {
      if (compute_forces)
        g2g_iteration<false, true, true>(compute_energy, fort_energy_ptr,
                                         fort_forces_ptr);
      else
        g2g_iteration<false, true, false>(compute_energy, fort_energy_ptr,
                                          fort_forces_ptr);
    } else {
      if (compute_forces)
        g2g_iteration<false, false, true>(compute_energy, fort_energy_ptr,
                                          fort_forces_ptr);
      else
        g2g_iteration<false, false, false>(compute_energy, fort_energy_ptr,
                                           fort_forces_ptr);
    }
  }
}

extern "C" void g2g_get_becke_dens_(double* fort_becke){
  // VERY dirty fix to becke charges...
  double total_dens = 0.0, factor = 1.0;
  int    n_elecs    = fortran_vars.nco*2;
  if (fortran_vars.OPEN) {
    n_elecs += fortran_vars.nunp;
  }
  for (int i = 0; i < fortran_vars.atoms; i++) {
    total_dens += fortran_vars.becke_atom_dens(i);
  }
  if (total_dens > 1E-36) {
    factor = (double) n_elecs / total_dens;
  } else {
    factor = 0.0;
  };  

  for (int i = 0; i < fortran_vars.atoms; i++) {
    fort_becke[i] = fortran_vars.atom_Z(i) - fortran_vars.becke_atom_dens(i) * factor;
  }
}

extern "C" void g2g_get_becke_spin_(double* fort_becke){
  if (!fortran_vars.OPEN) return;

  for (int i = 0; i < fortran_vars.atoms; i++) {
    fort_becke[i] = fortran_vars.becke_atom_spin(i);
  }
}

//===============================================================================================================
// Variant of g2g_solve_groups_ that writes the Fock contribution into a
// caller-supplied buffer instead of fortran_vars.rmm_output.data. Caller is
// responsible for zero-initializing fock_buffer before the call (it is treated
// as a pure accumulation target by partition.solve()'s Kahan merge).
//
// Used to overlap int3lu (CPU BLAS Coulomb fit) with g2g_solve_groups (GPU+CPU
// XC Fock) inside an OpenMP parallel sections block: the XC contribution lands
// in the scratch buffer, and the caller adds it to Fmat after the merge.
extern "C" void g2g_solve_groups_into_(
    const uint& computation_type,
    double* fort_energy_ptr,
    double* fort_forces_ptr,
    double* fock_buffer) {
  double* saved = fortran_vars.rmm_output.data;
  fortran_vars.rmm_output.data = fock_buffer;
  g2g_solve_groups_(computation_type, fort_energy_ptr, fort_forces_ptr);
  fortran_vars.rmm_output.data = saved;
}

// Open-shell variant: rebinds fortran_vars.rmm_output_a/_b to caller-supplied
// alpha/beta scratch buffers for the duration of the call. Both buffers must
// be pre-zeroed; partition.solve() Kahan-merges per-thread XC accumulators
// into them.
extern "C" void g2g_solve_groups_into_open_(
    const uint& computation_type,
    double* fort_energy_ptr,
    double* fort_forces_ptr,
    double* fock_buffer_a,
    double* fock_buffer_b) {
  double* saved_a = fortran_vars.rmm_output_a.data;
  double* saved_b = fortran_vars.rmm_output_b.data;
  fortran_vars.rmm_output_a.data = fock_buffer_a;
  fortran_vars.rmm_output_b.data = fock_buffer_b;
  g2g_solve_groups_(computation_type, fort_energy_ptr, fort_forces_ptr);
  fortran_vars.rmm_output_a.data = saved_a;
  fortran_vars.rmm_output_b.data = saved_b;
}
//================================================================================================================
/* general options */
namespace G2G {
uint max_function_exponent = 10;
double little_cube_size = 8.0;
uint min_points_per_cube = 1;
bool assign_all_functions = false;
double sphere_radius = 0.6;
bool remove_zero_weights = true;
bool energy_all_iterations = false;
double free_global_memory = 0.0;
bool timer_single = false;
bool timer_sum = false;
uint verbose = 0;
}

//=================================================================================================================
extern "C" void g2g_set_options_(double* fort_fgm, double* fort_lcs,
                                 double* fort_sr, bool* fort_aaf,
                                 bool* fort_eai, bool* fort_rzw,
                                 uint& fort_mppc, uint& fort_mfe,
                                 uint& fort_time, uint& fort_verbose) {
  uint timer_level = 0;

  free_global_memory = *fort_fgm;
  little_cube_size = *fort_lcs;
  sphere_radius = *fort_sr;
  assign_all_functions = *fort_aaf;
  energy_all_iterations = *fort_eai;
  remove_zero_weights = *fort_rzw;
  min_points_per_cube = fort_mppc;
  max_function_exponent = fort_mfe;
  timer_level = fort_time;
  verbose = fort_verbose;

  if (timer_level > 1) {
    timer_sum = true;
  }
  if (timer_level % 2 == 1) {
    timer_single = true;
  }
}

//=================================================================================================================
