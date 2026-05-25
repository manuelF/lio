#include "datatypes/datatypes.fh"
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module converger_data

   implicit none

!  Covergence methods and criteria, as per input file.
!  Fock damping = 1, DIIS = 2, Hybrid convergence = 3, Biased DIIS = 4
!  Biased DIIS + Hybrid convergence = 5
   integer      :: conver_method  = 2
   LIODBLE :: gOld           = 10.0D0
   LIODBLE :: damping_factor = 10.0D0

   ! DIIS and biased DIIS.
   integer      :: nDIIS          = 15
   logical      :: DIIS           = .true.
   LIODBLE :: DIIS_bias      = 1.05D0

   ! Hybrid convergence
   logical      :: hybrid_converg = .false.
   LIODBLE :: good_cut       = 1.0D-3

   ! Level shifting
   logical      :: level_shift    = .false.
   LIODBLE :: lvl_shift_en   = 0.25D0
   LIODBLE :: lvl_shift_cut  = 0.005D0

   ! DIIS error cut for each convergence strategy:
   LIODBLE :: EDIIS_start    = 1D-20
   LIODBLE :: DIIS_start     = 0.01D0
   LIODBLE :: bDIIS_start    = 1D-3

   ! Tolerace for SCF convergence
   integer      :: nMax           = 100
   LIODBLE :: tolD           = 1.0D-6
   LIODBLE :: EtolD          = 1.0D-1

   ! Options for linear search. Rho_LS =1 activates
   ! linear search after failed convergence, =2 means
   ! only attempt linear search.
   integer      :: Rho_LS = 0

   ! Internal variables
   LIODBLE, allocatable :: fock_damped(:,:,:)
   LIODBLE              :: rho_diff      = 1.0D0
   LIODBLE              :: DIIS_error    = 100.0D0
   logical                   :: DIIS_started  = .false.
   logical                   :: EDIIS_started = .false.
   logical                   :: bDIIS_started = .false.

   ! Internal variables for DIIS (and variants)
   LIODBLE, allocatable :: fockm(:,:,:,:)
   LIODBLE, allocatable :: FP_PFm(:,:,:,:)
   LIODBLE, allocatable :: bcoef(:)
   LIODBLE, allocatable :: EMAT(:,:)
   LIODBLE, allocatable :: energy_list(:)

   ! Circular-buffer head for fockm / FP_PFm. The newest entry sits at slot
   ! diis_head; subsequent older entries are at (diis_head - k) mod nDIIS,
   ! eliminating the explicit shift loop that previously ran per iteration.
   ! diis_head is advanced once per SCF iteration (alpha leg of the open-shell
   ! case). Use diis_slot_index() in converger_diis.f90 to translate the legacy
   ! "jj in [nDIIS-nDIIST+1, nDIIS]" indices into circular slot numbers.
   integer :: diis_head = 0

   ! Energy-rejection DIIS rollback. When a DIIS extrapolation produces a
   ! catastrophic upward energy jump (e.g., open-shell transition metals like
   ! heme: +26 Eh at iter 4), the next iter falls back to damping instead of
   ! DIIS. This lets the trajectory recover before resuming DIIS. The bad fockm
   ! entry stays in the subspace but is naturally de-weighted by its large
   ! commutator. Threshold is conservative (1 Eh) so it never fires in a well-
   ! behaved SCF.
   LIODBLE :: scf_prev_energy_rise   = 0.0D0
   logical :: scf_prev_was_diis      = .false.
   LIODBLE :: scf_rollback_threshold = 10.0D0
   ! Only fire rollback when the system is close enough to converged that
   ! DIIS *should* have produced a near-stationary step. rho_diff at the
   ! previous iter being small (e.g., < 0.1) means we're in the smooth
   ! regime where a sudden +ΔE > threshold is a real DIIS failure rather
   ! than normal early-SCF turbulence (heme catastrophe at iter 4: rho_diff
   ! ~ 0.02; fosfato no-restart wild iters: rho_diff > 0.4). This single
   ! gate cleanly separates the two cases without per-system tuning.
   LIODBLE :: scf_rollback_rho_gate  = 0.1D0

   ! Internal variables for EDIIS
   integer                   :: nediis          = 15
   logical                   :: EDIIS_not_ADIIS = .true.
   LIODBLE, allocatable :: ediis_fock(:,:,:,:)
   LIODBLE, allocatable :: ediis_dens(:,:,:,:)
   LIODBLE, allocatable :: BMAT(:,:)
   LIODBLE, allocatable :: EDIIS_E(:)
   LIODBLE, allocatable :: EDIIS_coef(:)

   ! Internal variables for Linear Search
   logical                   :: first_call = .true.
   LIODBLE              :: Elast      = 1000.0D0
   LIODBLE              :: Pstepsize  = 1.0D0
   LIODBLE, allocatable :: rho_lambda1(:)
   LIODBLE, allocatable :: rho_lambda0(:)
   LIODBLE, allocatable :: rhoa_lambda1(:)
   LIODBLE, allocatable :: rhoa_lambda0(:)
   LIODBLE, allocatable :: rhob_lambda1(:)
   LIODBLE, allocatable :: rhob_lambda0(:)

end module converger_data
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
