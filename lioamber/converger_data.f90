!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module converger_data

   implicit none

!  Covergens criterion
!  Damping=1, DIIS=2, Hybrid Converg=3
   integer :: conver_criter

   logical :: hagodiis
   integer :: ndiis
   real*8  :: damping_factor

   real*8, allocatable :: fock_damped(:,:,:)
   real*8, allocatable :: bcoef (:,:)
   real*8, allocatable :: fockm (:,:,:,:)
   real*8, allocatable :: FP_PFm (:,:,:,:)
   real*8, allocatable :: EMAT2 (:,:,:)

   ! Circular buffer head pointer per spin (P2 optimization)
   ! head_idx(spin) = physical slot where the newest data was stored (1..ndiis)
   integer :: head_idx(2) = (/0, 0/)

   ! Persistent work arrays (P1 optimization: allocate once in converger_init)
   real*8, allocatable :: fock00_w(:,:), fock_w(:,:), rho_w(:,:)
   real*8, allocatable :: suma_w(:,:)
   real*8, allocatable :: scratch1_w(:,:), scratch2_w(:,:)
   real*8, allocatable :: work_w(:)

   ! Persistent DIIS workspace (avoid per-call allocations)
   real*8, allocatable :: EMAT_w(:,:)
   real*8, allocatable :: sv_w(:)

end module converger_data
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
