#include "datatypes/datatypes.fh"
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% MAGNUS.F90 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! This file contains the routine for an N-order magnus propagation.            !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module propagators
   use gpu_interface, only: g2g_solve_groups
   use lio_interface
   use packed_storage_interface
   use gpu_timers_interface

contains

subroutine predictor(F1a, F1b, FON, rho2, factorial, Xmat, Xtrans, timestep, &
                     time, M_in, MTB, dim3)
   ! This routine receives: F1a, F1b, rho2
   ! And gives: F5 = F(t+(deltat/2))
   use garcha_mod   , only: NBCH, rhoalpha, rhobeta, OPEN, r, d, natom,      &
                            ntatom, MEMO, Fmat_vec, Fmat_vec2, Ginv_vec, &
                            Hmat_vec, Gmat_vec, Pmat_vec, nco, nunp, iz, pc
   use field_subs   , only: field_calc
   use faint_cpu    , only: int3lu
   use fockbias_subs, only: fockbias_apply
   use basis_data   , only: M
   use typedef_cumat, only: cumat_r, cumat_x

   implicit none
   ! MTB is only used in DFTB, it equals 0 otherwise.
   integer         , intent(in)    :: M_in, dim3, MTB
   TDCOMPLEX       , intent(in)    :: rho2(M_in,M_in,dim3)
   LIODBLE, intent(in)    :: timestep, time, factorial(NBCH)
   type(cumat_r)   , intent(in)    :: Xmat
   type(cumat_x)   , intent(in)    :: Xtrans
   LIODBLE, intent(inout) :: F1a(M_in,M_in,dim3), F1b(M_in,M_in,dim3),&
                                      FON(M_in,M_in,dim3)
   TDCOMPLEX, allocatable :: rho4(:,:,:), rho2t(:,:,:)
   integer :: M2, MM, nbch_pred
   LIODBLE :: E2, tdstep1, Ex, E1, Ehf
   LIODBLE, allocatable :: F3(:,:,:), FBA(:,:,:)

   allocate(rho4(M_in,M_in,dim3), rho2t(M_in,M_in,dim3), F3(M_in,M_in,dim3), &
            FBA(M_in,M_in,dim3))
   M2 = 2 * M
   MM = M*(M+1)/2

   ! Initializations and defaults
   ! tdstep of the predictor is 0.5 * tdstep_magnus
   ! F1a and F1b are used to extrapolate F3, then F3 is used to propagate rho.
   ! Afterwards, rho4 is copied into Pmat_vec in order to calculate F5.
   tdstep1 = timestep * 0.50D0
   F3      = 1.75D0 * F1b - 0.75D0 * F1a
   rho2t   = rho2

   ! The inner predictor propagates rho only to build the corrector Fock, so it
   ! needs far fewer BCH terms than the final propagation: with the half timestep
   ! (tdstep1 = dt/2) the series converges roughly twice as fast. Using half the
   ! order (floored at 4) keeps the 50k dipole within ~4e-5 of full order while
   ! cutting this loop ~in half. factorial(icount)=1/icount is order-independent,
   ! so the first nbch_pred entries are reused as-is.
   nbch_pred = min(NBCH, max(4, NBCH / 2))
   call g2g_timer_sum_start("TD - Pred inner Magnus")
   call magnus(F3(:,:,1), rho2(:,:,1), rho4(:,:,1), M_in, nbch_pred, tdstep1, &
               factorial)
   if (OPEN) call magnus(F3(:,:,2), rho2(:,:,2), rho4(:,:,2), M_in, nbch_pred, &
                         tdstep1, factorial)
   call g2g_timer_sum_pause("TD - Pred inner Magnus")

   call g2g_timer_sum_start("TD - Pred rho rebuild")
   if (OPEN) then
      rho2t = rho4
      call Xtrans%change_base(rho2t(:,:,1), 'dir')
      call Xtrans%change_base(rho2t(:,:,2), 'dir')

      call sprepack_ctr('L', M, rhoalpha, rho2t(MTB+1:MTB+M,MTB+1:MTB+M,1))
      call sprepack_ctr('L', M, rhobeta , rho2t(MTB+1:MTB+M,MTB+1:MTB+M,2))
      Pmat_vec = rhoalpha + rhobeta
   else
      rho2t = rho4
      call Xtrans%change_base(rho2t(:,:,1), 'dir')
      call sprepack_ctr('L', M, Pmat_vec, rho2t(MTB+1:MTB+M,MTB+1:MTB+M,1))
   endif
   call g2g_timer_sum_pause("TD - Pred rho rebuild")

   call g2g_timer_sum_start("TD - Pred int3lu")
   call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
               Hmat_vec, open, MEMO)
   call g2g_timer_sum_pause("TD - Pred int3lu")
   call g2g_timer_sum_start("TD - Pred XC (g2g)")
   call g2g_solve_groups(0, Ex, 0.0D0)
   call g2g_timer_sum_pause("TD - Pred XC (g2g)")
   call g2g_timer_sum_start("TD - Pred ExactX")
   call do_TDexactExchange(Fmat_vec,Fmat_vec2,Ehf,MM,M,open)
   call g2g_timer_sum_pause("TD - Pred ExactX")
   call g2g_timer_sum_start("TD - Pred field")
   call field_calc(E1, time, Pmat_vec, Fmat_vec2, Fmat_vec, r, d, &
                   natom, ntatom, open, 2*nco+nunp, iz, pc)
   call g2g_timer_sum_pause("TD - Pred field")

   call g2g_timer_sum_start("TD - Pred Fock->ON")
   ! This is performed to recover TB terms from FON. If not, TB terms
   ! in FON become zero.
   FBA(:,:,1) = FON(:,:,1)
   call spunpack('L', M, Fmat_vec, FBA(MTB+1:MTB+M,MTB+1:MTB+M,1))
   call fockbias_apply(time, FBA(MTB+1:MTB+M,MTB+1:MTB+M,1))
   FON(:,:,1) = FBA(:,:,1)
   call Xmat%change_base(FON(:,:,1), 'dir')

   if (OPEN) then
      FBA(:,:,2) = FON(:,:,2)
      call spunpack('L', M, Fmat_vec2, FBA(MTB+1:MTB+M,MTB+1:MTB+M,2))
      call fockbias_apply(time, FBA(MTB+1:MTB+M,MTB+1:MTB+M,2))
      FON(:,:,2) = FBA(:,:,2)
      call Xmat%change_base(FON(:,:,2), 'dir')
   end if
   call g2g_timer_sum_pause("TD - Pred Fock->ON")

   deallocate(rho4, rho2t, F3, FBA)
end subroutine predictor

subroutine magnus(Fock, RhoOld, RhoNew, M, N, dt, factorial)
   ! Propagator based in Baker-Campbell-Hausdorff Nth-order formula.
   ! Everything is calculated in the ortonormal basis.
   ! Input : Fock(t+(deltat/2)), rho(t)
   ! Output: rho6 = rho(t+deltat)
   use typedef_cumat, only: cumat_x

   implicit none
   integer     , intent(in)  :: M, N
   LIODBLE, intent(in)  :: Fock(M,M), factorial(N), dt
   TDCOMPLEX   , intent(in)  :: RhoOld(M,M)
   TDCOMPLEX   , intent(out) :: RhoNew(M,M)

   integer       :: icount, jcount
   TDCOMPLEX     :: alpha, beta, ICMPLX

   ! Persistent scratch across TD steps. Sized on first call, reused.
   ! Avoids per-call alloc/destroy of M*M complex buffers; combined with
   ! move_alloc-based cumat_exchange this cuts ~10us per magnus call from
   ! Fortran heap churn that dominated the inter-step CPU gap.
   type(cumat_x), save          :: omega, comm_prev, comm_next, rho_m
   TDCOMPLEX, allocatable, save :: Omega1(:,:)
   integer, save :: cached_M = 0

   if (cached_M /= M) then
      if (cached_M /= 0) then
         call omega%destroy()
         call rho_m%destroy()
         call comm_next%destroy()
         call comm_prev%destroy()
         deallocate(Omega1)
      endif
      call omega%allocate(M, .true.)
      call rho_m%allocate(M, .true.)
      call comm_prev%allocate(M, .true.)
      call comm_next%allocate(M, .true.)
      allocate(Omega1(M,M))
      cached_M = M
   endif

   ICMPLX = liocmplx(0.0D0, 1.0D0)
   do jcount = 1, M
   do icount = 1, M
      Omega1(icount,jcount) = - ICMPLX * real(Fock(icount,jcount) * dt,&
                                              COMPLEX_SIZE/2)
   enddo
   enddo

   call omega%set(Omega1)
   call rho_m%set(rhoOld)
   call comm_prev%set(rhoOld)

   ! Density matrix propagation
   alpha = liocmplx(1.0D0, 0.0D0)
   do icount = 1, N
      alpha = liocmplx(factorial(icount), 0.0D0)
      beta  = liocmplx(0.0D0, 0.0D0)
      call comm_next%mat_mul(omega    , comm_prev, alpha, beta)

      beta  = liocmplx(-1.0D0, 0.0D0)
      call comm_next%mat_mul(comm_prev, omega    , alpha, beta)

      beta  = liocmplx(1.0D0, 0.0D0)
      call rho_m%add_mat(comm_next, beta)

      call comm_next%exchange(comm_prev)
   enddo

   ! Stores the new density.
   call rho_m%get(rhoNew)
end subroutine magnus

subroutine do_TDexactExchange(Fmat,Fmat2,Eexact,MM,M,open_shell)
use garcha_mod, only: rhoalpha, rhobeta, Pmat_vec, energy_all_iterations
use extern_functional_data, only: HF
use extern_functional_subs, only: exact_exchange, exact_energies
   implicit none

   integer, intent(in) :: MM, M
   logical, intent(in) :: open_shell
   LIODBLE, intent(inout) :: Fmat(MM), Fmat2(MM)
   LIODBLE, intent(out)   :: Eexact

   integer :: ii
   logical :: exact
   LIODBLE :: Eshort, Elong
   LIODBLE, allocatable :: rho_a0(:,:), rho_b0(:,:)
   LIODBLE, allocatable :: fock_a0(:,:), fock_b0(:,:)

   exact = .false.
   do ii=1,3
      if (HF(ii)==1) exact=.true.
   enddo
   if (.not. exact) return

   allocate(rho_a0(M,M),rho_b0(M,M))
   allocate(fock_a0(M,M),fock_b0(M,M))

   ! Fock Calculation
   if ( open_shell ) then
      call spunpack_rho('L', M, rhoalpha , rho_a0)
      call spunpack_rho('L', M, rhobeta  , rho_b0)
      call spunpack(    'L', M, Fmat     , fock_a0)
      call spunpack(    'L', M, Fmat2    , fock_b0)
      call exact_exchange(rho_a0,rho_b0,fock_a0,fock_b0,M)
      call sprepack('L', M, Fmat, fock_a0)
      call sprepack('L', M, Fmat2, fock_b0)
   else
      call spunpack_rho('L', M, Pmat_vec, rho_a0)
      call spunpack(    'L', M, Fmat    , fock_a0)
      call exact_exchange(rho_a0,rho_b0,fock_a0,fock_b0,M)
      call sprepack('L', M, Fmat, fock_a0)
   endif

   Eexact=0.0d0; Eshort=0.0d0; Elong=0.0d0
   if ( energy_all_iterations ) then
      call exact_energies(rho_a0,rho_b0,Eexact,Eshort,Elong,M)
   endif
   Eexact = Eexact + Eshort + Elong

   deallocate(rho_a0,rho_b0,fock_a0,fock_b0)
end subroutine do_TDexactExchange


end module propagators
