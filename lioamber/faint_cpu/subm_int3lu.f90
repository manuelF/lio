!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% INT3LU %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! 2e integral gradients, 3 indexes: density fitting functions and wavefunction.!
! Calculates Coulomb elements for the Fock matrix, and 2e energy.              !
!                                                                              !
! EXTERNAL INPUT: system information.                                          !
!   · rho(MM): packed density matrix.                                          !
!   · Fmat(MM): packed Fock matrix (alpha in open shell).                      !
!   · Fmat_b(MM): packed Fock beta matrix (ignored in closed shell).           !
!   · Gmat(MMd): packed Coulomb G matrix (lower triangular).                   !
!   · Ginv(MMd): packed inverted Coulomb G matrix (lower triangular).          !
!   · Hmat(MM): packed 1e matrix elements.                                     !
!   · open_shell: boolean indicating open-shell calculation.                   !
!   · memo: if .true., use precalculated integrals (cool/cools); otherwise     !
!           recompute via aint_coulomb_fock.                                    !
!                                                                              !
! INTERNAL INPUT (from basis_data module):                                     !
!   · M: number of basis functions.                                            !
!   · Md: number of auxiliary (fitting) basis functions.                        !
!   · cool(Md*kknumd): precalculated 3-center integrals, double precision.     !
!     Laid out as a (Md x kknumd) column-major matrix: element (k, kk) is at   !
!     cool((kk-1)*Md + k). Each column kk holds the Md fitting integrals for   !
!     basis pair kk.                                                           !
!   · cools(Md*kknums): same layout as cool, but single precision.             !
!   · kkind(kknumd): maps double-precision pair index kk to packed rho/Fmat    !
!     position.                                                                !
!   · kkinds(kknums): maps single-precision pair index kk to packed rho/Fmat   !
!     position.                                                                !
!   · kknumd: count of double-precision integral pairs.                        !
!   · kknums: count of single-precision integral pairs.                        !
!   · af(Md): variational fitting coefficients (output, written here).         !
!                                                                              !
! EXTERNAL OUTPUTS:                                                            !
!   · E2: 2e Coulomb energy.                                                   !
!                                                                              !
! ALGORITHM (MEMO path):                                                       !
!   1. Rc accumulation: Rc(k) = sum_kk t(k,kk) * rho(kkind(kk))              !
!      This is a matrix-vector product: Rc = cool * rho_gathered               !
!      (DGEMV for double, SGEMV for single-precision integrals).               !
!   2. Fitting coefficients: af = Ginv * Rc                                    !
!      Packed symmetric matrix-vector product (DSPMV).                         !
!   3. Energy: Ea = af . Rc  (DDOT)                                            !
!              Eb = af^T * Gmat * af  (DSPMV + DDOT)                           !
!              E2 = Ea - Eb/2                                                  !
!   4. Fock update: Fmat(kkind(kk)) += sum_k af(k) * t(k,kk)                  !
!      This is the transpose product: terms = cool^T * af, then scatter-add    !
!      (DGEMV('T') for double, SGEMV('T') for single).                         !
!                                                                              !
! Original and debugged (or supposed to): Dario Estrin Jul/1992                !
! Refactored:                             Federico Pedron Sep/2018             !
! Optimized with BLAS:                    Claude/Manuel Mar/2026               !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module subm_int3lu
   implicit none
   private
   public :: int3lu

   ! Persistent work arrays — allocated once on first MEMO call, reused every
   ! iteration. Avoids 25 × 8 allocate/deallocate pairs per SCF run.
   double precision, allocatable, save :: Rc_w(:), aux_w(:)
   double precision, allocatable, save :: rho_gathered_w(:), terms_d_w(:)
   real            , allocatable, save :: rho_s_w(:), Rc_s_w(:)
   real            , allocatable, save :: af_s_w(:), terms_s_w(:)
   integer, save :: saved_Md = 0, saved_kknumd = 0, saved_kknums = 0

   ! Explicit BLAS interfaces — silences -Warray-temporaries on scalar-literal
   ! arguments (1.0D0, 0.0D0) and -Wimplicit-interface for these calls.
   ! Measured runtime impact on fosfato SCF: within noise (~0%). The win is
   ! compile-time argument-type checking, not throughput.
   interface
      subroutine dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
         character, intent(in) :: trans
         integer, intent(in) :: m, n, lda, incx, incy
         double precision, intent(in) :: alpha, beta
         double precision, intent(in) :: a(lda,*), x(*)
         double precision, intent(inout) :: y(*)
      end subroutine
      subroutine sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
         character, intent(in) :: trans
         integer, intent(in) :: m, n, lda, incx, incy
         real, intent(in) :: alpha, beta
         real, intent(in) :: a(lda,*), x(*)
         real, intent(inout) :: y(*)
      end subroutine
      subroutine dspmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
         character, intent(in) :: uplo
         integer, intent(in) :: n, incx, incy
         double precision, intent(in) :: alpha, beta
         double precision, intent(in) :: ap(*), x(*)
         double precision, intent(inout) :: y(*)
      end subroutine
      double precision function ddot(n, x, incx, y, incy)
         integer, intent(in) :: n, incx, incy
         double precision, intent(in) :: x(*), y(*)
      end function
   end interface

contains
subroutine int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, open_shell, memo)
   use basis_data, only: M, Md, cool, cools, kkind, kkinds, kknumd, kknums, &
                         af, MM, MMd

   implicit none
   logical         , intent(in) :: open_shell, memo
   double precision, intent(in) :: rho(:), Gmat(:), Ginv(:), Hmat(:)
   double precision, intent(inout) :: E2, Fmat_b(:), Fmat(:)

   double precision :: Ea, Eb
   integer          :: ll(3), k_ind, kk_ind, m_ind

   Ea = 0.D0 ; Eb = 0.D0

   MM = M * (M + 1) / 2
   MMd = Md * (Md + 1) / 2

   if (MEMO) then
      call g2g_timer_start('int3lu - start')

      ! Reallocate persistent work arrays only when sizes change (first call
      ! or if basis changes between SCF runs in MD).
      if (Md /= saved_Md .or. kknumd /= saved_kknumd .or. &
          kknums /= saved_kknums) then
         if (allocated(Rc_w))           deallocate(Rc_w)
         if (allocated(aux_w))          deallocate(aux_w)
         if (allocated(rho_gathered_w)) deallocate(rho_gathered_w)
         if (allocated(terms_d_w))      deallocate(terms_d_w)
         if (allocated(rho_s_w))        deallocate(rho_s_w)
         if (allocated(Rc_s_w))         deallocate(Rc_s_w)
         if (allocated(af_s_w))         deallocate(af_s_w)
         if (allocated(terms_s_w))      deallocate(terms_s_w)

         allocate(Rc_w(Md), aux_w(Md))
         if (kknumd > 0) allocate(rho_gathered_w(kknumd), terms_d_w(kknumd))
         if (kknums > 0) allocate(rho_s_w(kknums), Rc_s_w(Md), &
                                  af_s_w(Md), terms_s_w(kknums))
         saved_Md = Md
         saved_kknumd = kknumd
         saved_kknums = kknums
      endif

      do k_ind = 1, 3
         Ll(k_ind) = k_ind * (k_ind - 1) / 2
      enddo

      !--------------------------------------------------------------------
      ! STEP 1: Rc accumulation
      !   Rc(k) = sum over basis pairs kk of: rho(kkind(kk)) * cool(k, kk)
      !--------------------------------------------------------------------

      ! Double-precision integrals: Rc = cool(Md, kknumd) * rho_gathered
      Rc_w = 0.0D0
      if (kknumd > 0) then
         do kk_ind = 1, kknumd
            rho_gathered_w(kk_ind) = rho(kkind(kk_ind))
         enddo
         call dgemv('N', Md, kknumd, 1.0D0, cool, Md, rho_gathered_w, 1, &
                    0.0D0, Rc_w, 1)
      endif

      ! Single-precision integrals: Rc += cools(Md, kknums) * rho_s
      if (kknums > 0) then
         do kk_ind = 1, kknums
            rho_s_w(kk_ind) = real(rho(kkinds(kk_ind)))
         enddo
         call sgemv('N', Md, kknums, 1.0, cools, Md, rho_s_w, 1, &
                    0.0, Rc_s_w, 1)
         do k_ind = 1, Md
            Rc_w(k_ind) = Rc_w(k_ind) + dble(Rc_s_w(k_ind))
         enddo
      endif

      !--------------------------------------------------------------------
      ! STEP 2: Fitting coefficients  af = Ginv * Rc
      !--------------------------------------------------------------------
      call dspmv('L', Md, 1.0D0, Ginv, Rc_w, 1, 0.0D0, af, 1)

      ! Initialize Fock matrix from one-electron integrals
      Fmat(1:MM) = Hmat(1:MM)
      if (open_shell) Fmat_b(1:MM) = Hmat(1:MM)

      !--------------------------------------------------------------------
      ! STEP 3: Two-electron Coulomb energy
      !--------------------------------------------------------------------
      Ea = ddot(Md, af, 1, Rc_w, 1)
      call dspmv('L', Md, 1.0D0, Gmat, af, 1, 0.0D0, aux_w, 1)
      Eb = ddot(Md, af, 1, aux_w, 1)

      call g2g_timer_stop('int3lu - start')
      call g2g_timer_start('int3lu')

      !--------------------------------------------------------------------
      ! STEP 4: Fock matrix update (Coulomb contribution)
      !--------------------------------------------------------------------
      if (open_shell) then
         ! Double-precision Fock update (open-shell)
         if (kknumd > 0) then
            call dgemv('T', Md, kknumd, 1.0D0, cool, Md, af, 1, &
                       0.0D0, terms_d_w, 1)
            do kk_ind = 1, kknumd
               Fmat(kkind(kk_ind))   = Fmat(kkind(kk_ind))   + terms_d_w(kk_ind)
               Fmat_b(kkind(kk_ind)) = Fmat_b(kkind(kk_ind)) + terms_d_w(kk_ind)
            enddo
         endif

         ! Single-precision Fock update (open-shell)
         if (kknums > 0) then
            do k_ind = 1, Md
               af_s_w(k_ind) = real(af(k_ind))
            enddo
            call sgemv('T', Md, kknums, 1.0, cools, Md, af_s_w, 1, &
                       0.0, terms_s_w, 1)
            do kk_ind = 1, kknums
               Fmat(kkinds(kk_ind))   = Fmat(kkinds(kk_ind))   + &
                                        dble(terms_s_w(kk_ind))
               Fmat_b(kkinds(kk_ind)) = Fmat_b(kkinds(kk_ind)) + &
                                        dble(terms_s_w(kk_ind))
            enddo
         endif
      else
         ! Double-precision Fock update (closed-shell)
         if (kknumd > 0) then
            call dgemv('T', Md, kknumd, 1.0D0, cool, Md, af, 1, &
                       0.0D0, terms_d_w, 1)
            do kk_ind = 1, kknumd
               Fmat(kkind(kk_ind)) = Fmat(kkind(kk_ind)) + terms_d_w(kk_ind)
            enddo
         endif

         ! Single-precision Fock update (closed-shell)
         if (kknums > 0) then
            do k_ind = 1, Md
               af_s_w(k_ind) = real(af(k_ind))
            enddo
            call sgemv('T', Md, kknums, 1.0, cools, Md, af_s_w, 1, &
                       0.0, terms_s_w, 1)
            do kk_ind = 1, kknums
               Fmat(kkinds(kk_ind)) = Fmat(kkinds(kk_ind)) + &
                                      dble(terms_s_w(kk_ind))
            enddo
         endif
      endif
      call g2g_timer_stop('int3lu')
   else
      ! Non-MEMO path: recompute integrals on the fly via GPU analytic code.
      do k_ind = 1, MM
         Fmat(k_ind) = Hmat(k_ind)
         if (open_shell) Fmat_b(k_ind) = Hmat(k_ind)
      enddo

      call aint_coulomb_fock(Ea)
      do m_ind = 1, Md
         do k_ind = 1, m_ind
            Eb = Eb + af(k_ind) * af(m_ind) * &
                      Gmat(m_ind + (2*Md-k_ind)*(k_ind-1)/2)
         enddo
         do k_ind = m_ind+1, Md
            Eb = Eb + af(k_ind) * af(m_ind) * &
                      Gmat(k_ind + (2*Md-m_ind)*(m_ind-1)/2)
         enddo
      enddo
   endif

   E2 = Ea - Eb / 2.D0
   return
end subroutine int3lu
end module subm_int3lu
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
