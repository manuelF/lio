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
contains
subroutine int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, open_shell, memo)
   use basis_data, only: M, Md, cool, cools, kkind, kkinds, kknumd, kknums, &
                         af, MM, MMd

   implicit none
   logical         , intent(in) :: open_shell, memo
   double precision, intent(in) :: rho(:), Gmat(:), Ginv(:), Hmat(:)
   double precision, intent(inout) :: E2, Fmat_b(:), Fmat(:)

   ! Rc: contracted density in the fitting basis, Rc(k) = sum_kk t(k,kk)*rho_kk
   ! aux: temporary for DSPMV result (Gmat * af)
   double precision, allocatable :: Rc(:), aux(:)

   ! Temporaries for BLAS gather/scatter pattern:
   ! rho_gathered: contiguous copy of scattered rho values for DGEMV input
   ! terms_d: DGEMV output (one dot product per basis pair), double precision
   double precision, allocatable :: rho_gathered(:), terms_d(:)

   ! Single-precision temporaries for SGEMV on the cools array:
   ! rho_s: rho values converted to single for SGEMV input
   ! Rc_s: single-precision SGEMV output, accumulated into double Rc
   ! af_s: af converted to single for the Fock update SGEMV
   ! terms_s: single-precision SGEMV output for Fock scatter
   real            , allocatable :: rho_s(:), Rc_s(:), af_s(:), terms_s(:)

   double precision :: Ea, Eb
   integer          :: ll(3), k_ind, kk_ind, m_ind

   ! BLAS function declarations
   double precision, external :: ddot

   allocate(Rc(Md), aux(Md))
   Ea = 0.D0 ; Eb = 0.D0

   MM = M * (M + 1) / 2
   MMd = Md * (Md + 1) / 2

   if (MEMO) then
      call g2g_timer_start('int3lu - start')

      do k_ind = 1, 3
         Ll(k_ind) = k_ind * (k_ind - 1) / 2
      enddo

      !--------------------------------------------------------------------
      ! STEP 1: Rc accumulation
      !   Rc(k) = sum over basis pairs kk of: rho(kkind(kk)) * cool(k, kk)
      !
      !   cool is laid out as a (Md x kknumd) column-major matrix, so this
      !   is a standard matrix-vector product Rc = cool * rho_gathered.
      !   We first gather the scattered rho values into a contiguous array,
      !   then call DGEMV (double) or SGEMV (single).
      !--------------------------------------------------------------------

      ! Double-precision integrals: Rc = cool(Md, kknumd) * rho_gathered
      Rc = 0.0D0
      if (kknumd > 0) then
         allocate(rho_gathered(kknumd))
         do kk_ind = 1, kknumd
            rho_gathered(kk_ind) = rho(kkind(kk_ind))
         enddo
         call dgemv('N', Md, kknumd, 1.0D0, cool, Md, rho_gathered, 1, &
                    0.0D0, Rc, 1)
         deallocate(rho_gathered)
      endif

      ! Single-precision integrals: Rc += cools(Md, kknums) * rho_s
      ! Computed in single precision via SGEMV, then promoted to double.
      ! Precision loss is negligible since cools values are already single.
      if (kknums > 0) then
         allocate(rho_s(kknums), Rc_s(Md))
         do kk_ind = 1, kknums
            rho_s(kk_ind) = real(rho(kkinds(kk_ind)))
         enddo
         call sgemv('N', Md, kknums, 1.0, cools, Md, rho_s, 1, 0.0, Rc_s, 1)
         do k_ind = 1, Md
            Rc(k_ind) = Rc(k_ind) + dble(Rc_s(k_ind))
         enddo
         deallocate(rho_s, Rc_s)
      endif

      !--------------------------------------------------------------------
      ! STEP 2: Fitting coefficients  af = Ginv * Rc
      !
      !   Ginv is a symmetric matrix stored in LAPACK packed lower-triangular
      !   format: Ginv(i + (2*Md - j)*(j-1)/2) = Ginv_full(i, j) for i >= j.
      !   DSPMV('L') performs the symmetric matrix-vector product.
      !--------------------------------------------------------------------
      call dspmv('L', Md, 1.0D0, Ginv, Rc, 1, 0.0D0, af, 1)

      ! Initialize Fock matrix from one-electron integrals
      Fmat(1:MM) = Hmat(1:MM)
      if (open_shell) Fmat_b(1:MM) = Hmat(1:MM)

      !--------------------------------------------------------------------
      ! STEP 3: Two-electron Coulomb energy
      !   Ea = af . Rc          (direct Coulomb)
      !   Eb = af^T * Gmat * af (self-interaction correction)
      !   E2 = Ea - Eb/2
      !
      !   aux is used as temp storage for Gmat * af.
      !--------------------------------------------------------------------
      Ea = ddot(Md, af, 1, Rc, 1)
      call dspmv('L', Md, 1.0D0, Gmat, af, 1, 0.0D0, aux, 1)
      Eb = ddot(Md, af, 1, aux, 1)

      call g2g_timer_stop('int3lu - start')
      call g2g_timer_start('int3lu')

      !--------------------------------------------------------------------
      ! STEP 4: Fock matrix update (Coulomb contribution)
      !   Fmat(kkind(kk)) += sum_k af(k) * cool(k, kk)
      !
      !   This is the transpose of step 1: terms = cool^T * af gives a
      !   dot product per basis pair, then we scatter-add into Fmat.
      !   Note: multiple kkind entries may map to the same Fmat element
      !   (duplicate indices), which is handled correctly by the scatter loop.
      !
      !   For open-shell, both Fmat (alpha) and Fmat_b (beta) receive the
      !   same Coulomb contribution.
      !--------------------------------------------------------------------
      if (open_shell) then
         ! Double-precision Fock update (open-shell)
         if (kknumd > 0) then
            allocate(terms_d(kknumd))
            call dgemv('T', Md, kknumd, 1.0D0, cool, Md, af, 1, &
                       0.0D0, terms_d, 1)
            do kk_ind = 1, kknumd
               Fmat(kkind(kk_ind))   = Fmat(kkind(kk_ind))   + terms_d(kk_ind)
               Fmat_b(kkind(kk_ind)) = Fmat_b(kkind(kk_ind)) + terms_d(kk_ind)
            enddo
            deallocate(terms_d)
         endif

         ! Single-precision Fock update (open-shell)
         ! Convert af to single, SGEMV for dot products, scatter as double.
         if (kknums > 0) then
            allocate(af_s(Md), terms_s(kknums))
            do k_ind = 1, Md
               af_s(k_ind) = real(af(k_ind))
            enddo
            call sgemv('T', Md, kknums, 1.0, cools, Md, af_s, 1, &
                       0.0, terms_s, 1)
            do kk_ind = 1, kknums
               Fmat(kkinds(kk_ind))   = Fmat(kkinds(kk_ind))   + &
                                        dble(terms_s(kk_ind))
               Fmat_b(kkinds(kk_ind)) = Fmat_b(kkinds(kk_ind)) + &
                                        dble(terms_s(kk_ind))
            enddo
            deallocate(af_s, terms_s)
         endif
      else
         ! Double-precision Fock update (closed-shell)
         if (kknumd > 0) then
            allocate(terms_d(kknumd))
            call dgemv('T', Md, kknumd, 1.0D0, cool, Md, af, 1, &
                       0.0D0, terms_d, 1)
            do kk_ind = 1, kknumd
               Fmat(kkind(kk_ind)) = Fmat(kkind(kk_ind)) + terms_d(kk_ind)
            enddo
            deallocate(terms_d)
         endif

         ! Single-precision Fock update (closed-shell)
         if (kknums > 0) then
            allocate(af_s(Md), terms_s(kknums))
            do k_ind = 1, Md
               af_s(k_ind) = real(af(k_ind))
            enddo
            call sgemv('T', Md, kknums, 1.0, cools, Md, af_s, 1, &
                       0.0, terms_s, 1)
            do kk_ind = 1, kknums
               Fmat(kkinds(kk_ind)) = Fmat(kkinds(kk_ind)) + &
                                      dble(terms_s(kk_ind))
            enddo
            deallocate(af_s, terms_s)
         endif
      endif
      call g2g_timer_stop('int3lu')
   else
      ! Non-MEMO path: recompute integrals on the fly via GPU analytic code.
      ! Only the energy computation (Eb) is done here; Ea and af are set
      ! inside aint_coulomb_fock.
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
   deallocate(Rc, aux)
   return
end subroutine int3lu
end module subm_int3lu
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
