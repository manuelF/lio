module converger_subs

   implicit none

contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subroutine converger_init( M_in, ndiis_in, factor_in, do_diis, do_hybrid, OPshell )
   use converger_data, only: fockm, FP_PFm, conver_criter, fock_damped, &
                             hagodiis, damping_factor, bcoef, ndiis, EMAT2, &
                             head_idx, &
                             fock00_w, fock_w, rho_w, suma_w, &
                             scratch1_w, scratch2_w, work_w

   implicit none
   double precision, intent(in) :: factor_in
   integer         , intent(in) :: M_in, ndiis_in
   logical         , intent(in) :: do_diis, do_hybrid, OPshell

   hagodiis       = .false.
   damping_factor = factor_in
   ndiis          = ndiis_in

   if (do_hybrid) then
      conver_criter = 3
   else if (do_diis) then
      conver_criter = 2
   else
      conver_criter = 1
   endif

   ! Reset circular buffer heads
   head_idx = 0

   ! Added to change from damping to DIIS. - Nick
   if (conver_criter /= 1) then
      if(OPshell) then
         if (.not. allocated(fockm)  ) allocate(fockm (M_in, M_in, ndiis, 2))
         if (.not. allocated(FP_PFm) ) allocate(FP_PFm(M_in, M_in, ndiis, 2))
         if (.not. allocated(bcoef)  ) allocate(bcoef(ndiis+1, 2) )
         if (.not.allocated(EMAT2)   ) allocate(EMAT2(ndiis,ndiis,2))
      else
         if (.not. allocated(fockm) )  allocate(fockm (M_in, M_in, ndiis, 1))
         if (.not. allocated(FP_PFm) ) allocate(FP_PFm(M_in, M_in, ndiis, 1))
         if (.not. allocated(bcoef) )  allocate(bcoef (ndiis+1, 1))
         if (.not.allocated(EMAT2) )   allocate(EMAT2(ndiis,ndiis,1))
      end if
      fockm   = 0.0D0
      FP_PFm  = 0.0D0
      bcoef   = 0.0D0
      EMAT2   = 0.0D0
   endif

   if(OPshell) then
      if (.not. allocated(fock_damped) ) allocate(fock_damped(M_in, M_in, 2))
   else
      if (.not. allocated(fock_damped) ) allocate(fock_damped(M_in, M_in, 1))
   end if
   fock_damped(:,:,:) = 0.0D0

   ! Persistent work arrays (allocated once, reused every conver call)
   if (.not. allocated(fock00_w))   allocate(fock00_w(M_in, M_in))
   if (.not. allocated(fock_w))     allocate(fock_w(M_in, M_in))
   if (.not. allocated(rho_w))      allocate(rho_w(M_in, M_in))
   if (.not. allocated(work_w))     allocate(work_w(1000))
   if (conver_criter /= 1) then
      if (.not. allocated(suma_w))     allocate(suma_w(M_in, M_in))
      if (.not. allocated(scratch1_w)) allocate(scratch1_w(M_in, M_in))
      if (.not. allocated(scratch2_w)) allocate(scratch2_w(M_in, M_in))
   endif
end subroutine converger_init

   subroutine conver (niter, good, good_cut, M_in, rho_op, fock_op, &
#ifdef CUBLAS
                      devPtrX, devPtrY, spin)
#else
                      Xmat, Ymat, spin)
#endif
   use converger_data  , only: damping_factor, hagodiis, fockm, FP_PFm, ndiis, &
                               fock_damped, bcoef, EMAT2, conver_criter, &
                               head_idx, &
                               fock00_w, fock_w, rho_w, suma_w, &
                               scratch1_w, scratch2_w, work_w
   use typedef_operator, only: operator
   use fileio_data     , only: verbose

   implicit none
   ! Spin allows to store correctly alpha or beta information. - Carlos
   integer         , intent(in)    :: niter, M_in, spin
   double precision, intent(in)    :: good, good_cut
   type(operator)  , intent(inout) :: rho_op, fock_op

#ifdef  CUBLAS
   integer*8       , intent(in) :: devPtrX, devPtrY
#else
   double precision, intent(in) :: Xmat(M_in,M_in), Ymat(M_in,M_in)
#endif

   integer          :: ndiist, ii, jj, kk, lwork, info
   integer          :: slot_i, slot_j, slot_k
   integer          :: diis_rank
   double precision, allocatable :: EMAT(:,:)
   double precision, allocatable :: sv(:)
   double precision :: rcond_diis, bcoef_max


! INITIALIZATION
! If DIIS is turned on, update fockm with the current transformed F' (into ON
! basis) and update FP_PFm with the current transformed [F',P']
!
! (1)     Calculate F' and [F',P']
!       update fockm with F'
! now, scratch1 = A = F' * P'; scratch2 = A^T
! [F',P'] = A - A^T
! BASE CHANGE HAPPENS INSIDE OF FOCK_COMMUTS

   fock00_w = 0.0D0
   fock_w   = 0.0D0
   rho_w    = 0.0D0

   ! Saving rho and the first fock AO
   call rho_op%Gets_data_AO(rho_w)
   call fock_op%Gets_data_AO(fock00_w)

   ndiist = min( niter, ndiis )
   if (conver_criter /= 1) then
      suma_w = 0.0D0
      scratch1_w = 0.0D0
      scratch2_w = 0.0D0


! If DIIS is turned on, update fockm with the current transformed F' (into ON
! basis) and update FP_PFm with the current transformed [F',P']

      ! P2: Circular buffer — O(1) advance instead of O(ndiis*M^2) shift
      head_idx(spin) = mod(head_idx(spin), ndiis) + 1

#ifdef CUBLAS
      call rho_op%BChange_AOtoON(devPtrY, M_in, 'r')
      call fock_op%BChange_AOtoON(devPtrX, M_in, 'r')
#else
      call rho_op%BChange_AOtoON(Ymat, M_in, 'r')
      call fock_op%BChange_AOtoON(Xmat,M_in, 'r')
#endif
      call rho_op%Gets_data_ON(rho_w)
      call fock_op%Commut_data_r(rho_w, scratch1_w, M_in)

      FP_PFm(:,:,head_idx(spin),spin) = scratch1_w(:,:)
      call fock_op%Gets_data_ON( fockm(:,:,head_idx(spin),spin) )

   endif

   select case (conver_criter)
      ! Always do damping
      case (1)
         hagodiis = .false.

      ! Damping the first two steps, diis afterwards
      case (2)
         if (niter > 2) then
            hagodiis = .true.
         else
            hagodiis = .false.
         endif

      ! Damping until good enough, diis afterwards
      case(3)
         if ((good < good_cut) .and. (niter > 2)) then
            if ( (.not. hagodiis) .and. (verbose .gt. 3) ) &
                     write(6,'(A,I4)') "  Changing to DIIS at step: ", niter
            hagodiis=.true.
         endif

      case default
         write(*,'(A,I4)') 'ERROR - Wrong conver_criter = ', conver_criter
         stop
   endselect

   ! THIS IS DAMPING
   ! If we are not doing diis this iteration, apply damping to F, save this
   ! F in fock_damped for next iteration's damping and put F' = X^T * F * X in
   ! fock the newly constructed damped matrix is stored, for next iteration in
   ! fock_damped
   if (.not. hagodiis) then
      fock_w = fock00_w

      if (niter > 1) &
         fock_w = (fock_w  + damping_factor * fock_damped(:,:,spin)) / &
                  (1.0D0 + damping_factor)
      fock_damped(:,:,spin) = fock_w
      call fock_op%Sets_data_AO(fock_w)

#ifdef  CUBLAS
      call fock_op%BChange_AOtoON(devPtrX, M_in, 'r')
#else
      call fock_op%BChange_AOtoON(Xmat   , M_in, 'r')
#endif
   endif

   ! DIIS
   if (conver_criter /= 1) then
      allocate(EMAT(ndiist+1,ndiist+1))

      ! Read cached EMAT2 entries using circular buffer physical slot mapping.
      ! Unified logic for both niter <= ndiis and niter > ndiis cases.
      EMAT = 0.0D0
      if (niter .gt. 1) then
         do jj = 1, ndiist-1
            slot_j = circ_slot(jj, head_idx(spin), ndiist, ndiis)
         do ii = 1, ndiist-1
            slot_i = circ_slot(ii, head_idx(spin), ndiist, ndiis)
            EMAT(ii,jj) = EMAT2(slot_i, slot_j, spin)
         enddo
         enddo
      endif

      ! Compute newest row/column of EMAT (the head entry vs all entries).
      do kk = 1, ndiist
         slot_k = circ_slot(kk, head_idx(spin), ndiist, ndiis)
         scratch1_w(:,:) = FP_PFm(:,:,head_idx(spin),spin)
         scratch2_w(:,:) = FP_PFm(:,:,slot_k,spin)

         EMAT(ndiist,kk) = trace_product(scratch1_w, scratch2_w, M_in)
         if (kk.ne.ndiist) EMAT(kk,ndiist) = EMAT(ndiist,kk)
      enddo

      ! Lagrange multiplier row/column
      do kk = 1, ndiist
         EMAT(kk,ndiist+1) = -1.0d0
         EMAT(ndiist+1,kk) = -1.0d0
      enddo
      EMAT(ndiist+1, ndiist+1)= 0.0d0

      ! Save EMAT entries to EMAT2 using physical slot indices
      do jj = 1, ndiist
         slot_j = circ_slot(jj, head_idx(spin), ndiist, ndiis)
      do ii = 1, ndiist
         slot_i = circ_slot(ii, head_idx(spin), ndiist, ndiis)
         EMAT2(slot_i, slot_j, spin) = EMAT(ii,jj)
      enddo
      enddo

      !   THE MATRIX EMAT SHOULD HAVE THE FOLLOWING SHAPE:
      !      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
      !      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
      !      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
      !      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
      !      |     .            .      ...     . |
      !      |   -1.0         -1.0     ...    0. |
      !   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
      !   TIMES [F*P] FOR ITERATION J.

      if (hagodiis) then
         do ii = 1, ndiist
            bcoef(ii,spin) = 0.0d0
         enddo
         bcoef(ndiist+1,spin) = -1.0d0

         ! Use DGELSS (SVD-based least-squares) instead of DGELS (QR).
         ! DGELSS detects rank deficiency in nearly-singular EMAT and produces
         ! a minimum-norm solution, naturally limiting coefficient magnitudes.
         ! Near SCF convergence, error vectors [F',P'] become nearly parallel,
         ! making EMAT nearly singular. DGELS ignores this and produces wild
         ! coefficients (|c_k| ~ 1e6+), amplifying float32 GPU noise. DGELSS
         ! truncates near-zero singular values and returns bounded coefficients.
         allocate(sv(ndiist+1))
         rcond_diis = -1.0d0  ! Use machine epsilon as rank threshold

         LWORK = -1
         CALL DGELSS( ndiist+1, ndiist+1, 1, EMAT, ndiist+1, &
                      bcoef(:,spin), ndiist+1, sv, rcond_diis, &
                      diis_rank, work_w, LWORK, INFO )

         LWORK = MIN( 1000, INT( work_w( 1 ) ) )
         CALL DGELSS( ndiist+1, ndiist+1, 1, EMAT, ndiist+1, &
                      bcoef(:,spin), ndiist+1, sv, rcond_diis, &
                      diis_rank, work_w, LWORK, INFO )
         deallocate(sv)

         ! Safety check: if coefficients are still too large despite SVD
         ! regularization, fall back to using only the current Fock (newest).
         bcoef_max = 0.0d0
         do kk = 1, ndiist
            if (abs(bcoef(kk,spin)) > bcoef_max) &
               bcoef_max = abs(bcoef(kk,spin))
         enddo

         if (bcoef_max > 1.0d4 .or. INFO /= 0) then
            if (verbose > 3) then
               if (INFO /= 0) then
                  write(6,'(A,I4,A,I4)') &
                     '  DIIS: DGELSS failed (INFO=', INFO, &
                     '), using current Fock at iter ', niter
               else
                  write(6,'(A,ES10.2,A,I4)') &
                     '  DIIS: bcoef too large (max=', bcoef_max, &
                     '), using current Fock at iter ', niter
               endif
            endif
            ! Fall back: use only the newest Fock (no extrapolation)
            do kk = 1, ndiist
               bcoef(kk,spin) = 0.0d0
            enddo
            bcoef(ndiist,spin) = 1.0d0
         endif

         ! Build new Fock as a linear combination of previous steps.
         suma_w = 0.0D0
         do kk=1,ndiist
            slot_k = circ_slot(kk, head_idx(spin), ndiist, ndiis)
            do ii = 1, M_in
            do jj = 1, M_in
               suma_w(ii,jj) = suma_w(ii,jj) + bcoef(kk,spin) * &
                                                fockm(ii,jj,slot_k,spin)
            enddo
            enddo
         enddo
         fock_w = suma_w
         call fock_op%Sets_data_ON(fock_w)

      endif
   endif
end subroutine conver

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Computes Tr(A * B) directly without forming the full product matrix.
! Equivalent to: call matmuldiag(A, B, C, M); trace = sum(C(i,i), i=1..M)
double precision function trace_product(A, B, M)
   implicit none
   integer, intent(in) :: M
   real*8,  intent(in) :: A(M,M), B(M,M)
   integer :: i, k

   trace_product = 0.0d0
   do k = 1, M
   do i = 1, M
      trace_product = trace_product + A(i,k) * B(k,i)
   enddo
   enddo
end function trace_product

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! Maps logical DIIS index k (1=oldest, ndiist=newest) to physical buffer slot.
! head = physical slot where newest data is stored.
integer function circ_slot(k, head, ndiist_in, ndiis_in)
   implicit none
   integer, intent(in) :: k, head, ndiist_in, ndiis_in
   circ_slot = mod(head - ndiist_in + k - 1 + ndiis_in, ndiis_in) + 1
end function circ_slot

end module converger_subs
