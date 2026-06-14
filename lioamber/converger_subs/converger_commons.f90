
!#############################################################################!
! These subroutines are called from SCF in order to perform convergence       !
! acceleration (mainly via DIIS).                                             !
!                                                                             !
! Externally called subroutines (from SCF):                                   !
!   * converger_options_check                                                 !
!   * converger_init                                                          !
!   * converger_finalise                                                      !
!   * converger_setup                                                         !
!   * converger_fock                                                          !
!   * converger_check                                                         !
!                                                                             !
! Internal subroutines:                                                       !
!   * select_methods                                                          !
!                                                                             !
!#############################################################################!
subroutine converger_options_check(energ_all_iter)
   ! Checks input options so that there are not unfortunate clashes.
   use converger_data, only: damping_factor, gOld, DIIS, hybrid_converg, &
                             conver_method, rho_LS

   logical, intent(inout) :: energ_all_iter

   if (abs(gOld - 10.0D0) > 1e-12) then
      damping_factor = gOld
   else if (abs(damping_factor - 10.0D0) > 1e-12) then
      gOld = damping_factor
   endif

   if (conver_method == 2) then
      if (hybrid_converg) then
         diis          = .true.
         conver_method = 3
      else if (diis) then
         conver_method = 2
      else
         conver_method = 1
      endif
   else if (conver_method /= 1) then
      diis = .true.
   endif

   if ((Rho_LS > 1) .and. (conver_method /=1)) then
      hybrid_converg = .false.
      DIIS           = .false.
      conver_method  = 1
      write(*,'(A)') &
      '  WARNING - Turning to damping-only convergence for linear search.'
    end if

   ! For biased DIIS, full energy is needed.
   if ((conver_method == 4) .or. (conver_method == 5)) energ_all_iter = .true.
end subroutine converger_options_check

subroutine converger_init( M_in, OPshell )
   ! Initialises and allocates matrices.
   use converger_data, only: fock_damped, told, etold, scf_prev_was_diis, &
                             scf_prev_energy_rise
   implicit none
   integer         , intent(in) :: M_in
   logical         , intent(in) :: OPshell

   write(*,'(2x,A)') "Convergence criteria are: "
   write(*,'(2x,ES9.2,A33,ES9.2,A26)')                 &
            told, " in Rho mean squared difference, ", &
            Etold, " Eh in energy differences."

   if (OPshell) then
      if (.not. allocated(fock_damped) ) allocate(fock_damped(M_in, M_in, 2))
   else
      if (.not. allocated(fock_damped) ) allocate(fock_damped(M_in, M_in, 1))
   end if
   fock_damped(:,:,:) = 0.0D0

   scf_prev_was_diis    = .false.
   scf_prev_energy_rise = 0.0D0

   call diis_init(M_in, OPshell)
   call ediis_init(M_in, OPshell)
end subroutine converger_init

subroutine converger_finalise()
   ! Deallocates matrices for the convergence acceleration module.
   use converger_data, only: fock_damped
   implicit none

   if (allocated(fock_damped)) deallocate(fock_damped)

   call diis_finalise()
   call ediis_finalise()
   call rho_ls_finalise()
end subroutine converger_finalise

subroutine converger_setup(niter, M_in, dens_op, fock_op, energy, &
                           Xmat, Ymat, dens_opb, fock_opb)

   ! Sets up matrices needed for convergence acceleration, such as
   ! Emat for DIIS or Bmat for EDIIS
   use converger_data  , only: conver_method, ndiis, nediis, dens_bchange_done
   use fileio_data     , only: verbose
   use typedef_operator, only: operator
   use typedef_cumat   , only: cumat_r

   implicit none
   integer        , intent(in)              :: niter, M_in
   type(cumat_r)  , intent(in)              :: Xmat, Ymat
   LIODBLE   , intent(in)              :: energy
   type(operator) , intent(inout)           :: dens_op, fock_op
   type(operator) , intent(inout), optional :: dens_opb, fock_opb

   logical      :: open_shell
   integer      :: ndiist, nediist
   LIODBLE, allocatable :: rho(:,:)

   ! If DIIS is turned on, update fockm with the current transformed F' (into
   ! ON basis) and update FP_PFm with the current transformed [F',P']
   !
   ! (1)     Calculate F' and [F',P']
   !       update fockm with F'
   ! now, scratch1 = A = F' * P'; scratch2 = A^T
   ! [F',P'] = A - A^T
   ! BASE CHANGE HAPPENS INSIDE OF FOCK_COMMUTS
   ! Saving rho and the first fock AO

   ndiist  = min(niter, ndiis )
   nediist = min(niter, nediis)

   open_shell = .false.
   if (present(dens_opb)) open_shell = .true.
   allocate(rho(M_in,M_in))

   ! Gets [F,P] and therefore the DIIS error.
   if (conver_method /= 1) then
      call g2g_timer_sum_start('Conv setup - BChange AOtoON')
      if (.not. dens_bchange_done) then
         call dens_op%BChange_AOtoON(Ymat, M_in)
         if (open_shell) call dens_opb%BChange_AOtoON(Ymat, M_in)
      end if
      call fock_op%BChange_AOtoON(Xmat, M_in)
      if (open_shell) call fock_opb%BChange_AOtoON(Xmat, M_in)
      call g2g_timer_sum_pause('Conv setup - BChange AOtoON')

      call g2g_timer_sum_start('Conv setup - DIIS commut')
      ! `rho` here is just scratch for diis_fock_commut, which overwrites it
      ! immediately via Gets_data_ON. The earlier Gets_data_AO(rho) calls
      ! were wasted M*M copies; removed.
      call diis_fock_commut(dens_op, fock_op, rho, M_in, 1, ndiist)
      call diis_get_error(M_in, 1, verbose)
      if (open_shell) then
         call diis_fock_commut(dens_opb, fock_opb, rho, M_in, 2, ndiist)
         call diis_get_error(M_in, 2, verbose)
      endif
      call g2g_timer_sum_pause('Conv setup - DIIS commut')
   endif
   dens_bchange_done = .false.

   ! DIIS and EDIIS
   if (conver_method > 1) then

      ! DIIS
      call g2g_timer_sum_start('Conv setup - DIIS update_emat')
      call diis_update_emat(niter, ndiist, M_in, open_shell)
      call g2g_timer_sum_pause('Conv setup - DIIS update_emat')
      ! Stores energy for bDIIS
      if ((conver_method > 3) .and. (niter > 1)) call diis_update_energy(energy)

      ! EDIIS
      if (conver_method > 5) then
         call ediis_update_energy_fock_rho(fock_op, dens_op, nediist, 1, niter)
         if (open_shell) call ediis_update_energy_fock_rho(fock_op, dens_op, &
                                                           nediist, 2, niter)
         call ediis_update_bmat(energy, nediist, niter, M_in, open_shell)

      endif
   endif

   deallocate(rho)
end subroutine converger_setup

subroutine converger_fock(niter, M_in, fock_op, spin, n_orbs, HL_gap, Xmat)

   ! Gets new Fock using convergence acceleration.
   use converger_data  , only: damping_factor, fock_damped, conver_method, &
                                 level_shift, lvl_shift_en, lvl_shift_cut,   &
                                 ndiis, nediis, scf_prev_was_diis
   use fileio_data     , only: verbose
   use typedef_operator, only: operator
   use typedef_cumat   , only: cumat_r

   implicit none
   ! Spin allows to store correctly alpha or beta information. - Carlos
   integer        , intent(in)    :: niter, M_in, spin, n_orbs
   LIODBLE   , intent(in)    :: HL_gap
   type(cumat_r)  , intent(in)    :: Xmat
   type(operator) , intent(inout) :: fock_op

   logical :: bdiis_on, ediis_on, diis_on
   integer :: ndiist, nediist
   LIODBLE, allocatable :: fock(:,:)

   allocate(fock(M_in,M_in))
   fock = 0.0D0

   ndiist  = min(niter, ndiis )
   nediist = min(niter, nediis)

   ! Selects methods according to different criteria.
   call select_methods(diis_on, ediis_on, bdiis_on, niter, verbose, spin)

   ! THIS IS DAMPING
   ! THIS IS SPARTA!
   ! If we are not doing diis this iteration, apply damping to F, save this
   ! F in fock_damped for next iteration's damping and put F' = X^T * F * X in
   ! fock the newly constructed damped matrix is stored, for next iteration in
   ! fock_damped
   if ((.not. diis_on) .and. (.not. ediis_on)) then
      call fock_op%Gets_data_AO(fock)

      if (niter > 1) &
         fock = (fock  + damping_factor * fock_damped(:,:,spin)) / &
                  (1.0D0 + damping_factor)
      fock_damped(:,:,spin) = fock
      call fock_op%Sets_data_AO(fock)
      call fock_op%BChange_AOtoON(Xmat, M_in)
   endif

   ! DIIS and EDIIS
   if (conver_method > 1) then
      if (diis_on) then
         if (bdiis_on) call diis_emat_bias(ndiist)

         call diis_get_new_fock(fock, ndiist, M_in, spin)
         call fock_op%Sets_data_ON(fock)
      endif

      ! EDIIS
      if (conver_method > 5) then
         if (ediis_on) then
            call ediis_get_new_fock(fock, nediist, spin)
            call fock_op%Sets_data_ON(fock)
         endif
      endif
   endif

   ! Level shifting works weird when using DIIS-only methods, so it is disabled.
   if ((HL_gap < lvl_shift_cut) .and. (level_shift) .and. &
      (.not. diis_on) .and. (.not. ediis_on)) then
         call fock_op%Shift_diag_ON(lvl_shift_en, n_orbs+1)
         if (verbose > 3) write(*,'(2x,A)') "Applying level shift."
   endif

   ! Record whether this iter used DIIS extrapolation. Used by the next iter's
   ! energy-rejection rollback check in select_methods. Only the alpha pass
   ! (spin == 1) writes the flag; beta uses the same diis_on decision via
   ! select_methods, so they're always consistent.
   if (spin == 1) scf_prev_was_diis = diis_on .or. ediis_on

   deallocate(fock)
end subroutine converger_fock

subroutine select_methods(diis_on, ediis_on, bdiis_on, niter, verbose, spin)
   ! Selects the convergence acceleration method(s) to use in each step.
   use converger_data, only: good_cut, rho_diff, rho_LS, EDIIS_start, &
                             DIIS_start, bDIIS_start, conver_method,  &
                             ediis_started, diis_started, diis_error, &
                             bdiis_started, scf_prev_energy_rise,     &
                             scf_prev_was_diis, scf_rollback_threshold, &
                             scf_rollback_rho_gate
   implicit none
   integer, intent(in)      :: niter, verbose, spin
   logical, intent(out)     :: diis_on, ediis_on, bdiis_on

   bdiis_on = .false.
   diis_on  = .false.
   ediis_on = .false.
   ! Checks convergence criteria.
   select case (conver_method)
      ! Always do damping
      case (1)

      ! Damping the first two steps, diis afterwards
      case (2)
         if (niter > 2) diis_on = .true.
      case (4)
         bdiis_on = .true.
         if (niter > 2) diis_on = .true.

      ! Damping until good enough, diis afterwards
      case(3)
         if ((rho_diff < good_cut) .and. (niter > 2) .and. (rho_LS < 2)) then
            if (verbose > 3) write(*,'(2x,A)') "  Doing DIIS."
            diis_on = .true.
         endif
      case(5)
         bdiis_on = .true.
         if ((rho_diff < good_cut) .and. (niter > 2) .and. (rho_LS < 2)) then
            if (verbose > 3) write(*,'(2x,A)') "  Doing bDIIS."
            diis_on = .true.
         endif

      ! Mix of all criteria, according to DIIS error value.
      case(6)
         if (niter < 3) return
         if (spin == 1) then
            if ((diis_error < EDIIS_start) .and. (.not. ediis_started)) then
               ediis_started = .true.
               if (verbose > 3) write(*,'(2x,A)') "  Doing EDIIS."
            endif
            if ((diis_error < DIIS_start)  .and. (.not. diis_started)) then
               diis_started = .true.
               if (verbose > 3) write(*,'(2x,A)') "  Doing DIIS."
            endif
            if ((diis_error < bDIIS_start) .and. (.not. bdiis_started)) then
               if (verbose > 3) write(*,'(2x,A)') "  Doing bDIIS."
               bdiis_started = .true.
            endif
         endif

         if (ediis_started)  then
            ediis_on = .true.
         endif
         if (diis_started) then
            ediis_on = .false.
            diis_on  = .true.
         endif
         if (bdiis_started) then
            bdiis_on = .true.
         endif
      case default
         write(*,'(A,I4)') 'ERROR - Wrong conver_method = ', conver_method
         stop
   endselect

   if (rho_LS > 1) then
      ediis_on = .false.
      diis_on  = .false.
   endif

   ! Energy-rejection rollback: if the previous iter used DIIS *with a small
   ! error vector* (so DIIS should have worked) and the energy still jumped
   ! upward by more than scf_rollback_threshold, fall back to damping this
   ! iter so the trajectory can recover before re-engaging DIIS. Gating on
   ! scf_prev_diis_error blocks the rollback from firing during the initial
   ! wild iters of a poorly-initialised SCF (where huge ΔE swings are normal
   ! SCF turbulence, not a true DIIS failure).
   if (scf_prev_was_diis .and. &
       (rho_diff < scf_rollback_rho_gate) .and. &
       (scf_prev_energy_rise > scf_rollback_threshold)) then
      diis_on  = .false.
      ediis_on = .false.
      bdiis_on = .false.
      if (verbose > 3 .and. spin == 1) write(*,'(2x,A,ES10.3,A,ES10.3,A)') &
         "  DIIS rollback: previous step raised E by ", &
         scf_prev_energy_rise, " Eh (rho_diff ", rho_diff, &
         "); damping this iter."
   endif
end subroutine select_methods

subroutine converger_check(rho_old, rho_new, energy_old, energy_new, &
                             n_iterations, is_converged, open_shell, ls_change)
   ! Checks convergence
   use fileio        , only: write_energy_convergence
   use converger_data, only: told, Etold, rho_diff, diis_error, conver_method, &
                             rho_LS, nMax, scf_prev_energy_rise,              &
                             conv_commut_tol, conv_rho_relax, conv_ediff_tol

   ! Calculates convergence criteria in density matrix, and
   ! store new density matrix in Pmat_vec.
   implicit none
   integer     , intent(in)  :: n_iterations
   logical     , intent(in)  :: open_shell
   LIODBLE, intent(in)  :: energy_old, energy_new
   LIODBLE, intent(in)  :: rho_new(:,:), rho_old(:)
   logical     , intent(out) :: is_converged, ls_change

   integer      :: jj, kk, Rposition, M2
   LIODBLE :: del, e_diff

   M2       = 2 * size(rho_new,1)
   e_diff   = abs(energy_new - energy_old)

   ! Signed energy delta drives the DIIS rollback heuristic in select_methods.
   ! Positive => energy rose; if the previous iter was a DIIS step, next iter
   ! will fall back to damping.
   scf_prev_energy_rise = energy_new - energy_old

   rho_diff = 0.0D0
   do jj = 1 , size(rho_new,1)
   do kk = jj, size(rho_new,1)
         Rposition = kk + (M2 - jj) * (jj -1) / 2
         del       = (rho_new(jj,kk) - rho_old(Rposition))
         rho_diff  = rho_diff + 2.0D0 * del * del
      enddo
   enddo
   rho_diff = sqrt(rho_diff) / dble(size(rho_new,1))

   is_converged = .false.
   if (e_diff < Etold) then
      if ((conver_method == 1) .or. (rho_LS > 1)) then
         ! Damping / linear-search modes: no reliable commutator is maintained,
         ! so fall back to the density-change criterion alone (unchanged).
         if (rho_diff < told) is_converged = .true.
      else
         ! DIIS-family. Original tight test, OR a noise-robust stationarity
         ! test using the commutator ||[F,P]|| (diis_error). See conv_commut_tol
         ! / conv_rho_relax in converger_data. Whichever fires first; this only
         ! ever converges at or before the original criterion.
         if ((rho_diff < told) .and. (diis_error < 1D-4)) then
            is_converged = .true.
         elseif ((diis_error < conv_commut_tol) .and. &
                 (rho_diff   < conv_rho_relax)  .and. &
                 (e_diff     < conv_ediff_tol)) then
            is_converged = .true.
         endif
      endif
   endif
   call write_energy_convergence(n_iterations, energy_new, rho_diff, told, &
                                 e_diff, etold)
   if ((rho_LS == 1) .and. (n_iterations == nMax) .and. (.not. is_converged)) &
      call rho_ls_switch(open_shell, size(rho_old), LS_change)

end subroutine converger_check
