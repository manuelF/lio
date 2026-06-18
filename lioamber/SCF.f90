#include "datatypes/datatypes.fh"
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! DIRECT VERSION
! Calls all integrals generator subroutines : 1 el integrals,
! 2 el integrals, exchange fitting , so it gets S matrix, F matrix
! and P matrix in lower storage mode (symmetric matrices)
!
! Dario Estrin, 1992
!------------------------------------------------------------------------------!
! Modified to f90
! Nick, 2017
!------------------------------------------------------------------------------!
! Header with new format. Added comments on how to proceed with a cleanup of
! of the subroutines. Other things to do:
! TODO: change to 3 space indentation.
! TODO: break at line 80.
! TODO: change to lowercase (ex: implicit none)
!
! This log can be removed once all improvements have been made.
! FFR, 01/2018
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
subroutine SCF(E, fock_aop, rho_aop, fock_bop, rho_bop)
   use ehrensubs , only: ehrendyn_init
   use garcha_mod, only : NCO, natom, number_restr, MEMO, &
                          igrid, energy_freq, converge, noconverge, &
                          VCINP, Nunp, igrid2, nsol, r, pc, Iz, &
                          Eorbs, Dbug, doing_ehrenfest, &
                          MO_coef_at, MO_coef_at_b, Smat, &
                          rhoalpha, rhobeta, OPEN, RealRho, d, ntatom,  &
                          Eorbs_b, npas, npasw, Fmat_vec, Fmat_vec2,        &
                          Ginv_vec, Gmat_vec, Hmat_vec, Pmat_en_wgt, Pmat_vec, &
                          sqsm
   use ECP_mod, only : ecpmode
   use field_data, only: field, fx, fy, fz
   use field_subs, only: field_calc, field_setup_old
   use faint_cpu, only: int1, intsol, int2, int3mem, int3lu
   use tbdft_data, only : tbdft_calc, MTBDFT, MTB,rhoa_tbdft,rhob_tbdft
   use tbdft_subs, only : getXY_TBDFT, build_chimera_TBDFT, extract_rhoDFT, &
                          construct_rhoTBDFT, write_rhofirstTB
   use transport_data, only: generate_rho0
   use mask_ecp      , only: ECP_fock, ECP_energy
   use typedef_sop   , only: sop              ! Testing SOP
   use fockbias_subs , only: fockbias_loads, fockbias_setmat, fockbias_apply
   use SCF_aux       , only: seek_nan, standard_coefs, messup_densmat, fix_densmat
   use liosubs_math  , only: transform
   use converger_data, only: Rho_LS, nMax, dens_bchange_done
   use converger_subs, only: converger_init, converger_fock, converger_setup, &
                             converger_check, rho_ls_init, do_rho_ls,         &
                             rho_ls_switch
   use typedef_operator, only: operator
   use typedef_cumat   , only: cumat_r
   use trans_Data    , only: gaussian_convert, rho_exc, translation
   use initial_guess_subs, only: get_initial_guess
   use scf_extrapolation , only: scf_extrap_predict, scf_extrap_store
   use fileio       , only: write_energies, write_energy_convergence, &
                            write_final_convergence, write_ls_convergence, &
                            movieprint
   use fileio_data  , only: verbose, movie_nfreq
   use basis_data   , only: kkinds, kkind, cools, cool, Nuc, nshell, M, MM, &
                            c_raw, Md, kknumd, kknums
   use basis_subs, only: neighbour_list_2e
   use excited_data,  only: libint_recalc
   use excitedsubs ,  only: ExcProp
   use fstsh_data  ,  only: FSTSH
   use fstshsubs   ,  only: TSHmain
   use lj_switch   ,  only: ljs_add_fock_terms, ljs_add_fock_terms_op
   use dftd3, only: dftd3_energy
   use properties, only: do_lowdin
   use extern_functional_subs, only: libint_init, exact_exchange, exact_energies
   use gpu_timers_interface
   use linalg_interface
   use gpu_interface
   use openblas_interface
   use omp_lib
   use lio_interface, only: get_restrain_energy

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!

   use packed_storage_interface
   implicit none
   ! E is the total SCF energy.
   ! The others are fock and rho operators alpha and beta (FOCK/RHO Alpha/Beta
   ! OPerator). In the case of closed shell, rho_aop and fock_aop contain full
   ! Fock and Rho matrices.
   LIODBLE  , intent(inout)           :: E
   type(operator), intent(inout)           :: rho_aop, fock_aop
   type(operator), intent(inout), optional :: rho_bop, fock_bop


   integer :: niter
   logical :: converged     = .false.
   logical :: changed_to_LS = .false.
   integer :: igpu

!  The following two variables are in a part of the code that is never
!  used. Check if these must be taken out...
   LIODBLE, allocatable :: morb_coefon(:,:)

!------------------------------------------------------------------------------!
!  TBDFT: variables to use as input for some subroutines instead of M and NCO
   integer :: M_f
   integer :: NCOa_f
   integer :: NCOb_f

   LIODBLE, allocatable :: rho_a0(:,:), rho_b0(:,:)
   LIODBLE, allocatable :: fock_a0(:,:), fock_b0(:,:)
   LIODBLE, allocatable :: rho_a(:,:), rho_b(:,:)
   LIODBLE, allocatable :: fock_a(:,:), fock_b(:,:)
   LIODBLE, allocatable :: morb_coefat(:,:)
   LIODBLE, allocatable :: X_min(:,:)
   LIODBLE, allocatable :: Y_min(:,:)
   LIODBLE, allocatable :: X_min_trans(:,:)
   LIODBLE, allocatable :: Y_min_trans(:,:)
   LIODBLE, allocatable :: morb_energy(:)
   integer             :: ii, jj, kk, kkk

!------------------------------------------------------------------------------!
! FFR variables
   type(sop)           :: overop
   LIODBLE, allocatable :: tmpmat(:,:)
   LIODBLE, allocatable :: Wdens_ewd(:,:), Cscal_ewd(:,:)
   complex(kind=8), allocatable :: rho_movie(:,:)
   LIODBLE  :: HL_gap = 10.0D0

!------------------------------------------------------------------------------!
! Energy contributions and convergence

   LIODBLE :: E1          ! kinetic + nuclear attraction + e-/MM charge
                         !    interaction + effective core potetial
   LIODBLE :: E1s = 0.0D0 ! kinetic + nuclear attraction + effective core
                         !    potetial
   LIODBLE :: E2          ! Coulomb (e- - e-)
   LIODBLE :: Eecp        ! Efective core potential
   LIODBLE :: En          ! nuclear-nuclear repulsion
   LIODBLE :: Ens         ! MM point charge-nuclear interaction
   LIODBLE :: Es          ! ???
   LIODBLE :: E_restrain  ! distance restrain
   LIODBLE :: Exc         ! exchange-correlation
   LIODBLE :: Etrash      ! auxiliar variable
   LIODBLE :: Evieja      !
   LIODBLE :: ELJS        ! LJ Switch contribution to energy.

   ! Base change matrices (for ON-AO changes).
   type(cumat_r)       :: Xmat, Ymat

   ! TODO : Variables to eliminate...
   LIODBLE, allocatable :: xnano(:,:)

   ! Carlos: Open shell, variables.
   LIODBLE              :: ocupF
   integer             :: NCOa, NCOb

!------------------------------------------------------------------------------!
!  Overlap of int3lu (Coulomb fit + Fock, CPU BLAS) with g2g_solve_groups
!  (XC Fock, GPU + CPU partition). Toggle via LIO_OVERLAP_INT3LU_G2G=1 env var.
!  XC contributions land in fmat_xc_scratch(_b) via a rebound
!  fortran_vars.rmm_output(_a/_b) pointer (see g2g_solve_groups_into_*), and
!  the post-section merge adds them into Fmat_vec(/Fmat_vec2).
   logical, save :: overlap_int3lu_g2g_initialized = .false.
   logical, save :: overlap_int3lu_g2g = .false.
   integer, save :: overlap_blas_threads = 4
   integer, save :: overlap_omp_threads = 0   ! 0 = leave global OMP unchanged
   LIODBLE, allocatable, save :: fmat_xc_scratch(:)
   LIODBLE, allocatable, save :: fmat_xc_scratch_b(:)
   character(len=16) :: env_overlap_str
   integer :: env_overlap_status
   integer :: prev_blas_threads
   integer :: prev_max_levels
   integer :: prev_omp_threads
   LIODBLE :: t_int3lu, t_g2g
   LIODBLE :: t_iter0, t_fock_w, t_build_w, t_accel_w, t_diag_w, t_moc_w

   ! Variables related to VdW
   LIODBLE :: E_dftd

   ! Variables for Exact Hartree Fock ( FULL, SHORT, LONG )
   LIODBLE :: Eexact, Eshort, Elong

   call g2g_timer_start('SCF_full')
   call g2g_timer_start('SCF')
   call g2g_timer_sum_start('SCF')
   call g2g_timer_sum_start('Initialize SCF')

   changed_to_LS=.false. ! LINSEARCH
   call rho_ls_init(open, MM)

   E=0.0D0
   E1=0.0D0
   En=0.0D0
   E2=0.0D0
   Es=0.0D0
   Eecp=0.d0
   Ens=0.0D0
   E_restrain=0.d0
   E_dftd=0.0D0
   Eexact=0.D0
   Eljs = 0.0D0

   ! Distance Restrain
   IF (number_restr.GT.0) THEN
      call get_restrain_energy(E_restrain)
      WRITE(*,*) "DISTANCE RESTRAIN ADDED TO FORCES"
   END IF

   !carlos: ocupation factor
   !carlos: NCOa works in open shell and close shell
   NCOa   = NCO
   NCOa_f = NCOa
   ocupF  = 2.0d0
   allocate(rho_b0(1,1))
   if (OPEN) then
      ! Number of OM down
      NCOb   = NCO + Nunp
      NCOb_f = NCOb
      ocupF = 1.0d0
      deallocate(rho_b0)
      allocate(rho_b0(M,M),fock_b0(M,M))
   endif
   allocate(fock_a0(M,M), rho_a0(M,M))

   M_f = M
   if (tbdft_calc /= 0) then
      M_f    = MTBDFT
      NCOa_f = NCOa + MTB / 2
      if (OPEN) NCOb_f = NCOb + MTB / 2
   endif

   allocate(fock_a(M_f,M_f), rho_a(M_f,M_f))
   allocate(morb_energy(M_f), morb_coefat(M_f,M_f))
   if (OPEN) then
      allocate(fock_b(M_f,M_f), rho_b(M_f,M_f))
   end if

!------------------------------------------------------------------------------!
! TODO: damp and gold should no longer be here??
! TODO: Qc should probably be a separated subroutine? Apparently it is only
!       used in dipole calculation so...it only adds noise to have it here.
! TODO: convergence criteria should be set at namelist/keywords setting

      Evieja=0.d0
      niter=0

!------------------------------------------------------------------------------!
! TODO: this whole part which calculates the non-electron depending terms of
!       fock and the overlap matrix should probably be in a separated sub.
!       (diagonalization of overlap, starting guess, should be taken out)
!
! Reformat from here...

! Nano: calculating neighbour list helps to make 2 electrons integral scale
! linearly with natoms/basis
!
      call neighbour_list_2e(natom, ntatom, r, d)

! -Create integration grid for XC here
! -Assign points to groups (spheres/cubes)
! -Assign significant functions to groups
! -Calculate point weights
!
      call g2g_timer_sum_start('Exchange-correlation grid setup')
      call g2g_reload_atom_positions(igrid2, Iz)
      call g2g_timer_sum_stop('Exchange-correlation grid setup')

      call aint_query_gpu_level(igpu)
      if (igpu.gt.1) call aint_new_step()

! Calculate 1e part of F here (kinetic/nuc in int1, MM point charges
! in intsol)
!
      call g2g_timer_sum_start('1-e Fock')
      call g2g_timer_sum_start('Nuclear attraction')
      call int1(En, Fmat_vec, Hmat_vec, Smat, d, r, Iz, natom, &
                ntatom)
      call ECP_fock( MM, Hmat_vec )

! Other terms
!
      call g2g_timer_sum_stop('Nuclear attraction')
      if(nsol.gt.0.or.igpu.ge.4) then
          call g2g_timer_sum_start('QM/MM')
       if (igpu.le.1) then
          call g2g_timer_start('intsol')
          call intsol(Pmat_vec, Hmat_vec, Iz, pc, r, d, natom, ntatom, &
                      E1s, Ens, .true.)
          call g2g_timer_stop('intsol')
        else
          call aint_qmmm_init(nsol,r,pc)
          call g2g_timer_start('aint_qmmm_fock')
          call aint_qmmm_fock(E1s,Ens)
          call g2g_timer_stop('aint_qmmm_fock')
        endif
          call g2g_timer_sum_stop('QM/MM')
      endif

! Initialization of libint
      call libint_init(c_raw,libint_recalc)

! test
! TODO: test? remove or sistematize
!
      E1=0.D0
      do kk=1,MM
        E1 = E1 + Pmat_vec(kk) * Hmat_vec(kk)
      enddo
      call g2g_timer_sum_stop('1-e Fock')


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! OVERLAP DIAGONALIZATION
! TODO: Simplify, this has too much stuff going on...
! (maybe trans mats are not even necessary?)
!
        if (allocated(X_min)) deallocate(X_min)
        if (allocated(Y_min)) deallocate(Y_min)
        if (allocated(X_min_trans)) deallocate(X_min_trans)
        if (allocated(Y_min_trans)) deallocate(Y_min_trans)

        allocate(X_min(M,M), Y_min(M,M), X_min_trans(M,M), Y_min_trans(M,M))

        call g2g_timer_sum_start('Overlap diagonalization')
        call overop%Sets_smat( Smat )
        if (do_lowdin()) then
!          TODO: inputs insuficient; there is also the symetric orthog using
!                3 instead of 2 or 1. Use integer for onbasis_id
           call overop%Gets_orthog_4m( 2, 0.0d0, X_min, Y_min, X_min_trans, Y_min_trans)
        else
           call overop%Gets_orthog_4m( 1, 0.0d0, X_min, Y_min, X_min_trans, Y_min_trans)
        end if

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!
!
!  Fockbias setup
        if ( allocated(sqsm) ) deallocate(sqsm)
        if ( allocated(tmpmat) ) deallocate(tmpmat)
        allocate( sqsm(M,M), tmpmat(M,M) )
        call overop%Gets_orthog_2m( 2, 0.0d0, tmpmat, sqsm )
        call fockbias_loads( natom, nuc )
        call fockbias_setmat( sqsm )
        deallocate( tmpmat )


!TBDFT: Dimensions of Xmat and Ymat are modified for TBDFT.
!
! TODO: this is nasty, a temporary solution would be to have a Msize variable
!       be assigned M (or, even better, "basis_size") or MTBDFT
!       ("basis_size_dftb") according to the case
! Uses arrays fock_a y rho_a as temporary storage to initialize Xmat and Ymat.

   call getXY_TBDFT(M, X_min, Y_min, fock_a, rho_a)
   call Xmat%init(M_f, fock_a)
   call Ymat%init(M_f, rho_a)
   call g2g_timer_sum_stop('Overlap diagonalization')

   deallocate(X_min, Y_min, X_min_trans, Y_min_trans)

! Generates starting guess
!
   if ( (.not. VCINP) .and. (npas == 1) ) then
      call get_initial_guess(M, MM, NCO, NCOb, &
                             Xmat%matrix(MTB+1:MTB+M,MTB+1:MTB+M),        &
                             Hmat_vec, Smat, Pmat_vec, rhoalpha, rhobeta, &
                             OPEN, natom, Iz, nshell, Nuc)
   endif

!----------------------------------------------------------!
! Precalculate two-index (density basis) "G" matrix used in density fitting
! here (S_ij in Dunlap, et al JCP 71(8) 1979).
! Also, pre-calculate G^-1 if G is not ill-conditioned.
      call g2g_timer_sum_start('Coulomb G matrix')
      call int2(Gmat_vec, Ginv_vec, r, d, ntatom)
      call g2g_timer_sum_stop('Coulomb G matrix')

! Precalculate three-index (two in MO basis, one in density basis) matrix
! used in density fitting / Coulomb F element calculation here
! (t_i in Dunlap)
!
      call aint_query_gpu_level(igpu)
      if (igpu.gt.2) then
        call aint_coulomb_init()
      endif
      if (igpu.eq.5) MEMO = .false.
      !MEMO=.true.
      if (MEMO) then
         call g2g_timer_start('int3mem')
         call g2g_timer_sum_start('Coulomb precalc')
!        Large elements of t_i put into double-precision cool here
!        Size criteria based on size of pre-factor in Gaussian Product Theorem
!        (applied to MO basis indices)
         call int3mem(r, d, natom, ntatom)
         call g2g_timer_stop('int3mem')
         call g2g_timer_sum_stop('Coulomb precalc')
!        Kick off the background GPU upload of cool/cools for the int3lu
!        Coulomb-fit offload while the GPU is still quiet (pinning/alloc
!        mid-iteration stalls concurrent XC kernels through the driver
!        lock). See the notes in subm_int3lu.f90.
         call int3lu_gpu_prefetch(cool, cools, Md, kknumd, kknums)
      endif
!
!##########################################################!
! TODO: ...to here
!##########################################################!
!
!
!
!------------------------------------------------------------------------------!
! TODO: the following comment is outdated? Also, hybrid_converg switch should
!       be handled differently.
!
! Now, damping is performed on the density matrix
! The first 4 iterations ( it may be changed, if necessary)
! when the density is evaluated on the grid, the density
! matrix is used ( slower), after that it is calculated
! using the vectors . Since the vectors are not damped,
! only at the end of the SCF, the density matrix and the
! vectors are 'coherent'

!------------------------------------------------------------------------------!
!  One-time setup for int3lu/g2g overlap.
      if (.not. overlap_int3lu_g2g_initialized) then
         call get_environment_variable("LIO_OVERLAP_INT3LU_G2G", &
                                       env_overlap_str, &
                                       status=env_overlap_status)
         if (env_overlap_status == 0 .and. trim(env_overlap_str) == "1") then
            overlap_int3lu_g2g = .true.
            ! Allow override of BLAS thread split inside the int3lu section.
            call get_environment_variable("LIO_OVERLAP_BLAS_THREADS", &
                                          env_overlap_str, &
                                          status=env_overlap_status)
            if (env_overlap_status == 0) then
               read(env_overlap_str, *, iostat=env_overlap_status) &
                    overlap_blas_threads
               if (env_overlap_status /= 0 .or. overlap_blas_threads < 1) &
                    overlap_blas_threads = g2g_recommended_blas_threads()
            else
               overlap_blas_threads = g2g_recommended_blas_threads()
            endif

            ! OMP cap is applied LOCALLY around the parallel sections only.
            ! Process-wide omp_set_num_threads() changes FP-summation order in
            ! downstream OMP regions and causes open-shell heme to need 2-3x
            ! more SCF iterations. Caller can override via LIO_OVERLAP_OMP_THREADS,
            ! or pass 0 (default) to leave the ambient OMP thread count alone
            ! when no OMP_NUM_THREADS env var is set.
            call get_environment_variable("LIO_OVERLAP_OMP_THREADS", &
                                          env_overlap_str, &
                                          status=env_overlap_status)
            if (env_overlap_status == 0) then
               read(env_overlap_str, *, iostat=env_overlap_status) &
                    overlap_omp_threads
               if (env_overlap_status /= 0 .or. overlap_omp_threads < 0) &
                    overlap_omp_threads = 0
            else
               call get_environment_variable("OMP_NUM_THREADS", &
                                             env_overlap_str, &
                                             status=env_overlap_status)
               if (env_overlap_status == 0) then
                  ! User set OMP_NUM_THREADS explicitly: cap to recommended
                  ! around the sections so int3lu BLAS and g2g's CPU partition
                  ! fit on the physical cores.
                  overlap_omp_threads = g2g_recommended_omp_threads()
               else
                  ! No OMP_NUM_THREADS: ambient OMP is the runtime default
                  ! (often #logical cores). Leave it alone; capping it
                  ! perturbs heme convergence.
                  overlap_omp_threads = 0
               endif
            endif

            if (verbose > 1) write(*,'(A,I0,A,I0,A,I0,A)') &
               " [overlap] int3lu/g2g overlap ENABLED (OMP=", &
               omp_get_max_threads(), " ambient, OMP_cap=", &
               overlap_omp_threads, " g2g, BLAS=", &
               overlap_blas_threads, " int3lu)"
         endif
         overlap_int3lu_g2g_initialized = .true.
      endif

      if (overlap_int3lu_g2g) then
         if (.not. allocated(fmat_xc_scratch)) allocate(fmat_xc_scratch(MM))
         if (size(fmat_xc_scratch) /= MM) then
            deallocate(fmat_xc_scratch)
            allocate(fmat_xc_scratch(MM))
         endif
         if (OPEN) then
            if (.not. allocated(fmat_xc_scratch_b)) allocate(fmat_xc_scratch_b(MM))
            if (size(fmat_xc_scratch_b) /= MM) then
               deallocate(fmat_xc_scratch_b)
               allocate(fmat_xc_scratch_b(MM))
            endif
         endif
      endif

      call g2g_timer_sum_stop('Initialize SCF')

!------------------------------------------------------------------------------!
! TODO: Maybe evaluate conditions for loop continuance at the end of loop
!       and condense in a single "keep_iterating" or something like that.
   if (verbose > 1) then
      write(*,*)
      write(*,'(A)') "Starting SCF cycles."
   endif

   ! Cross-step density extrapolation (ASPC). On MD / geometry-optimization
   ! steps after the first, replace the plain VCINP "reuse last density" guess
   ! with a time-reversible extrapolation of the last few converged densities.
   ! No-op on single points and on step 1 (no history yet). Guess-only: the
   ! converged result is unchanged, only the iteration count moves.
   call scf_extrap_predict(MM, Pmat_vec, rhoalpha, rhobeta, OPEN, r, ntatom)

   converged = .false.
   call converger_init( M_f, OPEN )

   do 999 while ( (.not. converged) .and. (niter <= nMax) )
      call g2g_timer_start('Total iter')
      call g2g_timer_sum_start('Iteration')
      call g2g_timer_sum_start('Fock integrals')

      niter = niter + 1
      t_iter0   = omp_get_wtime()
      t_fock_w  = 0.0d0
      t_build_w = 0.0d0
      t_accel_w = 0.0d0
      t_diag_w  = 0.0d0
      t_moc_w   = 0.0d0

      ! Test for NaN
      if (Dbug) call SEEK_NaN(Pmat_vec,1,MM,"RHO Start")
      if (Dbug) call SEEK_NaN(Fmat_vec,1,MM,"FOCK Start")

      E2  = 0.0D0
      Exc = 0.0D0
      if (overlap_int3lu_g2g) then
!        Overlap path: int3lu (CPU BLAS) and g2g_solve_groups (GPU+CPU partition)
!        run concurrently. int3lu writes Fmat_vec(/Fmat_vec2) = Hmat + Coulomb
!        in-place; g2g writes XC into the zero-initialized scratch buffers via
!        rebound fortran_vars.rmm_output(_a/_b) pointers. After both finish we
!        add scratch into the Fock matrix. The BLAS inside int3lu is throttled
!        to overlap_blas_threads to leave most cores for g2g's CPU partition.
         prev_blas_threads = openblas_get_num_threads()
         prev_max_levels   = omp_get_max_active_levels()
         prev_omp_threads  = omp_get_max_threads()
         fmat_xc_scratch(1:MM) = 0.0d0
         if (OPEN) fmat_xc_scratch_b(1:MM) = 0.0d0

!        Enable nested OpenMP only for the duration of the sections so the
!        inner parallel-for inside g2g_solve actually spawns workers.
!        Restored immediately after to avoid leaking process-wide nesting
!        into TDDFT/Ehrenfest paths that call g2g/BLAS without expecting it.
         if (prev_max_levels < 2) call omp_set_max_active_levels(2)

!        Optionally cap ambient OMP threads for the duration of the parallel
!        sections only. Restored after the merge so other code (TD, Ehrenfest,
!        OMP reductions in compute_functions/weight) sees the original thread
!        count. Capping process-wide perturbs heme convergence (2-3x more iters).
         if (overlap_omp_threads > 0 .and. &
             overlap_omp_threads /= prev_omp_threads) then
            call omp_set_num_threads(overlap_omp_threads)
         endif

         call g2g_timer_sum_start('Coulomb fit + Fock')
         t_int3lu = 0.0d0
         t_g2g    = 0.0d0
         if (OPEN) then
!$omp parallel sections default(shared) num_threads(2)
!$omp section
            call openblas_set_num_threads(overlap_blas_threads)
            t_int3lu = -omp_get_wtime()
            call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
                        Hmat_vec, open, MEMO)
            t_int3lu = t_int3lu + omp_get_wtime()
            if (tbdft_calc == 0) then
               call spunpack_rho('L', M, rhoalpha, rho_a0)
               call rho_aop%Sets_data_AO(rho_a0)
               call rho_aop%BChange_AOtoON(Ymat, M_f)
               call spunpack_rho('L', M, rhobeta, rho_b0)
               call rho_bop%Sets_data_AO(rho_b0)
               call rho_bop%BChange_AOtoON(Ymat, M_f)
               dens_bchange_done = .true.
            end if
!$omp section
            t_g2g = -omp_get_wtime()
            call g2g_solve_groups_into_open(0, Exc, 0.0D0, fmat_xc_scratch, &
                                            fmat_xc_scratch_b)
            t_g2g = t_g2g + omp_get_wtime()
!$omp end parallel sections
         else
!$omp parallel sections default(shared) num_threads(2)
!$omp section
            call openblas_set_num_threads(overlap_blas_threads)
            t_int3lu = -omp_get_wtime()
            call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
                        Hmat_vec, open, MEMO)
            t_int3lu = t_int3lu + omp_get_wtime()
            if (tbdft_calc == 0) then
               call spunpack_rho('L', M, Pmat_vec, rho_a0)
               call rho_aop%Sets_data_AO(rho_a0)
               call rho_aop%BChange_AOtoON(Ymat, M_f)
               dens_bchange_done = .true.
            end if
!$omp section
            t_g2g = -omp_get_wtime()
            call g2g_solve_groups_into(0, Exc, 0.0D0, fmat_xc_scratch)
            t_g2g = t_g2g + omp_get_wtime()
!$omp end parallel sections
         endif
         call openblas_set_num_threads(prev_blas_threads)
         if (prev_max_levels < 2) call omp_set_max_active_levels(prev_max_levels)
         if (overlap_omp_threads > 0 .and. &
             overlap_omp_threads /= prev_omp_threads) then
            call omp_set_num_threads(prev_omp_threads)
         endif
         if (verbose > 3) then
            write(*,'(A,F6.1,A,F6.1,A,F6.1,A)') &
               "  [overlap] int3lu=", t_int3lu*1e3, "ms  g2g=", &
               t_g2g*1e3, "ms  idle=", abs(t_int3lu-t_g2g)*1e3, "ms"
         endif

!        Merge XC contributions into the Fock matrix(es).
         Fmat_vec(1:MM) = Fmat_vec(1:MM) + fmat_xc_scratch(1:MM)
         if (OPEN) Fmat_vec2(1:MM) = Fmat_vec2(1:MM) + fmat_xc_scratch_b(1:MM)
         call g2g_timer_sum_pause('Coulomb fit + Fock')

         if (Dbug) then
            call SEEK_NaN(Fmat_vec,1,MM,"FOCK Ex-Corr")
            if (open) call SEEK_NaN(Fmat_vec2,1,MM,"FOCK B Ex-Corr")
         endif
      else
         ! Computes Coulomb part of Fock, and energy on E2
         call g2g_timer_sum_start('Coulomb fit + Fock')
         call int3lu(E2, Pmat_vec, Fmat_vec2, Fmat_vec, Gmat_vec, Ginv_vec, &
                     Hmat_vec, open, MEMO)
         call g2g_timer_sum_pause('Coulomb fit + Fock')

         if (Dbug) then
            call SEEK_NaN(Fmat_vec,1,MM,"FOCK Coulomb")
            if (open) call SEEK_NaN(Fmat_vec2,1,MM,"FOCK B Coulomb")
         endif

         ! XC integration / Fock elements
         call g2g_timer_sum_start('Exchange-correlation Fock')
         call g2g_solve_groups(0,Exc, 0.0D0)
         call g2g_timer_sum_pause('Exchange-correlation Fock')

         ! Test for NaN
         if (Dbug) then
            call SEEK_NaN(Fmat_vec,1,MM,"FOCK Ex-Corr")
            if (open) call SEEK_NaN(Fmat_vec2,1,MM,"FOCK B Ex-Corr")
         endif
      endif


      ! Calculates 1e energy contributions (including solvent)
      E1 = 0.0D0
      if (generate_rho0) then
         ! REACTION FIELD CASE
         if (field) call field_setup_old(1.0D0, 0, fx, fy, fz)
         call field_calc(E1, 0.0D0, Pmat_vec, Fmat_vec2, Fmat_vec, r, d, &
                         natom, ntatom, open, 2*NCO+NUNP, Iz, pc)
         do kk = 1, MM
            E1 = E1 + Pmat_vec(kk) * Hmat_vec(kk)
         enddo
      else
         do kk=1,MM
            E1 = E1 + Pmat_vec(kk) * Hmat_vec(kk)
         enddo
      endif

      ! Calculates total energy
      E = E1 + E2 + En + Exc
      call g2g_timer_sum_pause('Fock integrals')
      t_fock_w = omp_get_wtime() - t_iter0

      ! Unpacks Fock/density, applies bias + LJ + exact-exchange terms.
      call g2g_timer_sum_start('Fock matrix build')
      if (OPEN) then
         call spunpack_rho('L', M, rhoalpha , rho_a0)
         call spunpack_rho('L', M, rhobeta  , rho_b0)
         call spunpack(    'L', M, Fmat_vec , fock_a0)
         call spunpack(    'L', M, Fmat_vec2, fock_b0)
         call fockbias_apply( 0.0d0, fock_a0)
         call fockbias_apply( 0.0d0, fock_b0)
      else
         call spunpack_rho('L', M, Pmat_vec, rho_a0)
         call spunpack(    'L', M, Fmat_vec, fock_a0)
         call fockbias_apply(0.0d0, fock_a0)
      end if

      if (OPEN) then
         call ljs_add_fock_terms_op(fock_a0, ELJS, rho_a0, Smat, fock_b0, rho_b0)
      else
         call ljs_add_fock_terms(fock_a0, ELJS, rho_a0, Smat)
      endif
      E = E + ELJS


!     EXACT EXCHANGE TERMS ( FULL, SHORT, LONG )
      call exact_exchange(rho_a0,rho_b0,fock_a0,fock_b0,M)

      if (tbdft_calc == 0) then
         fock_a = fock_a0
         rho_a  = rho_a0
         if (OPEN) fock_b = fock_b0
         if (OPEN) rho_b  = rho_b0
      else
         ! TBDFT: We extract rho and fock before convergence acceleration
         ! routines. Then, Fock and Rho for TBDFT are builded.
         call build_chimera_TBDFT (M, fock_a0, fock_a)
         call construct_rhoTBDFT(M, rho_a, rho_a0 ,rhoa_tbdft, niter,OPEN)
         if (OPEN) then
            call build_chimera_TBDFT(M, fock_b0, fock_b)
            call construct_rhoTBDFT(M, rho_b, rho_b0 ,rhob_tbdft,niter, OPEN)
         end if
      endif

      ! Stores matrices in operators, and sets up matrices in convergence
      ! acceleration algorithms (DIIS/EDIIS).
      call rho_aop%Sets_data_AO(rho_a)
      call fock_aop%Sets_data_AO(fock_a)
      call g2g_timer_sum_pause('Fock matrix build')
      t_build_w = omp_get_wtime() - t_iter0 - t_fock_w

      call g2g_timer_sum_start('SCF acceleration setup')
      if (OPEN) then
         call rho_bop%Sets_data_AO(rho_b)
         call fock_bop%Sets_data_AO(fock_b)
         call converger_setup(niter, M_f, rho_aop, fock_aop, E,  Xmat, Ymat, &
                              rho_bop, fock_bop)
      else
         call converger_setup(niter, M_f, rho_aop, fock_aop, E,  Xmat, Ymat)
      endif
      call g2g_timer_sum_pause('SCF acceleration setup')
      t_accel_w = omp_get_wtime() - t_iter0 - t_fock_w - t_build_w

      ! Convergence accelerator processing.
      ! In closed shell, rho_a is the total density matrix; in open shell,
      ! it is the alpha density.
      call g2g_timer_sum_start('SCF acceleration')
      call converger_fock(niter, M_f, fock_aop, 1, NCOa_f, HL_gap, Xmat)
      call g2g_timer_sum_pause('SCF acceleration')

      ! Fock(ON) diagonalization
      if ( allocated(morb_coefon) ) deallocate(morb_coefon)
      allocate( morb_coefon(M_f,M_f) )
      call g2g_timer_sum_start('SCF - Fock Diagonalization (sum)')
      t_diag_w = t_diag_w - omp_get_wtime()
      call fock_aop%Diagon_datamat( morb_coefon, morb_energy )
      call g2g_timer_sum_pause('SCF - Fock Diagonalization (sum)')
      t_diag_w = t_diag_w + omp_get_wtime()

      ! Base change of coeficients ( (X^-1)*C ) and construction of new
      ! density matrix.
      call g2g_timer_sum_start('SCF - MOC base change (sum)')
      call Xmat%multiply(morb_coefat, morb_coefon)
      call standard_coefs( morb_coefat )
      call g2g_timer_sum_pause('SCF - MOC base change (sum)')

      ! Builds the new AO density matrix from the MO coefficients.
      call g2g_timer_sum_start('Density build')
      if ( allocated(morb_coefon) ) deallocate(morb_coefon)
      call rho_aop%Dens_build(M_f, NCOa_f, ocupF, morb_coefat)
      call rho_aop%Gets_data_AO(rho_a)
      call messup_densmat(rho_a)
      call g2g_timer_sum_pause('Density build')

      Eorbs      = morb_energy
      MO_coef_at = morb_coefat

      if (OPEN) then
         ! In open shell, performs the previous operations for beta operators.
         call g2g_timer_sum_start('SCF acceleration')
         call converger_fock(niter, M_f, fock_bop, 2, NCOb_f, HL_gap, Xmat)
         call g2g_timer_sum_pause('SCF acceleration')

         ! Fock(ON) diagonalization
         if ( allocated(morb_coefon) ) deallocate(morb_coefon)
         allocate( morb_coefon(M_f,M_f) )

         call g2g_timer_sum_start('SCF - Fock Diagonalization (sum)')
         t_diag_w = t_diag_w - omp_get_wtime()
         call fock_bop%Diagon_datamat( morb_coefon, morb_energy )
         call g2g_timer_sum_pause('SCF - Fock Diagonalization (sum)')
         t_diag_w = t_diag_w + omp_get_wtime()

         ! Base change of coeficients ( (X^-1)*C ) and construction of new
         ! density matrix.
         call g2g_timer_sum_start('SCF - MOC base change (sum)')
         call Xmat%multiply(morb_coefat, morb_coefon)
         call standard_coefs( morb_coefat )
         call g2g_timer_sum_pause('SCF - MOC base change (sum)')

         call g2g_timer_sum_start('Density build')
         if ( allocated(morb_coefon) ) deallocate(morb_coefon)
         call rho_bop%Dens_build(M_f, NCOb_f, ocupF, morb_coefat)
         call rho_bop%Gets_data_AO(rho_b)
         call messup_densmat( rho_b )
         call g2g_timer_sum_pause('Density build')

         Eorbs_b      = morb_energy
         MO_coef_at_b = morb_coefat
      endif

      ! Calculates HOMO-LUMO gap.
      HL_gap = abs(Eorbs(NCOa_f+1) - Eorbs(NCOa_f))
      if (OPEN) HL_gap = abs(min(Eorbs(NCOa_f+1),Eorbs_b(NCOb_f+1)) &
                             - max(Eorbs(NCOa_f),Eorbs_b(NCOb_f)))

      ! We are not sure how to translate the sumation over molecular orbitals
      ! and energies when changing from TBDFT system to DFT subsystem. Forces
      ! may be broken due to this. This should not be affecting normal DFT
      ! calculations.

      ! Perfoms TBDFT checks and extracts density matrices. Allocates xnano,
      ! which contains the total (alpha+beta) density matrix.
      call g2g_timer_sum_start('Rho update & check')
      allocate ( xnano(M,M) )

      if (tbdft_calc == 0) then
         xnano = rho_a
         if (OPEN) xnano = xnano + rho_b
      else
         rhoa_TBDFT = rho_a
         call extract_rhoDFT(M, rho_a, rho_a0)
         xnano = rho_a0

         if (OPEN) then
            rhob_TBDFT = rho_b
            call extract_rhoDFT(M, rho_b, rho_b0)
            xnano = xnano + rho_b0
         endif
      endif

      ! Optional linear search in Rho; each step re-evaluates the energy
      ! (int3lu + g2g), so it can dominate the iteration when active.
      call g2g_timer_sum_start('Rho linear search')
      if ((rho_LS > 1) .and. (niter > 10)) then
         ! Performs a linear search in Rho if activated. This uses the
         ! vector-form densities as the old densities, and matrix-form
         ! densities as the new ones.
         if (open) then
            call do_rho_ls(En, E1, E2, Exc, xnano, Pmat_vec, Hmat_vec,    &
                           Fmat_vec, Fmat_vec2, Gmat_vec, Ginv_vec, memo, &
                           rho_a, rho_b, rhoalpha, rhobeta)
         else
            call do_rho_ls(En, E1, E2, Exc, xnano, Pmat_vec, Hmat_vec, &
                           Fmat_vec, Fmat_vec2, Gmat_vec, Ginv_vec, memo)
         endif
      endif
      call g2g_timer_sum_pause('Rho linear search')

      E = E + Eexact
      ! Checks convergence criteria and starts linear search if able.
      call converger_check(Pmat_vec, xnano, Evieja, E, niter, converged, &
                           open, changed_to_LS)

      ! Updates old density matrices with the new ones and updates energy.
      call sprepack('L', M, Pmat_vec, xnano)
      if (OPEN) then
         if (tbdft_calc /= 0) then
            call sprepack('L', M, rhoalpha, rho_a0)
            call sprepack('L', M, rhobeta , rho_b0)
         else
            call sprepack('L', M, rhoalpha, rho_a)
            call sprepack('L', M, rhobeta , rho_b)
         endif
      endif
      deallocate ( xnano )
      Evieja = E
      call g2g_timer_sum_pause('Rho update & check')

      if (verbose >= 2) then
         t_moc_w = omp_get_wtime() - t_iter0 - t_fock_w - t_build_w &
                                   - t_accel_w - t_diag_w
         write(*,'(A,I3,5(A,F7.1),A,F7.1,A)') &
            "  [iter]", niter,                 &
            "  fock=",  t_fock_w *1e3,         &
            " build=",  t_build_w*1e3,         &
            " accel=",  t_accel_w*1e3,         &
            " diag=",   t_diag_w *1e3,         &
            " rest=",   t_moc_w  *1e3,         &
            " tot=", (omp_get_wtime()-t_iter0)*1e3, "ms"
      end if

      call g2g_timer_stop('Total iter')
      call g2g_timer_sum_pause('Iteration')
999 continue

   call g2g_timer_sum_start('Finalize SCF')

   ! Checks of convergence
   if (niter >= nMax) then
      call write_final_convergence(.false., nMax, Evieja)
      noconverge = noconverge + 1
      converge   = 0
   else
      call write_final_convergence(.true., niter, Evieja)
      converge   = converge + 1
      noconverge = 0

      ! Push the converged density onto the ASPC history so the next MD /
      ! geometry step can extrapolate from it. Only converged densities are
      ! stored, so a failed step never poisons the trajectory history.
      call scf_extrap_store(MM, Pmat_vec, rhoalpha, rhobeta, OPEN, r, ntatom)
   endif

   if (changed_to_LS) then
      changed_to_LS = .false.
      nMax          = nMax / 2
      Rho_LS        = 1
   endif

   if (noconverge > 4) then
      write(6,'(A)') "FATAL ERROR - No convergence achieved "&
                    &"4 consecutive times."
      stop
   endif


   if (MOD(npas,energy_freq).eq.0) then
!       Resolve with last density to get XC energy
        call g2g_timer_sum_start('Exchange-correlation energy')
        call g2g_new_grid(igrid)
        call g2g_solve_groups(1, Exc, 0.0D0)
        call g2g_timer_sum_stop('Exchange-correlation energy')

!       COmputing the QM/MM contribution to total energy
!       Total SCF energy =
!       E1   - kinetic + nuclear attraction + QM/MM interaction + effective
!              core potential
!       E2   - Coulomb
!       En   - nuclear-nuclear repulsion
!       Ens  - MM point charge - nuclear interaction
!       Es   - Full QM/MM interaction energy
!       Exc  - exchange-correlation
!       Eecp - Efective core potential
!       E_restrain - distance restrain

!       One electron Kinetic (with aint >3) or Kinetic + Nuc-elec (aint >=3)
        call int1(En, Fmat_vec, Hmat_vec, Smat, d, r, Iz, natom, &
                  ntatom)

!       Computing the E1-fock without the MM atoms
        if (nsol.gt.0.and.igpu.ge.1) then
          call aint_qmmm_init(0,r,pc)
          call aint_qmmm_fock(E1s,Etrash)
          call aint_qmmm_init(nsol,r,pc)
        endif

!       E1s (here) is the 1e-energy without the MM contribution
        E1s=0.D0
        do kk=1,MM
          E1s = E1s + Pmat_vec(kk) * Hmat_vec(kk)
        enddo

!       Es is the QM/MM energy computated as total 1e - E1s + QMnuc-MMcharges
!       NucleusQM-CHarges MM
        Es = Ens
        Es = Es + E1 - E1s

        ! Calculates DTFD3 Grimme's corrections to energy.
        call g2g_timer_sum_start("DFTD3 Energy")
        call dftd3_energy(E_dftd, d, natom, .true.)
        call g2g_timer_sum_pause("DFTD3 Energy")

!       All Exact Exchange Energy
        Eexact=0.0d0; Eshort=0.0d0; Elong=0.0d0
        call exact_energies(rho_a0,rho_b0,Eexact,Eshort,Elong,M)
        Eexact = Eexact + Eshort + Elong

!       Part of the QM/MM contrubution are in E1
        E = E1 + E2 + En + Ens + Exc + E_restrain + E_dftd + Eexact + ELJS
!       Write Energy Contributions
        if (npas.eq.1) npasw = 0

        if (npas.gt.npasw) then
           call ECP_energy( MM, Pmat_vec, Eecp, Es )
           call write_energies(E1, E2, En, Ens, Eecp, Exc, ecpmode, E_restrain, &
                               number_restr, nsol, E_dftd, Eexact, Es, ELJS)
           npasw=npas+10
        end if
      endif ! npas

      ! Calculation of energy weighted density matrix.
      !   W = - C_occ . diag(eps) . C_occ^T  (alpha + beta for open shell),
      ! stored packed-triangular with the off-diagonal entries doubled. The
      ! hand loop below is O(M^2 * NCO) of strided scalar FMAs (~10^10 ops,
      ! ~30 s on a 100-atom/M=2600 case); the BLAS fast path expresses it as
      ! one (two for open) DGEMM, cutting it to well under a second. The
      ! result feeds only the force/gradient routines (dft_get_qm_forces,
      ! WSgradcalc), never the SCF trajectory, so reordering the summation is
      ! numerically inert (verified bit-for-bit: max|blas-scalar|/max ~1e-15).
      ! The MTB>0 (TBDFT) layout keeps the original loop.
      call g2g_timer_sum_start('energy-weighted density')
      kkk = 0
      Pmat_en_wgt = 0.0D0
      if (MTB == 0) then
         allocate(Wdens_ewd(M,M), Cscal_ewd(M,NCOa_f))
         do kk = 1, NCOa_f
            Cscal_ewd(:,kk) = Eorbs(kk) * MO_coef_at(1:M,kk)
         enddo
         call DGEMM('N','T', M, M, NCOa_f, 1.0D0, MO_coef_at, M_f, &
                    Cscal_ewd, M, 0.0D0, Wdens_ewd, M)
         if (OPEN) then
            deallocate(Cscal_ewd); allocate(Cscal_ewd(M,NCOb_f))
            do kk = 1, NCOb_f
               Cscal_ewd(:,kk) = Eorbs_b(kk) * MO_coef_at_b(1:M,kk)
            enddo
            call DGEMM('N','T', M, M, NCOb_f, 1.0D0, MO_coef_at_b, M_f, &
                       Cscal_ewd, M, 1.0D0, Wdens_ewd, M)
         endif
         do jj = 1, M
            kkk = kkk + 1
            if (OPEN) then
               Pmat_en_wgt(kkk) = -1.0D0 * Wdens_ewd(jj,jj)
            else
               Pmat_en_wgt(kkk) = -2.0D0 * Wdens_ewd(jj,jj)
            endif
            do ii = jj+1, M
               kkk = kkk + 1
               if (OPEN) then
                  Pmat_en_wgt(kkk) = -2.0D0 * Wdens_ewd(ii,jj)
               else
                  Pmat_en_wgt(kkk) = -4.0D0 * Wdens_ewd(ii,jj)
               endif
            enddo
         enddo
         deallocate(Wdens_ewd, Cscal_ewd)

      else if (.not. OPEN) then
         ! Closed shell, TBDFT layout (MTB > 0).
         do jj = MTB+1, MTB+M
            kkk = kkk +1
            do kk = MTB+1, NCOa_f
               Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - 2.0D0 * Eorbs(kk) * &
                                  MO_coef_at(jj,kk) * MO_coef_at(jj,kk)
            enddo

            do ii = MTB+jj+1, M_f
               kkk = kkk +1
               do kk = MTB+1, NCOa_f
                  Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - 4.0D0 * Eorbs(kk) * &
                                     MO_coef_at(ii,kk) * MO_coef_at(jj,kk)
               enddo
            enddo
         enddo

      else
         ! Open shell, TBDFT layout (MTB > 0).
         do jj = MTB+1, MTB+M
            kkk = kkk +1
            do kk = MTB+1, NCOa_f
               Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - Eorbs(kk) * &
                                  MO_coef_at(jj,kk) * MO_coef_at(jj,kk)
            enddo
            do kk = MTB+1, NCOb_f
               Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - Eorbs_b(kk) * &
                                  MO_coef_at_b(jj,kk) * MO_coef_at_b(jj,kk)
            enddo

            do ii = MTB+jj+1, MTB+M
               kkk = kkk +1
               do kk = MTB+1, NCOa_f
                  Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - 2.0D0 * Eorbs(kk) * &
                                     MO_coef_at(ii,kk) * MO_coef_at(jj,kk)
               enddo
               do kk = MTB+1, NCOb_f
                  Pmat_en_wgt(kkk) = Pmat_en_wgt(kkk) - 2.0D0 * Eorbs_b(kk) * &
                                     MO_coef_at_b(ii,kk) * MO_coef_at_b(jj,kk)
               enddo
            enddo
         enddo
      endif

      call g2g_timer_sum_stop('energy-weighted density')


   if (gaussian_convert) then       ! Density matrix translation from Gaussian09
      allocate(rho_exc(M,M))
      call translation(M,rho_exc)   ! Reorganizes Rho to LIO format.

      do jj=1,M                     ! Stores matrix in vector form.
         Pmat_vec(jj + (2*M-jj)*(jj-1)/2) = rho_exc(jj,jj)
         do kk = jj+1, M
            Pmat_vec( kk + (2*M-jj)*(jj-1)/2) = rho_exc(jj,kk) * 2.0D0
         enddo
      enddo

      deallocate(rho_exc)
   endif                            ! End of translation

!  Excited States and TSH routines
   if ( FSTSH ) then
      call TSHmain(MO_coef_at,Eorbs,E)
   else
      call ExcProp(E, MO_coef_at, Eorbs)
   endif

!------------------------------------------------------------------------------!
! TODO: have ehrendyn call SCF and have SCF always save the resulting rho in
!       a module so that ehrendyn can retrieve it afterwards.
!       Remove all of this.
!
      if (doing_ehrenfest) then
         call spunpack('L',M,Pmat_vec,RealRho)
         call fix_densmat(RealRho)
         call ehrendyn_init(natom, M, RealRho)
      endif


!------------------------------------------------------------------------------!
! TODO: Deallocation of variables that should be removed
! TODO: MEMO should be handled differently...

      if (MEMO) then
        ! Release the GPU-resident copy (and host pinning) of cool/cools
        ! before freeing them (see g2g/cuda/coulomb_fit.cu).
        call int3lu_gpu_invalidate()
        deallocate(kkind,kkinds)
        deallocate(cool,cools)
      endif
!------------------------------------------------------------------------------!
! MovieMaker
!     Skip entirely when movie output is off (movie_nfreq == 0, the default):
!     movieprint() early-returns in that case, so the spunpack/fix_densmat and
!     the dcmplx() temporary were pure waste. The complex copy is also kept on
!     the heap (named allocatable, not an expression temporary) so it does not
!     overflow the stack on large-M systems under -fstack-arrays.
      if (movie_nfreq /= 0) then
         call spunpack('L',M,Pmat_vec,RealRho)
         call fix_densmat(RealRho)
         allocate(rho_movie(M,M))
         rho_movie = dcmplx(RealRho)
         call movieprint( natom, M, npas-1, Iz, r, rho_movie )
         deallocate(rho_movie)
      endif

      call Xmat%destroy()
      call Ymat%destroy()

      call g2g_timer_stop('SCF')
      call g2g_timer_sum_stop('Finalize SCF')
      call g2g_timer_sum_stop('SCF')
      call g2g_timer_stop('SCF_full')
      end subroutine SCF
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
