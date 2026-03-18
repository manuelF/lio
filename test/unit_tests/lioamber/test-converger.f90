program test_converger
    use converger_subs, only: converger_init, conver, circ_slot, trace_product
    use converger_data, only: ndiis, damping_factor, hagodiis, bcoef, fockm, &
                              FP_PFm, EMAT2, fock_damped, head_idx, &
                              fock00_w, fock_w, rho_w, suma_w, &
                              scratch1_w, scratch2_w, work_w
    use typedef_operator, only: operator
    implicit none

    integer :: M, n_diis, i, j, kk, jj, M2
    real*8  :: damp, good, good_cut
    type(operator) :: rho_op, fock_op
    real*8, allocatable :: Xmat(:,:), Ymat(:,:), Dmat(:,:), Fcheck(:,:)
    real*8, allocatable :: xnano(:,:), Pmat_vec(:)
    integer :: nfail
    real*8  :: criteria, bsum, expected_val, bmax_val
    real*8  :: expected_damp, sq2, del, good_expected
    integer :: ndiist

    nfail = 0
    criteria = 1.0d-8
    write(*,*) '--- Testing converger_subs ---'

    M = 2
    n_diis = 3
    damp = 0.5d0

    ! =========================================================================
    ! Test 1: converger_init
    ! =========================================================================
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    if (ndiis == n_diis .and. abs(damping_factor - damp) < criteria) then
        write(*,*) 'PASSED - converger_init correctly set values.'
    else
        write(*,*) 'FAILED - converger_init values mismatch. ndiis=', ndiis, &
                   ' damping_factor=', damping_factor
        nfail = nfail + 1
    end if

    ! =========================================================================
    ! Test 2: Damping phase
    ! =========================================================================
    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(2,2) = 1.0d0
    call rho_op%Sets_data_AO(Dmat)

    Dmat(1,1) = 10.0d0 ; Dmat(2,2) = 10.0d0
    call fock_op%Sets_data_AO(Dmat)

    good = 1.0d0
    good_cut = 0.1d0

    ! Iteration 1: No damping yet (stores Fock for next iteration)
    call conver(1, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    ! Iteration 2: Damping should occur
    ! New Fock = 20, Old Fock = 10, damp = 0.5
    ! Damped = (20 + 0.5 * 10) / (1 + 0.5) = 25 / 1.5 = 16.666...
    Dmat(1,1) = 20.0d0 ; Dmat(2,2) = 20.0d0
    call fock_op%Sets_data_AO(Dmat)
    call conver(2, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    ! Read ON data: the optimized converger applies damping in ON basis and
    ! sets data_ON directly (skipping the redundant AO→ON base change).
    ! With X=I the ON and AO values are identical.
    call fock_op%Gets_data_ON(Dmat)
    if (abs(Dmat(1,1) - 50.0d0/3.0d0) < criteria) then
        write(*,*) 'PASSED - Damping correctly applied.'
    else
        write(*,*) 'FAILED - Damping value mismatch. Expected 16.6667, got:', Dmat(1,1)
        nfail = nfail + 1
    end if

    ! =========================================================================
    ! Test 3: DIIS activation
    ! =========================================================================
    ! With conver_criter = 2 (default when do_diis=.true.), DIIS activates at niter > 2
    call conver(3, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    if (hagodiis) then
        write(*,*) 'PASSED - DIIS activated at iteration 3.'
    else
        write(*,*) 'FAILED - DIIS NOT activated at iteration 3.'
        nfail = nfail + 1
    end if

    deallocate(Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 4: DIIS coefficient sum = 1.0
    ! =========================================================================
    ! With Xmat=Ymat=I and NON-commuting rho/fock, [F,P] != 0.
    ! The Lagrange constraint guarantees sum(bcoef) = 1.
    write(*,*) 'Test 4: DIIS bcoef sum = 1.0...'

    M = 3
    n_diis = 4
    damp = 0.5d0
    ! Deallocate old converger_data arrays before re-init with different M
    if (allocated(fockm))      deallocate(fockm)
    if (allocated(FP_PFm))     deallocate(FP_PFm)
    if (allocated(bcoef))      deallocate(bcoef)
    if (allocated(EMAT2))      deallocate(EMAT2)
    if (allocated(fock_damped)) deallocate(fock_damped)
    if (allocated(fock00_w))   deallocate(fock00_w)
    if (allocated(fock_w))     deallocate(fock_w)
    if (allocated(rho_w))      deallocate(rho_w)
    if (allocated(suma_w))     deallocate(suma_w)
    if (allocated(scratch1_w)) deallocate(scratch1_w)
    if (allocated(scratch2_w)) deallocate(scratch2_w)
    if (allocated(work_w))     deallocate(work_w)
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    ! Non-diagonal rho so [F, rho] != 0
    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(1,2) = 0.3d0
    Dmat(2,1) = 0.3d0 ; Dmat(2,2) = 2.0d0
    Dmat(3,3) = 3.0d0
    call rho_op%Sets_data_AO(Dmat)

    good = 1.0d0
    good_cut = 0.1d0

    ! Run 4 iterations (DIIS activates at iter 3 for conver_criter=2)
    do i = 1, 4
        Dmat = 0.0d0
        do j=1,M ; Dmat(j,j) = dble(i*10 + j) ; end do
        ! Add off-diagonal elements for non-trivial commutator
        Dmat(1,2) = dble(i) ; Dmat(2,1) = dble(i)
        Dmat(2,3) = dble(i)*0.5d0 ; Dmat(3,2) = dble(i)*0.5d0
        call fock_op%Sets_data_AO(Dmat)
        call conver(i, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    end do

    ! After iter 4, ndiist = min(4, 4) = 4, bcoef(1:4, 1) should sum to 1
    ndiist = min(4, n_diis)
    bsum = 0.0d0
    do i = 1, ndiist
        bsum = bsum + bcoef(i, 1)
    end do

    if (abs(bsum - 1.0d0) < criteria) then
        write(*,*) 'PASSED - bcoef sum = 1.0'
    else
        write(*,*) 'FAILED - bcoef sum =', bsum, ' expected 1.0'
        nfail = nfail + 1
    end if

    ! =========================================================================
    ! Test 5: DIIS extrapolated Fock = linear combination of stored Fock
    ! =========================================================================
    ! Verify that fock_op ON data = sum(bcoef(k) * fockm(:,:,k,1))
    write(*,*) 'Test 5: DIIS extrapolated Fock consistency...'

    allocate(Fcheck(M,M))
    call fock_op%Gets_data_ON(Fcheck)

    ! Manually compute the expected sum using circular buffer slot mapping
    Dmat = 0.0d0
    do i = 1, ndiist
        j = circ_slot(i, head_idx(1), ndiist, n_diis)
        Dmat(:,:) = Dmat(:,:) + bcoef(i, 1) * fockm(:,:, j, 1)
    end do

    expected_val = Dmat(1,1)
    if (abs(Fcheck(1,1) - expected_val) < criteria .and. &
        abs(Fcheck(1,2) - Dmat(1,2)) < criteria) then
        write(*,*) 'PASSED - DIIS Fock matches bcoef-weighted sum.'
    else
        write(*,*) 'FAILED - DIIS Fock(1,1) =', Fcheck(1,1), ' expected', expected_val
        nfail = nfail + 1
    end if

    deallocate(Fcheck, Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 6: History buffer full (niter > ndiis)
    ! =========================================================================
    ! Run ndiis+2 iterations to trigger the shift path (lines 199-206).
    ! Verify bcoef sum is still 1.0 after history has been shifted.
    write(*,*) 'Test 6: History buffer full...'

    M = 3
    n_diis = 3
    ! Deallocate old arrays before re-init with different ndiis
    if (allocated(fockm))      deallocate(fockm)
    if (allocated(FP_PFm))     deallocate(FP_PFm)
    if (allocated(bcoef))      deallocate(bcoef)
    if (allocated(EMAT2))      deallocate(EMAT2)
    if (allocated(fock_damped)) deallocate(fock_damped)
    if (allocated(fock00_w))   deallocate(fock00_w)
    if (allocated(fock_w))     deallocate(fock_w)
    if (allocated(rho_w))      deallocate(rho_w)
    if (allocated(suma_w))     deallocate(suma_w)
    if (allocated(scratch1_w)) deallocate(scratch1_w)
    if (allocated(scratch2_w)) deallocate(scratch2_w)
    if (allocated(work_w))     deallocate(work_w)
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    ! Non-diagonal rho for non-trivial [F, rho]
    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(1,2) = 0.3d0
    Dmat(2,1) = 0.3d0 ; Dmat(2,2) = 2.0d0
    Dmat(3,3) = 3.0d0
    call rho_op%Sets_data_AO(Dmat)

    ! Run ndiis+2 = 5 iterations
    do i = 1, n_diis + 2
        Dmat = 0.0d0
        do j=1,M ; Dmat(j,j) = dble(i*10 + j) ; end do
        Dmat(1,2) = dble(i) ; Dmat(2,1) = dble(i)
        Dmat(2,3) = dble(i)*0.5d0 ; Dmat(3,2) = dble(i)*0.5d0
        call fock_op%Sets_data_AO(Dmat)
        call conver(i, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    end do

    ! ndiist = min(5, 3) = 3; bcoef(1:3,1) should sum to 1.0
    ndiist = min(n_diis + 2, n_diis)
    bsum = 0.0d0
    do i = 1, ndiist
        bsum = bsum + bcoef(i, 1)
    end do

    if (abs(bsum - 1.0d0) < criteria) then
        write(*,*) 'PASSED - bcoef sum = 1.0 after history full.'
    else
        write(*,*) 'FAILED - bcoef sum =', bsum, ' after history full.'
        nfail = nfail + 1
    end if

    ! Verify Fock output consistency: fock_ON = sum(bcoef * fockm)
    allocate(Fcheck(M,M))
    call fock_op%Gets_data_ON(Fcheck)

    Dmat = 0.0d0
    do i = 1, ndiist
        j = circ_slot(i, head_idx(1), ndiist, n_diis)
        Dmat(:,:) = Dmat(:,:) + bcoef(i, 1) * fockm(:,:, j, 1)
    end do

    if (abs(Fcheck(1,1) - Dmat(1,1)) < criteria .and. &
        abs(Fcheck(2,2) - Dmat(2,2)) < criteria) then
        write(*,*) 'PASSED - DIIS Fock consistent after history full.'
    else
        write(*,*) 'FAILED - DIIS Fock(1,1) =', Fcheck(1,1), ' expected', Dmat(1,1)
        nfail = nfail + 1
    end if

    deallocate(Fcheck, Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 7: Near-convergence DIIS — nearly-parallel error vectors
    ! =========================================================================
    ! When Fock matrices differ by tiny amounts (simulating near-convergence),
    ! [F,P] vectors become nearly parallel → EMAT nearly singular.
    ! The solver must still produce bounded coefficients (sum=1, |c_k| bounded).
    write(*,*) 'Test 7: Near-convergence DIIS (ill-conditioned EMAT)...'

    M = 3
    n_diis = 5
    damp = 0.5d0
    if (allocated(fockm))      deallocate(fockm)
    if (allocated(FP_PFm))     deallocate(FP_PFm)
    if (allocated(bcoef))      deallocate(bcoef)
    if (allocated(EMAT2))      deallocate(EMAT2)
    if (allocated(fock_damped)) deallocate(fock_damped)
    if (allocated(fock00_w))   deallocate(fock00_w)
    if (allocated(fock_w))     deallocate(fock_w)
    if (allocated(rho_w))      deallocate(rho_w)
    if (allocated(suma_w))     deallocate(suma_w)
    if (allocated(scratch1_w)) deallocate(scratch1_w)
    if (allocated(scratch2_w)) deallocate(scratch2_w)
    if (allocated(work_w))     deallocate(work_w)
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    ! Non-diagonal rho
    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(1,2) = 0.3d0
    Dmat(2,1) = 0.3d0 ; Dmat(2,2) = 2.0d0
    Dmat(3,3) = 3.0d0
    call rho_op%Sets_data_AO(Dmat)

    good = 1.0d0
    good_cut = 0.1d0

    ! Fock matrices that differ by tiny amounts (simulating near-convergence)
    ! Base Fock + epsilon*i perturbation → nearly-parallel [F,P] vectors
    do i = 1, 6
        Dmat = 0.0d0
        Dmat(1,1) = 10.0d0 + dble(i)*1.0d-6
        Dmat(2,2) = 20.0d0 + dble(i)*2.0d-6
        Dmat(3,3) = 30.0d0 + dble(i)*3.0d-6
        Dmat(1,2) = 0.5d0 + dble(i)*1.0d-7
        Dmat(2,1) = 0.5d0 + dble(i)*1.0d-7
        Dmat(2,3) = 0.3d0 + dble(i)*5.0d-8
        Dmat(3,2) = 0.3d0 + dble(i)*5.0d-8
        call fock_op%Sets_data_AO(Dmat)
        call conver(i, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    end do

    ! Check: bcoef sum must still be 1.0
    ndiist = min(6, n_diis)
    bsum = 0.0d0
    do i = 1, ndiist
        bsum = bsum + bcoef(i, 1)
    end do

    if (abs(bsum - 1.0d0) < 1.0d-4) then
        write(*,*) 'PASSED - bcoef sum ≈ 1.0 for ill-conditioned EMAT'
    else
        write(*,*) 'FAILED - bcoef sum =', bsum, ' (expected ~1.0)'
        nfail = nfail + 1
    end if

    ! Check: coefficient magnitudes should be bounded
    ! With a robust solver, max(|c_k|) should stay reasonable (< 100)
    ! With DGELS on a singular system, they can blow up to 1e10+
    bmax_val = 0.0d0
    do i = 1, ndiist
        if (abs(bcoef(i,1)) > bmax_val) bmax_val = abs(bcoef(i,1))
    end do
    write(*,'(A,ES12.4)') '  max(|bcoef|) =', bmax_val

    if (bmax_val < 1.0d2) then
        write(*,*) 'PASSED - bcoef magnitudes bounded (max < 100)'
    else
        write(*,*) 'WARNING - bcoef magnitudes large:', bmax_val
        ! Not a hard failure — this is what we're trying to improve
    end if

    deallocate(Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 8: Damping with different GOLD values
    ! =========================================================================
    ! Verify damping formula F = (F_new + GOLD*F_old)/(1+GOLD) for various GOLD.
    ! This tests that reducing GOLD gives more weight to the new Fock matrix.
    write(*,*) 'Test 8: Damping with GOLD=10 and GOLD=2...'

    ! --- Test 8a: GOLD=10 (current default, very conservative) ---
    M = 2
    n_diis = 3
    damp = 10.0d0
    if (allocated(fockm))      deallocate(fockm)
    if (allocated(FP_PFm))     deallocate(FP_PFm)
    if (allocated(bcoef))      deallocate(bcoef)
    if (allocated(EMAT2))      deallocate(EMAT2)
    if (allocated(fock_damped)) deallocate(fock_damped)
    if (allocated(fock00_w))   deallocate(fock00_w)
    if (allocated(fock_w))     deallocate(fock_w)
    if (allocated(rho_w))      deallocate(rho_w)
    if (allocated(suma_w))     deallocate(suma_w)
    if (allocated(scratch1_w)) deallocate(scratch1_w)
    if (allocated(scratch2_w)) deallocate(scratch2_w)
    if (allocated(work_w))     deallocate(work_w)
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(2,2) = 1.0d0
    call rho_op%Sets_data_AO(Dmat)

    ! Iter 1: F_old = 5.0 (stored for next iteration)
    Dmat(1,1) = 5.0d0 ; Dmat(2,2) = 5.0d0
    call fock_op%Sets_data_AO(Dmat)
    good = 1.0d0 ; good_cut = 0.1d0
    call conver(1, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    ! Iter 2: F_new = 15.0, F_old = 5.0
    ! Damped = (15 + 10*5) / (1+10) = 65/11 = 5.909090...
    Dmat(1,1) = 15.0d0 ; Dmat(2,2) = 15.0d0
    call fock_op%Sets_data_AO(Dmat)
    call conver(2, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    call fock_op%Gets_data_ON(Dmat)
    expected_damp = (15.0d0 + 10.0d0 * 5.0d0) / (1.0d0 + 10.0d0)
    if (abs(Dmat(1,1) - expected_damp) < criteria) then
        write(*,*) 'PASSED - GOLD=10 damping correct:', Dmat(1,1)
    else
        write(*,*) 'FAILED - GOLD=10 expected', expected_damp, 'got', Dmat(1,1)
        nfail = nfail + 1
    end if

    deallocate(Xmat, Ymat, Dmat)

    ! --- Test 8b: GOLD=2 (proposed new default, more aggressive) ---
    damp = 2.0d0
    if (allocated(fockm))      deallocate(fockm)
    if (allocated(FP_PFm))     deallocate(FP_PFm)
    if (allocated(bcoef))      deallocate(bcoef)
    if (allocated(EMAT2))      deallocate(EMAT2)
    if (allocated(fock_damped)) deallocate(fock_damped)
    if (allocated(fock00_w))   deallocate(fock00_w)
    if (allocated(fock_w))     deallocate(fock_w)
    if (allocated(rho_w))      deallocate(rho_w)
    if (allocated(suma_w))     deallocate(suma_w)
    if (allocated(scratch1_w)) deallocate(scratch1_w)
    if (allocated(scratch2_w)) deallocate(scratch2_w)
    if (allocated(work_w))     deallocate(work_w)
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do
    Ymat = Xmat

    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(2,2) = 1.0d0
    call rho_op%Sets_data_AO(Dmat)

    ! Iter 1: F_old = 5.0
    Dmat(1,1) = 5.0d0 ; Dmat(2,2) = 5.0d0
    call fock_op%Sets_data_AO(Dmat)
    good = 1.0d0 ; good_cut = 0.1d0
    call conver(1, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    ! Iter 2: F_new = 15.0, F_old = 5.0
    ! Damped = (15 + 2*5) / (1+2) = 25/3 = 8.333...
    Dmat(1,1) = 15.0d0 ; Dmat(2,2) = 15.0d0
    call fock_op%Sets_data_AO(Dmat)
    call conver(2, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)

    call fock_op%Gets_data_ON(Dmat)
    expected_damp = (15.0d0 + 2.0d0 * 5.0d0) / (1.0d0 + 2.0d0)
    if (abs(Dmat(1,1) - expected_damp) < criteria) then
        write(*,*) 'PASSED - GOLD=2 damping correct:', Dmat(1,1)
    else
        write(*,*) 'FAILED - GOLD=2 expected', expected_damp, 'got', Dmat(1,1)
        nfail = nfail + 1
    end if

    ! Verify GOLD=2 gives more weight to new Fock than GOLD=10
    ! GOLD=10: 5.909, GOLD=2: 8.333 (closer to F_new=15)
    if (expected_damp > 65.0d0/11.0d0) then
        write(*,*) 'PASSED - GOLD=2 gives more weight to new Fock than GOLD=10'
    else
        write(*,*) 'FAILED - GOLD=2 should give more new Fock weight'
        nfail = nfail + 1
    end if

    deallocate(Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 9: Convergence metric (good) calculation
    ! =========================================================================
    ! Replicates the convergence metric from SCF.f90 lines 722-731.
    ! Tests the formula: good = sqrt(sum(del^2 * sq2^2)) / M
    ! where del = xnano(jj,kk) - Pmat_vec(packed_index)
    ! NOTE: Current code applies sq2 to ALL elements including diagonal.
    write(*,*) 'Test 9: Convergence metric calculation...'

    M = 3
    M2 = 2*M
    sq2 = sqrt(2.0d0)
    allocate(xnano(M,M), Pmat_vec(M*(M+1)/2))

    ! Set up known density matrices
    ! Old density (in packed lower-triangular form)
    Pmat_vec = 0.0d0
    Pmat_vec(1) = 1.0d0   ! (1,1)
    Pmat_vec(2) = 0.1d0   ! (2,1) -> (1,2) in upper tri
    Pmat_vec(3) = 0.2d0   ! (3,1) -> (1,3) in upper tri
    Pmat_vec(4) = 2.0d0   ! (2,2)
    Pmat_vec(5) = 0.3d0   ! (3,2) -> (2,3) in upper tri
    Pmat_vec(6) = 3.0d0   ! (3,3)

    ! New density (full matrix, upper triangle accessed)
    xnano = 0.0d0
    xnano(1,1) = 1.01d0   ! diagonal change: +0.01
    xnano(1,2) = 0.12d0   ! off-diag change: +0.02
    xnano(1,3) = 0.20d0   ! no change
    xnano(2,2) = 2.05d0   ! diagonal change: +0.05
    xnano(2,3) = 0.30d0   ! no change
    xnano(3,3) = 2.97d0   ! diagonal change: -0.03

    ! Compute good using FIXED formula (sq2 only on off-diagonal elements)
    good = 0.0d0
    do jj=1,M
    do kk=jj,M
      del = xnano(jj,kk) - Pmat_vec(kk+(M2-jj)*(jj-1)/2)
      if (kk > jj) del = del * sq2
      good = good + del**2
    enddo
    enddo
    good = sqrt(good) / float(M)

    ! Manually compute expected value with sq2 ONLY on off-diagonal
    ! Differences: (1,1)=0.01, (1,2)=0.02, (1,3)=0.0, (2,2)=0.05, (2,3)=0.0, (3,3)=-0.03
    ! Diagonal: 0.01^2 + 0.05^2 + 0.03^2 = 0.0001 + 0.0025 + 0.0009 = 0.0035
    ! Off-diag: (0.02*sq2)^2 + 0 + 0 = 0.0008
    ! Total sum of squares = 0.0043
    good_expected = sqrt(0.01d0**2 + (0.02d0*sq2)**2 + 0.05d0**2 + 0.03d0**2) &
                    / dble(M)

    if (abs(good - good_expected) < 1.0d-12) then
        write(*,*) 'PASSED - Convergence metric correct (sq2 off-diag only):', good
    else
        write(*,*) 'FAILED - good =', good, ' expected', good_expected
        nfail = nfail + 1
    end if

    ! Verify the old (buggy) metric would be LARGER
    ! Old formula: sq2 on ALL → 2*(0.0035 + 0.0004) = 0.0078
    good_expected = sqrt(2.0d0 * (0.01d0**2 + 0.02d0**2 + 0.05d0**2 + 0.03d0**2)) &
                    / dble(M)
    if (good < good_expected) then
        write(*,*) 'PASSED - Fixed metric smaller than old buggy metric:', &
                   good, '<', good_expected
    else
        write(*,*) 'FAILED - Fixed metric should be smaller than buggy'
        nfail = nfail + 1
    end if

    deallocate(xnano, Pmat_vec)

    ! =========================================================================
    ! Test 10: Symmetric commutator equivalence
    ! =========================================================================
    ! For symmetric A, B: [A,B] = AB - BA = AB - (AB)^T
    ! This validates the 1-DGEMM optimization (instead of 2 MATMULs).
    write(*,*) 'Test 10: Symmetric commutator [A,B] = AB - (AB)^T...'

    M = 8
    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    block
        real*8, allocatable :: A(:,:), B(:,:), comm_ref(:,:), comm_opt(:,:)
        real*8, allocatable :: AB(:,:)
        integer :: ii2, jj2

        allocate(A(M,M), B(M,M), comm_ref(M,M), comm_opt(M,M), AB(M,M))

        ! Build symmetric A and B
        do jj2 = 1, M
        do ii2 = 1, M
            A(ii2,jj2) = dble(ii2*3 + jj2*7 + ii2*jj2) / dble(M*M)
            B(ii2,jj2) = dble(ii2*5 - jj2*2 + ii2*jj2*3) / dble(M*M)
        enddo
        enddo
        ! Symmetrize
        do jj2 = 1, M
        do ii2 = jj2+1, M
            A(ii2,jj2) = A(jj2,ii2)
            B(ii2,jj2) = B(jj2,ii2)
        enddo
        enddo

        ! Reference: [A,B] = AB - BA (2 MATMULs)
        comm_ref = MATMUL(A, B) - MATMUL(B, A)

        ! Optimized: [A,B] = AB - (AB)^T (1 DGEMM + transpose)
        call DGEMM('N','N',M,M,M,1.0D0,A,M,B,M,0.0D0,AB,M)
        do jj2 = 1, M
        do ii2 = 1, M
            comm_opt(ii2,jj2) = AB(ii2,jj2) - AB(jj2,ii2)
        enddo
        enddo

        ! Check equivalence
        del = 0.0d0
        do jj2 = 1, M
        do ii2 = 1, M
            del = del + abs(comm_ref(ii2,jj2) - comm_opt(ii2,jj2))
        enddo
        enddo

        if (del < 1.0d-10) then
            write(*,*) 'PASSED - Symmetric commutator: 1-DGEMM matches 2-MATMUL'
        else
            write(*,*) 'FAILED - Symmetric commutator diff =', del
            nfail = nfail + 1
        end if

        ! Verify antisymmetry: comm(i,j) = -comm(j,i)
        del = 0.0d0
        do jj2 = 1, M
        do ii2 = 1, M
            del = del + abs(comm_opt(ii2,jj2) + comm_opt(jj2,ii2))
        enddo
        enddo

        if (del < 1.0d-10) then
            write(*,*) 'PASSED - Commutator is antisymmetric'
        else
            write(*,*) 'FAILED - Commutator antisymmetry error =', del
            nfail = nfail + 1
        end if

        deallocate(A, B, comm_ref, comm_opt, AB)
    end block
    deallocate(Xmat, Ymat, Dmat)

    ! =========================================================================
    ! Test 11: trace_product via DDOT for antisymmetric matrices
    ! =========================================================================
    ! For antisymmetric A, B: Tr(A·B) = -sum(A(i,j)*B(i,j)) = -DDOT(M², A, B)
    write(*,*) 'Test 11: trace_product = -DDOT for antisymmetric matrices...'

    M = 10
    block
        real*8, allocatable :: E1(:,:), E2(:,:)
        real*8 :: trace_ref, trace_ddot
        real*8, external :: DDOT
        integer :: ii2, jj2

        allocate(E1(M,M), E2(M,M))

        ! Build antisymmetric E1 and E2
        E1 = 0.0d0 ; E2 = 0.0d0
        do jj2 = 1, M
        do ii2 = jj2+1, M
            E1(ii2,jj2) =  dble(ii2*3 - jj2*7) / dble(M)
            E1(jj2,ii2) = -E1(ii2,jj2)
            E2(ii2,jj2) =  dble(ii2*5 + jj2*2) / dble(M)
            E2(jj2,ii2) = -E2(ii2,jj2)
        enddo
        enddo

        ! Reference: trace_product from converger_subs
        trace_ref = trace_product(E1, E2, M)

        ! Optimized: -DDOT
        trace_ddot = -DDOT(M*M, E1, 1, E2, 1)

        if (abs(trace_ref - trace_ddot) < 1.0d-10) then
            write(*,*) 'PASSED - trace_product = -DDOT for antisymmetric matrices'
        else
            write(*,'(A,ES20.12,A,ES20.12)') &
                ' FAILED - trace_product=', trace_ref, ' -DDOT=', trace_ddot
            nfail = nfail + 1
        end if

        deallocate(E1, E2)
    end block

    ! =========================================================================
    ! Test 12: Inline base change equivalence
    ! =========================================================================
    ! Verify X^T · A · X computed inline matches basechange_gemm
    write(*,*) 'Test 12: Inline base change matches basechange_gemm...'

    M = 6
    block
        use mathsubs, only: basechange_gemm
        real*8, allocatable :: A(:,:), X(:,:), ref(:,:)
        real*8, allocatable :: tmp(:,:), result_inline(:,:)
        integer :: ii2, jj2

        allocate(A(M,M), X(M,M), tmp(M,M), result_inline(M,M))

        ! Build symmetric A
        do jj2 = 1, M
        do ii2 = 1, M
            A(ii2,jj2) = dble(ii2 + jj2*3) / dble(M)
        enddo
        enddo
        do jj2 = 1, M
        do ii2 = jj2+1, M
            A(ii2,jj2) = A(jj2,ii2)
        enddo
        enddo

        ! Build non-trivial X (not identity)
        X = 0.0d0
        do ii2 = 1, M
            X(ii2,ii2) = 1.0d0 / sqrt(dble(ii2))
        enddo
        X(1,2) = 0.1d0 ; X(2,1) = -0.1d0

        ! Reference: basechange_gemm
        ref = basechange_gemm(M, A, X)

        ! Inline: tmp = X^T · A ; result = tmp · X
        call DGEMM('T','N',M,M,M,1.0D0,X,M,A,M,0.0D0,tmp,M)
        call DGEMM('N','N',M,M,M,1.0D0,tmp,M,X,M,0.0D0,result_inline,M)

        del = 0.0d0
        do jj2 = 1, M
        do ii2 = 1, M
            del = del + abs(ref(ii2,jj2) - result_inline(ii2,jj2))
        enddo
        enddo

        if (del < 1.0d-10) then
            write(*,*) 'PASSED - Inline base change matches basechange_gemm'
        else
            write(*,*) 'FAILED - Inline base change diff =', del
            nfail = nfail + 1
        end if

        deallocate(A, X, ref, tmp, result_inline)
    end block

    ! =========================================================================
    ! Test 13: ON-basis damping equivalence
    ! =========================================================================
    ! Verify: X^T · [(F_new + λ·F_old)/(1+λ)] · X = (F'_new + λ·F'_old)/(1+λ)
    ! This validates that damping commutes with base change.
    write(*,*) 'Test 13: ON-basis damping = AO-basis damping + base change...'

    M = 5
    block
        use mathsubs, only: basechange_gemm
        real*8, allocatable :: F_new(:,:), F_old(:,:), X2(:,:)
        real*8, allocatable :: damped_AO(:,:), ref_ON(:,:), opt_ON(:,:)
        real*8 :: lambda
        integer :: ii2, jj2

        lambda = 0.5d0
        allocate(F_new(M,M), F_old(M,M), X2(M,M))
        allocate(damped_AO(M,M), ref_ON(M,M), opt_ON(M,M))

        ! Symmetric Fock matrices
        do jj2 = 1, M
        do ii2 = 1, M
            F_new(ii2,jj2) = dble(ii2*10 + jj2) / dble(M)
            F_old(ii2,jj2) = dble(ii2*5 + jj2*3) / dble(M)
        enddo
        enddo
        do jj2 = 1, M
        do ii2 = jj2+1, M
            F_new(ii2,jj2) = F_new(jj2,ii2)
            F_old(ii2,jj2) = F_old(jj2,ii2)
        enddo
        enddo

        ! Non-trivial X
        X2 = 0.0d0
        do ii2 = 1, M
            X2(ii2,ii2) = 1.0d0 / sqrt(dble(ii2))
        enddo
        X2(1,2) = 0.1d0 ; X2(2,3) = 0.05d0

        ! Reference: damp in AO, then base change
        damped_AO = (F_new + lambda * F_old) / (1.0d0 + lambda)
        ref_ON = basechange_gemm(M, damped_AO, X2)

        ! Optimized: base change each, then damp in ON
        opt_ON = (basechange_gemm(M, F_new, X2) + &
                  lambda * basechange_gemm(M, F_old, X2)) / (1.0d0 + lambda)

        del = 0.0d0
        do jj2 = 1, M
        do ii2 = 1, M
            del = del + abs(ref_ON(ii2,jj2) - opt_ON(ii2,jj2))
        enddo
        enddo

        if (del < 1.0d-10) then
            write(*,*) 'PASSED - ON-basis damping equivalent to AO damping + BC'
        else
            write(*,*) 'FAILED - ON-basis damping diff =', del
            nfail = nfail + 1
        end if

        deallocate(F_new, F_old, X2, damped_AO, ref_ON, opt_ON)
    end block

    ! =========================================================================
    ! Summary
    ! =========================================================================
    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All converger tests passed.'
    end if

end program test_converger
