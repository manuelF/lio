program test_converger
    use converger_subs, only: converger_init, conver, circ_slot
    use converger_data, only: ndiis, damping_factor, hagodiis, bcoef, fockm, &
                              FP_PFm, EMAT2, fock_damped, head_idx, &
                              fock00_w, fock_w, rho_w, suma_w, &
                              scratch1_w, scratch2_w, work_w
    use typedef_operator, only: operator
    implicit none

    integer :: M, n_diis, i, j
    real*8  :: damp, good, good_cut
    type(operator) :: rho_op, fock_op
    real*8, allocatable :: Xmat(:,:), Ymat(:,:), Dmat(:,:), Fcheck(:,:)
    integer :: nfail
    real*8  :: criteria, bsum, expected_val, bmax_val
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

    call fock_op%Gets_data_AO(Dmat)
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
