program test_converger
    use converger_subs, only: converger_init, conver
    use converger_data, only: ndiis, damping_factor, hagodiis
    use typedef_operator, only: operator
    implicit none

    integer :: M, n_diis, i
    real*8  :: damp, good, good_cut
    type(operator) :: rho_op, fock_op
    real*8, allocatable :: Xmat(:,:), Ymat(:,:), Dmat(:,:)
    integer :: nfail
    real*8  :: criteria

    nfail = 0
    criteria = 1.0d-8
    write(*,*) '--- Testing converger_subs ---'

    M = 2
    n_diis = 3
    damp = 0.5d0

    ! 1. Test converger_init
    call converger_init(M, n_diis, damp, .true., .false., .false.)

    if (ndiis == n_diis .and. abs(damping_factor - damp) < criteria) then
        write(*,*) 'PASSED - converger_init correctly set values.'
    else
        write(*,*) 'FAILED - converger_init values mismatch. ndiis=', ndiis, &
                   ' damping_factor=', damping_factor
        nfail = nfail + 1
    end if

    ! 2. Test conver (Damping phase)
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

    ! 3. Test DIIS activation
    ! With conver_criter = 2 (default when do_diis=.true.), DIIS activates at niter > 2
    call conver(3, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    if (hagodiis) then
        write(*,*) 'PASSED - DIIS activated at iteration 3.'
    else
        write(*,*) 'FAILED - DIIS NOT activated at iteration 3.'
        nfail = nfail + 1
    end if

    deallocate(Xmat, Ymat, Dmat)

    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All converger tests passed.'
    end if

end program test_converger
