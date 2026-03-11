program test_converger
    use converger_subs, only: converger_init, conver
    use converger_data, only: ndiis, damping_factor, hagodiis
    use typedef_operator, only: operator
    implicit none
    
    integer :: M, n_diis, i
    real*8  :: damp, good, good_cut
    type(operator) :: rho_op, fock_op
    real*8, allocatable :: Xmat(:,:), Ymat(:,:), Dmat(:,:)
    
    write(*,*) '--- Testing converger_subs ---'
    
    M = 2
    n_diis = 3
    damp = 0.5d0
    
    ! 1. Test converger_init
    write(*,*) 'Testing converger_init...'
    call converger_init(M, n_diis, damp, .true., .false., .false.)
    
    if (ndiis == n_diis .and. damping_factor == damp) then
        write(*,*) 'PASSED: converger_init correctly set values.'
    else
        write(*,*) 'FAILED: converger_init values mismatch.'
    end if

    ! 2. Test conver (Damping phase)
    write(*,*) 'Testing conver (Damping phase)...'
    allocate(Xmat(M,M), Ymat(M,M), Dmat(M,M))
    Xmat = 0.0d0 ; do i=1,M ; Xmat(i,i) = 1.0d0 ; end do ! Identity for base change
    Ymat = Xmat
    
    Dmat = 0.0d0
    Dmat(1,1) = 1.0d0 ; Dmat(2,2) = 1.0d0
    call rho_op%Sets_data_AO(Dmat)
    
    Dmat(1,1) = 10.0d0 ; Dmat(2,2) = 10.0d0
    call fock_op%Sets_data_AO(Dmat)
    
    good = 1.0d0
    good_cut = 0.1d0
    
    ! Iteration 1: No damping yet
    call conver(1, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    write(*,*) 'Iteration 1 done.'
    
    ! Iteration 2: Damping should occur
    Dmat(1,1) = 20.0d0 ; Dmat(2,2) = 20.0d0
    call fock_op%Sets_data_AO(Dmat)
    call conver(2, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    
    call fock_op%Gets_data_AO(Dmat)
    ! Expected: (20 + 0.5 * 10) / 1.5 = 25 / 1.5 = 16.666...
    if (abs(Dmat(1,1) - 16.6666666666666d0) < 1.0d-8) then
        write(*,*) 'PASSED: Damping correctly applied.'
    else
        write(*,*) 'FAILED: Damping value mismatch. Got ', Dmat(1,1)
    end if

    ! 3. Test DIIS activation
    write(*,*) 'Testing DIIS activation...'
    ! Force DIIS activation by setting good < good_cut or niter > 2 depending on criteria
    ! With conver_criter = 2 (default if do_diis=.true.), it activates at niter > 2
    
    call conver(3, good, good_cut, M, rho_op, fock_op, Xmat, Ymat, 1)
    if (hagodiis) then
        write(*,*) 'PASSED: DIIS activated at iteration 3.'
    else
        write(*,*) 'FAILED: DIIS NOT activated at iteration 3.'
    end if

    deallocate(Xmat, Ymat, Dmat)
    
end program test_converger
