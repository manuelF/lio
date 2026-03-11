program test_propagators
    use propagators, only: magnus
    implicit none

    integer, parameter :: M = 2, N = 4
    real*8 :: Fock(M,M), dt, factorial(N)
    complex :: RhoOld(M,M), RhoNew(M,M)
    integer :: nfail
    real*8 :: criteria, trace_old, trace_new

    nfail = 0
    criteria = 1.0d-6  ! single-precision complex (TD_SIMPLE)
    write(*,*) '--- Testing propagators ---'

    ! Initializations
    Fock = 0.0d0
    Fock(1,1) = 1.0d0 ; Fock(2,2) = 2.0d0

    RhoOld = (0.0d0, 0.0d0)
    RhoOld(1,1) = (1.0d0, 0.0d0) ; RhoOld(2,2) = (1.0d0, 0.0d0)

    dt = 0.01d0

    ! Factorial array for Magnus (1/k!)
    factorial(1) = 1.0d0
    factorial(2) = 1.0d0 / 2.0d0
    factorial(3) = 1.0d0 / 6.0d0
    factorial(4) = 1.0d0 / 24.0d0

    ! 1. Test magnus with commuting matrices
    ! Fock and RhoOld are both diagonal, so [F, rho] = 0
    ! Magnus propagation should leave rho unchanged
    call magnus(Fock, RhoOld, RhoNew, M, N, dt, factorial)

    if (all(abs(RhoNew - RhoOld) < criteria)) then
        write(*,*) 'PASSED - Magnus with commuting matrices preserves rho.'
    else
        write(*,*) 'FAILED - Magnus with commuting matrices. Got:', RhoNew
        nfail = nfail + 1
    end if

    ! 2. Test magnus with non-commuting matrices
    ! Add off-diagonal terms so [F, rho] /= 0
    RhoOld(1,2) = (0.5d0, 0.0d0)
    RhoOld(2,1) = (0.5d0, 0.0d0)

    call magnus(Fock, RhoOld, RhoNew, M, N, dt, factorial)
    if (abs(RhoNew(1,2) - RhoOld(1,2)) > criteria) then
        write(*,*) 'PASSED - Magnus updates non-commuting matrices.'
    else
        write(*,*) 'FAILED - Magnus did not update non-commuting matrices.'
        nfail = nfail + 1
    end if

    ! 3. Test trace conservation: Tr(rho_new) == Tr(rho_old)
    ! Magnus propagation is unitary, so trace must be preserved
    trace_old = real(RhoOld(1,1)) + real(RhoOld(2,2))
    trace_new = real(RhoNew(1,1)) + real(RhoNew(2,2))
    if (abs(trace_new - trace_old) < 1.0d-10) then
        write(*,*) 'PASSED - Magnus preserves trace.'
    else
        write(*,*) 'FAILED - Trace not conserved. Old:', trace_old, ' New:', trace_new
        nfail = nfail + 1
    end if

    ! 4. Test Hermiticity: rho_new(i,j) == conj(rho_new(j,i))
    if (abs(RhoNew(1,2) - conjg(RhoNew(2,1))) < 1.0d-10) then
        write(*,*) 'PASSED - Magnus preserves Hermiticity.'
    else
        write(*,*) 'FAILED - Hermiticity broken. (1,2)=', RhoNew(1,2), &
                   ' conj(2,1)=', conjg(RhoNew(2,1))
        nfail = nfail + 1
    end if

    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All propagator tests passed.'
    end if

end program test_propagators
