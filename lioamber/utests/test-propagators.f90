program test_propagators
    use propagators, only: magnus
    implicit none
    
    integer, parameter :: M = 2, N = 4
    real*8 :: Fock(M,M), dt, factorial(N)
    complex :: RhoOld(M,M), RhoNew(M,M)
    integer :: i
    real*8 :: criteria
    
    criteria = 1.0d-12
    write(*,*) '--- Testing propagators ---'
    
    ! Initializations
    Fock = 0.0d0
    Fock(1,1) = 1.0d0 ; Fock(2,2) = 2.0d0
    
    RhoOld = 0.0d0
    RhoOld(1,1) = (1.0d0, 0.0d0) ; RhoOld(2,2) = (1.0d0, 0.0d0)
    
    dt = 0.01d0
    
    ! Factorial array for Magnus (1/k!)
    factorial(1) = 1.0d0
    factorial(2) = 1.0d0 / 2.0d0
    factorial(3) = 1.0d0 / 6.0d0
    factorial(4) = 1.0d0 / 24.0d0
    
    ! 1. Test magnus
    write(*,*) 'Testing magnus...'
    ! Since Fock and RhoOld commute (both diagonal), ConmNext should be 0, and RhoNew should equal RhoOld
    call magnus(Fock, RhoOld, RhoNew, M, N, dt, factorial)
    
    if (all(abs(RhoNew - RhoOld) < criteria)) then
        write(*,*) 'PASSED: Magnus with commuting matrices.'
    else
        write(*,*) 'FAILED: Magnus with commuting matrices.'
        print *, 'Got:', RhoNew
    end if

    ! 2. Test magnus with non-commuting matrices
    RhoOld(1,2) = (0.5d0, 0.0d0)
    RhoOld(2,1) = (0.5d0, 0.0d0)
    
    call magnus(Fock, RhoOld, RhoNew, M, N, dt, factorial)
    if (abs(RhoNew(1,1) - RhoOld(1,1)) > criteria .or. abs(RhoNew(1,2) - RhoOld(1,2)) > criteria) then
         write(*,*) 'PASSED: Magnus correctly updated non-commuting matrices.'
    else
         write(*,*) 'FAILED: Magnus did not update non-commuting matrices.'
    end if

end program test_propagators
