!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
include "../../../lioamber/properties.f90"
program test_properties
    implicit none
    integer :: nfail

    nfail = 0

    ! Electronic Population Analysis.     [EPA]
    call test_lowdin(nfail)
    call test_mulliken(nfail)

    ! Orbital energy related functions.   [OEF]
    call test_degeneration(nfail)
    call test_softness(nfail)

    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All properties tests passed.'
    end if

end program test_properties


!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% Electronic Population Analysis.                                    [EPA] %%!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
subroutine test_mulliken(nfail)
    implicit none
    integer, intent(inout) :: nfail
    real*8       :: Rho(2,2), S(2,2), outVec(2), criteria
    integer      :: atomOrb(2)

    write(*,*) '- Mulliken Population Tests -'
    atomOrb(1) = 1
    atomOrb(2) = 2
    criteria   = 1.0d-9

    ! Test 1: null S matrix.
    Rho    = 1.0d0
    S      = 0.0d0
    outVec = 0.0d0
    call mulliken_calc(2, 2, Rho, S, atomOrb, outVec)
    if (abs(outVec(1)) < criteria .and. abs(outVec(2)) < criteria) then
        write(*,*) 'PASSED - Null S gives null populations.'
    else
        write(*,*) 'FAILED - Null S gives null populations.'
        nfail = nfail + 1
    end if

    ! Test 2: null Rho matrix.
    Rho    = 0.0d0
    S      = 1.0d0
    outVec = 0.0d0
    call mulliken_calc(2, 2, Rho, S, atomOrb, outVec)
    if (abs(outVec(1)) < criteria .and. abs(outVec(2)) < criteria) then
        write(*,*) 'PASSED - Null Rho gives null populations.'
    else
        write(*,*) 'FAILED - Null Rho gives null populations.'
        nfail = nfail + 1
    end if

    ! Test 3: diagonal Rho, non-trivial S.
    Rho(1,1) = 1.0d0 ; Rho(1,2) = 0.0d0
    Rho(2,1) = 0.0d0 ; Rho(2,2) = 1.0d0
    S(1,1)   = 2.0d0 ; S(1,2)   = 2.0d0
    S(2,1)   = 1.0d0 ; S(2,2)   = 1.0d0
    outVec   = 0.0d0
    call mulliken_calc(2, 2, Rho, S, atomOrb, outVec)
    if (abs(outVec(1)+2.0d0) < criteria .and. abs(outVec(2)+1.0d0) < criteria) then
        write(*,*) 'PASSED - Diagonal Rho with non-trivial S.'
    else
        write(*,*) 'FAILED - Diagonal Rho with non-trivial S. Got:', outVec
        nfail = nfail + 1
    end if

    ! Test 4: non-symmetric Rho with negative element.
    Rho(1,1) = 2.0d0 ; Rho(1,2) = 1.0d0
    Rho(2,1) =-1.0d0 ; Rho(2,2) = 3.0d0
    S(1,1)   = 2.0d0 ; S(1,2)   = 2.0d0
    S(2,1)   = 1.0d0 ; S(2,2)   = 1.0d0
    outVec   = 0.0d0
    call mulliken_calc(2, 2, Rho, S, atomOrb, outVec)
    if (abs(outVec(1)+6.0d0) < criteria .and. abs(outVec(2)+2.0d0) < criteria) then
        write(*,*) 'PASSED - Non-symmetric Rho with negative element.'
    else
        write(*,*) 'FAILED - Non-symmetric Rho with negative element. Got:', outVec
        nfail = nfail + 1
    end if

end subroutine test_mulliken

subroutine test_lowdin(nfail)
    implicit none
    integer, intent(inout) :: nfail
    real*8       :: Rho(2,2), SQS(2,2), outVec(2), criteria
    integer      :: atomOrb(2)

    write(*,*) '- Lowdin Population Tests -'
    atomOrb(1) = 1
    atomOrb(2) = 2
    criteria   = 1.0d-9

    ! Test 1: null SQS matrix.
    Rho    = 1.0d0
    SQS    = 0.0d0
    outVec = 0.0d0
    call lowdin_calc(2, 2, Rho, SQS, atomOrb, outVec)
    if (abs(outVec(1)) < criteria .and. abs(outVec(2)) < criteria) then
        write(*,*) 'PASSED - Null SQS gives null populations.'
    else
        write(*,*) 'FAILED - Null SQS gives null populations.'
        nfail = nfail + 1
    end if

    ! Test 2: null Rho matrix.
    Rho    = 0.0d0
    SQS    = 1.0d0
    outVec = 0.0d0
    call lowdin_calc(2, 2, Rho, SQS, atomOrb, outVec)
    if (abs(outVec(1)) < criteria .and. abs(outVec(2)) < criteria) then
        write(*,*) 'PASSED - Null Rho gives null populations.'
    else
        write(*,*) 'FAILED - Null Rho gives null populations.'
        nfail = nfail + 1
    end if

    ! Test 3: diagonal Rho, non-trivial SQS.
    Rho(1,1) = 1.0d0 ; Rho(1,2) = 0.0d0
    Rho(2,1) = 0.0d0 ; Rho(2,2) = 1.0d0
    SQS(1,1) = 2.0d0 ; SQS(1,2) = 2.0d0
    SQS(2,1) = 1.0d0 ; SQS(2,2) = 1.0d0
    outVec   = 0.0d0
    call lowdin_calc(2, 2, Rho, SQS, atomOrb, outVec)
    if (abs(outVec(1)+6.0d0) < criteria .and. abs(outVec(2)+3.0d0) < criteria) then
        write(*,*) 'PASSED - Diagonal Rho with non-trivial SQS.'
    else
        write(*,*) 'FAILED - Diagonal Rho with non-trivial SQS. Got:', outVec
        nfail = nfail + 1
    end if

    ! Test 4: non-symmetric Rho with negative element.
    Rho(1,1) = 2.0d0 ; Rho(1,2) = 1.0d0
    Rho(2,1) =-1.0d0 ; Rho(2,2) = 3.0d0
    SQS(1,1) = 2.0d0 ; SQS(1,2) = 2.0d0
    SQS(2,1) = 1.0d0 ; SQS(2,2) = 1.0d0
    outVec   = 0.0d0
    call lowdin_calc(2, 2, Rho, SQS, atomOrb, outVec)
    if (abs(outVec(1)+12.0d0) < criteria .and. abs(outVec(2)+6.0d0) < criteria) then
        write(*,*) 'PASSED - Non-symmetric Rho with negative element.'
    else
        write(*,*) 'FAILED - Non-symmetric Rho with negative element. Got:', outVec
        nfail = nfail + 1
    end if

end subroutine test_lowdin

subroutine test_degeneration(nfail)
    implicit none
    integer, intent(inout) :: nfail
    integer              :: M, nDeg, nOrb
    integer, allocatable :: nDegMO(:)
    real*8 , allocatable :: energies(:)

    write(*,*) '- Get Degeneration Tests -'

    ! Test 1: single orbital.
    M = 1
    allocate(energies(M), nDegMO(M))
    nOrb     = 1
    energies = 1.0d0
    nDeg     = 0
    nDegMO   = 0
    call get_degeneration(energies, nOrb, M, nDeg, nDegMO)
    if (nDeg == 1 .and. nDegMO(1) == 1) then
        write(*,*) 'PASSED - Single orbital degeneration.'
    else
        write(*,*) 'FAILED - Single orbital degeneration. nDeg=', nDeg
        nfail = nfail + 1
    end if
    deallocate(energies, nDegMO)

    ! Test 2: similar but not equal energies (within threshold).
    M = 10
    allocate(energies(M), nDegMO(M))
    energies    = 1.0d0
    energies(1) = 2.0d0
    energies(2) = 2.0d0
    energies(3) = 2.000011d0
    nDeg   = 0
    nDegMO = 0
    nOrb   = 3
    call get_degeneration(energies, nOrb, M, nDeg, nDegMO)
    if (nDeg == 1 .and. nDegMO(1) == 3) then
        write(*,*) 'PASSED - Nearly degenerate energies.'
    else
        write(*,*) 'FAILED - Nearly degenerate energies. nDeg=', nDeg
        nfail = nfail + 1
    end if

    ! Test 3: two degenerate groups.
    nOrb = 1
    nDeg = 0
    nDegMO = 0
    call get_degeneration(energies, nOrb, M, nDeg, nDegMO)
    if (nDeg == 2 .and. nDegMO(1) == 2 .and. nDegMO(2) == 1) then
        write(*,*) 'PASSED - Two degenerate groups.'
    else
        write(*,*) 'FAILED - Two degenerate groups. nDeg=', nDeg
        nfail = nfail + 1
    end if

    ! Test 4: many degenerate orbitals.
    nOrb = 4
    nDeg = 0
    nDegMO = 0
    call get_degeneration(energies, nOrb, M, nDeg, nDegMO)
    if (nDeg == 7 .and. nDegMO(1) == 4 .and. nDegMO(7) == 10) then
        write(*,*) 'PASSED - Many degenerate orbitals.'
    else
        write(*,*) 'FAILED - Many degenerate orbitals. nDeg=', nDeg
        nfail = nfail + 1
    end if
    deallocate(energies, nDegMO)
end subroutine test_degeneration

subroutine test_softness(nfail)
    implicit none
    integer, intent(inout) :: nfail
    real*8 :: enAH, enAL, enBH, enBL, soft, criteria

    write(*,*) '- Get Softness -'

    enAH = -1.0d0 ; enBH = -2.0d0
    enAL =  3.0d0 ; enBL =  4.0d0
    soft =  0.0d0
    criteria = 1.0d-9

    ! softness = 4 / (enAH + enBH - enAL - enBL) = 4 / (-1-2-3-4) = 4/(-10) = -0.4
    call get_softness(enAH, enAL, enBH, enBL, soft)
    if (abs(soft - (-0.4d0)) < criteria) then
        write(*,*) 'PASSED - Softness correctly calculated.'
    else
        write(*,*) 'FAILED - Softness. Expected -0.4, got:', soft
        nfail = nfail + 1
    end if
end subroutine test_softness
