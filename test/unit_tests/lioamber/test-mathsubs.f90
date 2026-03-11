program test_mathsubs
    use mathsubs, only: basechange_d_gemm, basechange_z_gemm
    implicit none

    integer, parameter :: M = 2
    real*8 :: Umat(M,M), Mati_d(M,M)
    complex*16 :: Mati_z(M,M)
    real*8, allocatable :: Mato_d(:,:)
    complex*16, allocatable :: Mato_z(:,:)
    integer :: i
    real*8 :: criteria
    integer :: nfail

    criteria = 1.0d-12
    nfail = 0
    write(*,*) '--- Testing mathsubs ---'

    ! Identity transformation matrix
    Umat = 0.0d0
    do i=1,M
        Umat(i,i) = 1.0d0
    end do

    ! 1. Test basechange_d_gemm with identity
    Mati_d(1,1) = 1.0d0 ; Mati_d(1,2) = 2.0d0
    Mati_d(2,1) = 3.0d0 ; Mati_d(2,2) = 4.0d0
    Mato_d = basechange_d_gemm(M, Mati_d, Umat)
    if (all(abs(Mato_d - Mati_d) < criteria)) then
        write(*,*) 'PASSED - basechange_d_gemm with identity.'
    else
        write(*,*) 'FAILED - basechange_d_gemm with identity. Got:', Mato_d
        nfail = nfail + 1
    end if

    ! 2. Test basechange_z_gemm with identity
    Mati_z(1,1) = cmplx(1.0d0, 0.1d0)
    Mati_z(1,2) = cmplx(2.0d0, 0.2d0)
    Mati_z(2,1) = cmplx(3.0d0, 0.3d0)
    Mati_z(2,2) = cmplx(4.0d0, 0.4d0)
    Mato_z = basechange_z_gemm(M, Mati_z, Umat)
    if (all(abs(Mato_z - Mati_z) < criteria)) then
        write(*,*) 'PASSED - basechange_z_gemm with identity.'
    else
        write(*,*) 'FAILED - basechange_z_gemm with identity. Got:', Mato_z
        nfail = nfail + 1
    end if

    ! 3. Test basechange_d_gemm with swap matrix (non-trivial transform)
    ! Swap matrix: U = [[0,1],[1,0]]
    Umat(1,1) = 0.0d0 ; Umat(1,2) = 1.0d0
    Umat(2,1) = 1.0d0 ; Umat(2,2) = 0.0d0
    ! Symmetric input: [[5, 2], [2, 3]]
    Mati_d(1,1) = 5.0d0 ; Mati_d(1,2) = 2.0d0
    Mati_d(2,1) = 2.0d0 ; Mati_d(2,2) = 3.0d0
    ! Expected U^T M U = [[3, 2], [2, 5]] (diagonal swapped, off-diag preserved)
    Mato_d = basechange_d_gemm(M, Mati_d, Umat)
    if (abs(Mato_d(1,1) - 3.0d0) < criteria .and. &
        abs(Mato_d(2,2) - 5.0d0) < criteria .and. &
        abs(Mato_d(1,2) - 2.0d0) < criteria) then
        write(*,*) 'PASSED - basechange_d_gemm with swap matrix.'
    else
        write(*,*) 'FAILED - basechange_d_gemm with swap. Got:', Mato_d
        nfail = nfail + 1
    end if

    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All mathsubs tests passed.'
    end if

end program test_mathsubs
