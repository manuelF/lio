program test_mathsubs
    use mathsubs, only: basechange_d_gemm, basechange_z_gemm
    implicit none
    
    integer, parameter :: M = 2
    real*8 :: Umat(M,M), Mati_d(M,M)
    complex*16 :: Mati_z(M,M)
    real*8, allocatable :: Mato_d(:,:)
    complex*16, allocatable :: Mato_z(:,:)
    integer :: i, j
    real*8 :: criteria
    
    criteria = 1.0d-12
    write(*,*) '--- Testing mathsubs ---'
    
    ! Identity transformation matrix
    Umat = 0.0d0
    do i=1,M
        Umat(i,i) = 1.0d0
    end do
    
    ! 1. Test basechange_d_gemm
    write(*,*) 'Testing basechange_d_gemm...'
    Mati_d(1,1) = 1.0d0 ; Mati_d(1,2) = 2.0d0
    Mati_d(2,1) = 3.0d0 ; Mati_d(2,2) = 4.0d0
    
    Mato_d = basechange_d_gemm(M, Mati_d, Umat)
    
    if (all(abs(Mato_d - Mati_d) < criteria)) then
        write(*,*) 'PASSED: basechange_d_gemm with identity.'
    else
        write(*,*) 'FAILED: basechange_d_gemm with identity.'
        print *, 'Got:', Mato_d
    end if
    
    ! 2. Test basechange_z_gemm
    write(*,*) 'Testing basechange_z_gemm...'
    Mati_z(1,1) = cmplx(1.0d0, 0.1d0) ; Mati_z(1,2) = cmplx(2.0d0, 0.2d0)
    Mati_z(2,1) = cmplx(3.0d0, 0.3d0) ; Mati_z(2,2) = cmplx(4.0d0, 0.4d0)
    
    Mato_z = basechange_z_gemm(M, Mati_z, Umat)
    
    if (all(abs(Mato_z - Mati_z) < criteria)) then
        write(*,*) 'PASSED: basechange_z_gemm with identity.'
    else
        write(*,*) 'FAILED: basechange_z_gemm with identity.'
        print *, 'Got:', Mato_z
    end if

end program test_mathsubs
