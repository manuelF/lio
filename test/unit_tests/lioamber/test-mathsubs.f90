program test_mathsubs
    use mathsubs, only: basechange_d_gemm, basechange_z_gemm
    use linear_algebra, only: matmuldiag
    implicit none

    integer, parameter :: M = 2
    real*8 :: Umat(M,M), Mati_d(M,M)
    complex*16 :: Mati_z(M,M)
    real*8, allocatable :: Mato_d(:,:)
    complex*16, allocatable :: Mato_z(:,:)
    integer :: i, k
    real*8 :: criteria
    integer :: nfail
    real*8 :: A2(M,M), B2(M,M), C2(M,M), trace_val, trace_ref

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

    ! 4. Test matmuldiag / trace equivalence
    ! A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    ! A*B = [[19, 22], [43, 50]]
    ! Tr(A*B) = 19 + 50 = 69
    ! matmuldiag computes C where C(i,i) = sum_k A(i,k)*B(k,i) (diagonal of A*B)
    A2(1,1) = 1.0d0 ; A2(1,2) = 2.0d0
    A2(2,1) = 3.0d0 ; A2(2,2) = 4.0d0
    B2(1,1) = 5.0d0 ; B2(1,2) = 6.0d0
    B2(2,1) = 7.0d0 ; B2(2,2) = 8.0d0

    call matmuldiag(A2, B2, C2, M)
    trace_val = 0.0d0
    do i = 1, M
        trace_val = trace_val + C2(i,i)
    end do
    trace_ref = 69.0d0

    if (abs(trace_val - trace_ref) < criteria) then
        write(*,*) 'PASSED - matmuldiag trace = Tr(A*B).'
    else
        write(*,*) 'FAILED - matmuldiag trace =', trace_val, ' expected', trace_ref
        nfail = nfail + 1
    end if

    ! Verify individual diagonal entries: C(1,1) = 19, C(2,2) = 50
    if (abs(C2(1,1) - 19.0d0) < criteria .and. abs(C2(2,2) - 50.0d0) < criteria) then
        write(*,*) 'PASSED - matmuldiag diagonal entries correct.'
    else
        write(*,*) 'FAILED - matmuldiag diag: C(1,1)=', C2(1,1), ' C(2,2)=', C2(2,2)
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
