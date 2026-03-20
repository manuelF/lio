!----------------------------------------------------------------------
! Unit tests for BLAS replacements in SCF hot path:
!
!   1. calc_fock_commuts: 4 triple-nested loops → 4 DGEMM calls
!      Tests that the subroutine correctly computes:
!        F' = X^T * F * X       (Fock base change)
!        [F',P'] = A - A^T      where A = X^T * F * P * Y
!
!   2. matmul(Xmat, C) → DGEMM equivalence
!      Tests that DGEMM('N','N',...) matches matmul for MO coefficient
!      base change in the SCF loop.
!
!   3. DDOT equivalence for E1 dot product
!      Tests that DDOT(N, a, 1, b, 1) matches manual loop sum.
!----------------------------------------------------------------------
program test_scf_blas
    implicit none

    integer :: nfail
    double precision :: tol
    tol = 1.0d-12
    nfail = 0

    write(*,*) '=== Testing SCF BLAS replacements ==='

    ! --- Test 1: calc_fock_commuts small case (M=3) ---
    call test_fock_commuts_small(nfail, tol)

    ! --- Test 2: calc_fock_commuts medium case (M=8) ---
    call test_fock_commuts_medium(nfail, tol)

    ! --- Test 3: calc_fock_commuts identity X=Y=I (M=4) ---
    call test_fock_commuts_identity(nfail, tol)

    ! --- Test 4: matmul vs DGEMM small (M=4) ---
    call test_matmul_vs_dgemm_small(nfail, tol)

    ! --- Test 5: matmul vs DGEMM medium (M=16) ---
    call test_matmul_vs_dgemm_medium(nfail, tol)

    ! --- Test 6: DDOT vs manual loop ---
    call test_ddot_vs_loop(nfail, tol)

    ! --- Test 7: density build dens = 2*C*C^T via DGEMM (closed-shell) ---
    call test_density_build_closed(nfail, tol)

    ! --- Test 8: density build open-shell (alpha + beta) ---
    call test_density_build_open(nfail, tol)

    ! --- Test 9: DIIS Fock accumulation via DAXPY ---
    call test_diis_accumulation(nfail, tol)

    ! --- Test 10: commutator [A,B] = AB - BA via DGEMM (real*8) ---
    call test_commutator_dd(nfail, tol)

    ! --- Test 11: commutator [A,B] via ZGEMM (complex*16) ---
    call test_commutator_zz(nfail, tol)

    ! --- Test 12: Lowdin charge via DGEMM (diag of S^½ * rho * S^½) ---
    call test_lowdin_dgemm(nfail, tol)

    ! --- Summary ---
    write(*,*)
    if (nfail == 0) then
        write(*,*) 'All SCF BLAS tests PASSED'
    else
        write(*,*) 'FAILED:', nfail, 'tests'
        stop 1
    end if

end program test_scf_blas

!----------------------------------------------------------------------
! Test 1: calc_fock_commuts with M=3
!----------------------------------------------------------------------
subroutine test_fock_commuts_small(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 3

    double precision :: F(M,M), P(M,M), X(M,M), Y(M,M)
    double precision :: scratch(M,M), scratch1(M,M)
    double precision :: F_copy(M,M)
    double precision :: Fprime_ref(M,M), comm_ref(M,M), A_ref(M,M)
    double precision :: tmp1(M,M), tmp2(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 1: calc_fock_commuts small (M=3)'

    ! Build symmetric F and P, arbitrary X and Y
    F(1,:) = (/ 2.0d0, 1.0d0, 0.5d0 /)
    F(2,:) = (/ 1.0d0, 3.0d0, 0.7d0 /)
    F(3,:) = (/ 0.5d0, 0.7d0, 4.0d0 /)

    P(1,:) = (/ 1.0d0, 0.3d0, 0.1d0 /)
    P(2,:) = (/ 0.3d0, 2.0d0, 0.4d0 /)
    P(3,:) = (/ 0.1d0, 0.4d0, 1.5d0 /)

    X(1,:) = (/ 0.9d0, 0.2d0, 0.1d0 /)
    X(2,:) = (/ 0.1d0, 0.8d0, 0.3d0 /)
    X(3,:) = (/ 0.05d0, 0.1d0, 0.7d0 /)

    Y(1,:) = (/ 1.1d0, 0.15d0, 0.05d0 /)
    Y(2,:) = (/ 0.2d0, 0.9d0,  0.25d0 /)
    Y(3,:) = (/ 0.1d0, 0.05d0, 1.3d0 /)

    F_copy = F

    ! Reference: F' = X^T * F * X
    ! tmp1 = X^T * F
    call DGEMM('T','N',M,M,M,1.0d0,X,M,F,M,0.0d0,tmp1,M)
    ! Fprime_ref = tmp1 * X
    call DGEMM('N','N',M,M,M,1.0d0,tmp1,M,X,M,0.0d0,Fprime_ref,M)

    ! Reference: A = X^T * F * P * Y (using already-computed tmp1 = X^T * F)
    ! tmp2 = tmp1 * P
    call DGEMM('N','N',M,M,M,1.0d0,tmp1,M,P,M,0.0d0,tmp2,M)
    ! A_ref = tmp2 * Y
    call DGEMM('N','N',M,M,M,1.0d0,tmp2,M,Y,M,0.0d0,A_ref,M)

    ! [F',P'] = A - A^T
    do j = 1, M
    do i = 1, M
        comm_ref(i,j) = A_ref(i,j) - A_ref(j,i)
    end do
    end do

    ! Call the subroutine under test
    call calc_fock_commuts(F_copy, P, X, Y, scratch, scratch1, M)

    ! F_copy should now contain F' = X^T * F * X
    maxerr = maxval(abs(F_copy - Fprime_ref))
    if (maxerr > tol) then
        write(*,*) '  FAIL: F'' max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: F'' (max err =', maxerr, ')'
    end if

    ! scratch = A, scratch1 = A^T
    ! The commutator [F',P'] = scratch - scratch1
    maxerr = maxval(abs(scratch - A_ref))
    if (maxerr > tol) then
        write(*,*) '  FAIL: A (scratch) max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A (max err =', maxerr, ')'
    end if

    maxerr = 0.0d0
    do j = 1, M
    do i = 1, M
        maxerr = max(maxerr, abs(scratch1(i,j) - A_ref(j,i)))
    end do
    end do
    if (maxerr > tol) then
        write(*,*) '  FAIL: A^T (scratch1) max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A^T (max err =', maxerr, ')'
    end if

end subroutine test_fock_commuts_small

!----------------------------------------------------------------------
! Test 2: calc_fock_commuts with M=8 (random-ish data)
!----------------------------------------------------------------------
subroutine test_fock_commuts_medium(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 8

    double precision :: F(M,M), P(M,M), X(M,M), Y(M,M)
    double precision :: scratch(M,M), scratch1(M,M)
    double precision :: F_copy(M,M)
    double precision :: Fprime_ref(M,M), A_ref(M,M)
    double precision :: tmp1(M,M), tmp2(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 2: calc_fock_commuts medium (M=8)'

    ! Build deterministic "random" matrices using simple formula
    do j = 1, M
    do i = 1, M
        F(i,j) = sin(dble(i*3 + j*7)) * 2.0d0
        P(i,j) = cos(dble(i*5 + j*11)) * 1.5d0
        X(i,j) = sin(dble(i*2 + j*13)) * 0.5d0
        Y(i,j) = cos(dble(i*7 + j*3)) * 0.6d0
    end do
    end do
    ! Make F and P symmetric
    do j = 1, M
    do i = j+1, M
        F(i,j) = F(j,i)
        P(i,j) = P(j,i)
    end do
    end do

    F_copy = F

    ! Reference: F' = X^T * F * X
    call DGEMM('T','N',M,M,M,1.0d0,X,M,F,M,0.0d0,tmp1,M)
    call DGEMM('N','N',M,M,M,1.0d0,tmp1,M,X,M,0.0d0,Fprime_ref,M)

    ! Reference: A = X^T * F * P * Y
    call DGEMM('N','N',M,M,M,1.0d0,tmp1,M,P,M,0.0d0,tmp2,M)
    call DGEMM('N','N',M,M,M,1.0d0,tmp2,M,Y,M,0.0d0,A_ref,M)

    ! Call subroutine
    call calc_fock_commuts(F_copy, P, X, Y, scratch, scratch1, M)

    ! Check F'
    maxerr = maxval(abs(F_copy - Fprime_ref))
    if (maxerr > tol) then
        write(*,*) '  FAIL: F'' max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: F'' (max err =', maxerr, ')'
    end if

    ! Check A
    maxerr = maxval(abs(scratch - A_ref))
    if (maxerr > tol) then
        write(*,*) '  FAIL: A max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A (max err =', maxerr, ')'
    end if

    ! Check A^T
    maxerr = 0.0d0
    do j = 1, M
    do i = 1, M
        maxerr = max(maxerr, abs(scratch1(i,j) - A_ref(j,i)))
    end do
    end do
    if (maxerr > tol) then
        write(*,*) '  FAIL: A^T max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A^T (max err =', maxerr, ')'
    end if

end subroutine test_fock_commuts_medium

!----------------------------------------------------------------------
! Test 3: calc_fock_commuts with X=Y=I (M=4)
! When X=Y=I: F' = F, A = F*P, [F',P'] = FP - PF
!----------------------------------------------------------------------
subroutine test_fock_commuts_identity(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 4

    double precision :: F(M,M), P(M,M), X(M,M), Y(M,M)
    double precision :: scratch(M,M), scratch1(M,M)
    double precision :: F_orig(M,M)
    double precision :: FP(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 3: calc_fock_commuts identity X=Y=I (M=4)'

    ! Identity transform
    X = 0.0d0
    Y = 0.0d0
    do i = 1, M
        X(i,i) = 1.0d0
        Y(i,i) = 1.0d0
    end do

    ! Symmetric F and P
    F = 0.0d0
    P = 0.0d0
    do j = 1, M
    do i = 1, M
        F(i,j) = 1.0d0 / dble(i + j)
        P(i,j) = 1.0d0 / dble(i + j + 1)
    end do
    end do

    F_orig = F

    ! Reference: A = F*P
    call DGEMM('N','N',M,M,M,1.0d0,F,M,P,M,0.0d0,FP,M)

    call calc_fock_commuts(F, P, X, Y, scratch, scratch1, M)

    ! F should be unchanged (F' = I^T*F*I = F)
    maxerr = maxval(abs(F - F_orig))
    if (maxerr > tol) then
        write(*,*) '  FAIL: F'' should equal F, max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: F'' = F (max err =', maxerr, ')'
    end if

    ! scratch = A = F*P
    maxerr = maxval(abs(scratch - FP))
    if (maxerr > tol) then
        write(*,*) '  FAIL: A = F*P max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A = F*P (max err =', maxerr, ')'
    end if

    ! scratch1 = A^T = (F*P)^T
    maxerr = 0.0d0
    do j = 1, M
    do i = 1, M
        maxerr = max(maxerr, abs(scratch1(i,j) - FP(j,i)))
    end do
    end do
    if (maxerr > tol) then
        write(*,*) '  FAIL: A^T max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: A^T (max err =', maxerr, ')'
    end if

end subroutine test_fock_commuts_identity

!----------------------------------------------------------------------
! Test 4: matmul vs DGEMM small (M=4)
! Tests: morb_coefat = matmul(Xmat, morb_coefon) vs DGEMM
!----------------------------------------------------------------------
subroutine test_matmul_vs_dgemm_small(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 4

    double precision :: Xmat(M,M), coefon(M,M)
    double precision :: result_matmul(M,M), result_dgemm(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 4: matmul vs DGEMM small (M=4)'

    do j = 1, M
    do i = 1, M
        Xmat(i,j) = sin(dble(i*3 + j*5)) * 1.5d0
        coefon(i,j) = cos(dble(i*7 + j*2)) * 2.0d0
    end do
    end do

    result_matmul = matmul(Xmat, coefon)
    call DGEMM('N','N',M,M,M,1.0d0,Xmat,M,coefon,M,0.0d0,result_dgemm,M)

    maxerr = maxval(abs(result_matmul - result_dgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: matmul == DGEMM (max err =', maxerr, ')'
    end if

end subroutine test_matmul_vs_dgemm_small

!----------------------------------------------------------------------
! Test 5: matmul vs DGEMM medium (M=16)
!----------------------------------------------------------------------
subroutine test_matmul_vs_dgemm_medium(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 16

    double precision :: Xmat(M,M), coefon(M,M)
    double precision :: result_matmul(M,M), result_dgemm(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 5: matmul vs DGEMM medium (M=16)'

    do j = 1, M
    do i = 1, M
        Xmat(i,j) = sin(dble(i*3 + j*5)) * 1.5d0
        coefon(i,j) = cos(dble(i*7 + j*2)) * 2.0d0
    end do
    end do

    result_matmul = matmul(Xmat, coefon)
    call DGEMM('N','N',M,M,M,1.0d0,Xmat,M,coefon,M,0.0d0,result_dgemm,M)

    maxerr = maxval(abs(result_matmul - result_dgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: matmul == DGEMM (max err =', maxerr, ')'
    end if

end subroutine test_matmul_vs_dgemm_medium

!----------------------------------------------------------------------
! Test 6: DDOT vs manual dot product loop
! Mimics: E1 = sum(Pmat_vec(k)*Hmat_vec(k), k=1..MM)
!----------------------------------------------------------------------
subroutine test_ddot_vs_loop(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: N = 100

    double precision :: a(N), b(N)
    double precision :: sum_loop, sum_ddot
    double precision :: DDOT
    double precision :: err
    integer :: k

    write(*,*) 'Test 6: DDOT vs manual loop (N=100)'

    do k = 1, N
        a(k) = sin(dble(k*3)) * 2.5d0
        b(k) = cos(dble(k*7)) * 1.3d0
    end do

    ! Manual loop (original code pattern)
    sum_loop = 0.0d0
    do k = 1, N
        sum_loop = sum_loop + a(k) * b(k)
    end do

    ! DDOT
    sum_ddot = DDOT(N, a, 1, b, 1)

    err = abs(sum_loop - sum_ddot)
    if (err > tol) then
        write(*,*) '  FAIL: |loop - DDOT| =', err
        nfail = nfail + 1
    else
        write(*,*) '  PASS: loop == DDOT (err =', err, ')'
    end if

end subroutine test_ddot_vs_loop

!----------------------------------------------------------------------
! Test 7: Density build dens = 2*C*C^T via DGEMM (closed-shell)
! Matches restart_coef.f90 read_coef_restart_cd triple loop:
!   dens(i,j) = sum_k 2*coef(i,k)*coef(j,k)
!----------------------------------------------------------------------
subroutine test_density_build_closed(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 6, NCO = 3

    double precision :: coef(M,NCO), dens_loop(M,M), dens_dgemm(M,M)
    double precision :: maxerr
    integer :: i, j, k

    write(*,*) 'Test 7: density build closed-shell (M=6, NCO=3)'

    ! Fill coefficients
    do j = 1, NCO
    do i = 1, M
        coef(i,j) = sin(dble(i*3 + j*7)) * 1.5d0
    end do
    end do

    ! Original triple loop
    dens_loop = 0.0d0
    do i = 1, M
    do j = 1, M
    do k = 1, NCO
        dens_loop(i,j) = dens_loop(i,j) + 2.0d0*coef(i,k)*coef(j,k)
    end do
    end do
    end do

    ! DGEMM: dens = 2 * C * C^T
    call DGEMM('N','T',M,M,NCO,2.0d0,coef,M,coef,M,0.0d0,dens_dgemm,M)

    maxerr = maxval(abs(dens_loop - dens_dgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: loop == DGEMM (max err =', maxerr, ')'
    end if

end subroutine test_density_build_closed

!----------------------------------------------------------------------
! Test 8: Density build open-shell (alpha + beta)
! Matches restart_coef.f90 read_coef_restart_od:
!   dens_a(i,j) = sum_k coef_a(i,k)*coef_a(j,k)
!   dens_b(i,j) = sum_k coef_b(i,k)*coef_b(j,k)
!   dens_t = dens_a + dens_b
!----------------------------------------------------------------------
subroutine test_density_build_open(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 6, NCOa = 3, NCOb = 2

    double precision :: coef_a(M,NCOa), coef_b(M,NCOb)
    double precision :: dens_a_loop(M,M), dens_b_loop(M,M), dens_t_loop(M,M)
    double precision :: dens_a_blas(M,M), dens_b_blas(M,M), dens_t_blas(M,M)
    double precision :: maxerr
    integer :: i, j, k

    write(*,*) 'Test 8: density build open-shell (M=6, NCOa=3, NCOb=2)'

    do j = 1, NCOa
    do i = 1, M
        coef_a(i,j) = sin(dble(i*3 + j*7)) * 1.5d0
    end do
    end do
    do j = 1, NCOb
    do i = 1, M
        coef_b(i,j) = cos(dble(i*5 + j*11)) * 1.2d0
    end do
    end do

    ! Original triple loops
    dens_a_loop = 0.0d0
    dens_b_loop = 0.0d0
    dens_t_loop = 0.0d0
    do i = 1, M
    do j = 1, M
        do k = 1, NCOa
            dens_t_loop(i,j) = dens_t_loop(i,j) + coef_a(i,k)*coef_a(j,k)
            dens_a_loop(i,j) = dens_a_loop(i,j) + coef_a(i,k)*coef_a(j,k)
        end do
        do k = 1, NCOb
            dens_t_loop(i,j) = dens_t_loop(i,j) + coef_b(i,k)*coef_b(j,k)
            dens_b_loop(i,j) = dens_b_loop(i,j) + coef_b(i,k)*coef_b(j,k)
        end do
    end do
    end do

    ! BLAS: dens_a = C_a * C_a^T, dens_b = C_b * C_b^T, dens_t = dens_a + dens_b
    call DGEMM('N','T',M,M,NCOa,1.0d0,coef_a,M,coef_a,M,0.0d0,dens_a_blas,M)
    call DGEMM('N','T',M,M,NCOb,1.0d0,coef_b,M,coef_b,M,0.0d0,dens_b_blas,M)
    dens_t_blas = dens_a_blas + dens_b_blas

    maxerr = maxval(abs(dens_a_loop - dens_a_blas))
    if (maxerr > tol) then
        write(*,*) '  FAIL: dens_a max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: dens_a (max err =', maxerr, ')'
    end if

    maxerr = maxval(abs(dens_b_loop - dens_b_blas))
    if (maxerr > tol) then
        write(*,*) '  FAIL: dens_b max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: dens_b (max err =', maxerr, ')'
    end if

    maxerr = maxval(abs(dens_t_loop - dens_t_blas))
    if (maxerr > tol) then
        write(*,*) '  FAIL: dens_t max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: dens_t (max err =', maxerr, ')'
    end if

end subroutine test_density_build_open

!----------------------------------------------------------------------
! Test 9: DIIS Fock accumulation via DAXPY
! Original: suma_w(i,j) += bcoef(k) * fockm(i,j,k)  (triple loop)
! BLAS: DAXPY(M*M, bcoef(k), fockm(1,1,k), 1, suma_w, 1) per k
!----------------------------------------------------------------------
subroutine test_diis_accumulation(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 6, NDIIS = 4

    double precision :: fockm(M,M,NDIIS), bcoef(NDIIS)
    double precision :: suma_loop(M,M), suma_daxpy(M,M)
    double precision :: maxerr
    integer :: ii, jj, kk

    write(*,*) 'Test 9: DIIS Fock accumulation (M=6, ndiis=4)'

    ! Fill test data
    do kk = 1, NDIIS
    do jj = 1, M
    do ii = 1, M
        fockm(ii,jj,kk) = sin(dble(ii*3 + jj*7 + kk*13)) * 2.0d0
    end do
    end do
    end do
    bcoef(1) = 0.3d0
    bcoef(2) = 0.5d0
    bcoef(3) = -0.2d0
    bcoef(4) = 0.4d0

    ! Original triple loop
    suma_loop = 0.0d0
    do kk = 1, NDIIS
        do ii = 1, M
        do jj = 1, M
            suma_loop(ii,jj) = suma_loop(ii,jj) + bcoef(kk) * fockm(ii,jj,kk)
        end do
        end do
    end do

    ! DAXPY version
    suma_daxpy = 0.0d0
    do kk = 1, NDIIS
        call DAXPY(M*M, bcoef(kk), fockm(1,1,kk), 1, suma_daxpy, 1)
    end do

    maxerr = maxval(abs(suma_loop - suma_daxpy))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: loop == DAXPY (max err =', maxerr, ')'
    end if

end subroutine test_diis_accumulation

!----------------------------------------------------------------------
! Test 10: commutator [A,B] = AB - BA via DGEMM (real*8)
!----------------------------------------------------------------------
subroutine test_commutator_dd(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 6

    double precision :: A(M,M), B(M,M)
    double precision :: comm_matmul(M,M), comm_dgemm(M,M)
    double precision :: AB(M,M), BA(M,M)
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 10: commutator [A,B] real*8 (M=6)'

    do j = 1, M
    do i = 1, M
        A(i,j) = sin(dble(i*3 + j*7)) * 2.0d0
        B(i,j) = cos(dble(i*5 + j*11)) * 1.5d0
    end do
    end do

    ! MATMUL version (original)
    comm_matmul = matmul(A,B) - matmul(B,A)

    ! DGEMM version: AB = A*B, then comm = AB - BA
    ! AB = A*B
    call DGEMM('N','N',M,M,M,1.0d0,A,M,B,M,0.0d0,AB,M)
    ! comm = AB (copy)
    comm_dgemm = AB
    ! comm = comm - B*A = AB - BA
    call DGEMM('N','N',M,M,M,-1.0d0,B,M,A,M,1.0d0,comm_dgemm,M)

    maxerr = maxval(abs(comm_matmul - comm_dgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: matmul == DGEMM (max err =', maxerr, ')'
    end if

end subroutine test_commutator_dd

!----------------------------------------------------------------------
! Test 11: commutator [A,B] via ZGEMM (complex*16)
!----------------------------------------------------------------------
subroutine test_commutator_zz(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 5

    complex*16 :: A(M,M), B(M,M)
    complex*16 :: comm_matmul(M,M), comm_zgemm(M,M)
    complex*16 :: AB(M,M)
    complex*16 :: one, zero, neg_one
    double precision :: maxerr
    integer :: i, j

    write(*,*) 'Test 11: commutator [A,B] complex*16 (M=5)'

    one = dcmplx(1.0d0, 0.0d0)
    zero = dcmplx(0.0d0, 0.0d0)
    neg_one = dcmplx(-1.0d0, 0.0d0)

    do j = 1, M
    do i = 1, M
        A(i,j) = dcmplx(sin(dble(i*3+j*7)), cos(dble(i*2+j*5)))
        B(i,j) = dcmplx(cos(dble(i*5+j*11)), sin(dble(i*7+j*3)))
    end do
    end do

    ! MATMUL version
    comm_matmul = matmul(A,B) - matmul(B,A)

    ! ZGEMM version: AB = A*B, comm = AB - B*A
    call ZGEMM('N','N',M,M,M,one,A,M,B,M,zero,AB,M)
    comm_zgemm = AB
    call ZGEMM('N','N',M,M,M,neg_one,B,M,A,M,one,comm_zgemm,M)

    maxerr = maxval(abs(comm_matmul - comm_zgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: matmul == ZGEMM (max err =', maxerr, ')'
    end if

end subroutine test_commutator_zz

!----------------------------------------------------------------------
! Test 12: Lowdin charge via DGEMM
! Original: newterm = sum_i sum_j sqsmat(k,i)*rhomat(i,j)*sqsmat(j,k)
! This is diag(S^½ * rho * S^½) at element k = (S^½ * rho * S^½)(k,k)
! DGEMM: tmp = S^½ * rho, result = tmp * S^½, then extract diagonal
!----------------------------------------------------------------------
subroutine test_lowdin_dgemm(nfail, tol)
    implicit none
    integer, intent(inout) :: nfail
    double precision, intent(in) :: tol
    integer, parameter :: M = 5

    double precision :: sqsmat(M,M), rhomat(M,M)
    double precision :: charges_loop(M), charges_dgemm(M)
    double precision :: tmp(M,M), result(M,M)
    double precision :: newterm, maxerr
    integer :: i, j, k

    write(*,*) 'Test 12: Lowdin charge via DGEMM (M=5)'

    ! Build symmetric matrices
    do j = 1, M
    do i = 1, M
        sqsmat(i,j) = sin(dble(i*3 + j*7)) * 0.5d0
        rhomat(i,j) = cos(dble(i*5 + j*11)) * 1.5d0
    end do
    end do
    ! Symmetrize
    do j = 1, M
    do i = j+1, M
        sqsmat(i,j) = sqsmat(j,i)
        rhomat(i,j) = rhomat(j,i)
    end do
    end do

    ! Original triple loop (per basis function k)
    do k = 1, M
        newterm = 0.0d0
        do i = 1, M
        do j = 1, M
            newterm = newterm + sqsmat(k,i) * rhomat(i,j) * sqsmat(j,k)
        end do
        end do
        charges_loop(k) = newterm
    end do

    ! DGEMM: tmp = sqsmat * rhomat, result = tmp * sqsmat
    call DGEMM('N','N',M,M,M,1.0d0,sqsmat,M,rhomat,M,0.0d0,tmp,M)
    call DGEMM('N','N',M,M,M,1.0d0,tmp,M,sqsmat,M,0.0d0,result,M)
    ! Extract diagonal
    do k = 1, M
        charges_dgemm(k) = result(k,k)
    end do

    maxerr = maxval(abs(charges_loop - charges_dgemm))
    if (maxerr > tol) then
        write(*,*) '  FAIL: max error =', maxerr
        nfail = nfail + 1
    else
        write(*,*) '  PASS: loop == DGEMM (max err =', maxerr, ')'
    end if

end subroutine test_lowdin_dgemm
