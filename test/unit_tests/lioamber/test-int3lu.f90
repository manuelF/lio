!----------------------------------------------------------------------
! Unit tests for subm_int3lu :: int3lu
!
! Tests the MEMO=.true. path, which uses BLAS (DGEMV, SGEMV, DSPMV,
! DDOT) to compute:
!   1. Rc = cool * rho_gathered  +  cools * rho_s   (density contraction)
!   2. af = Ginv * Rc                                (fitting coefficients)
!   3. Ea = dot(af, Rc), Eb = af^T * Gmat * af      (Coulomb energy)
!   4. Fmat += cool^T * af  +  cools^T * af_s       (Fock update)
!
! Each verify_* routine recomputes the expected result using the original
! scalar loop formulas, then compares against the BLAS output.
!
! Test matrix:
!   1. Small closed-shell   (M=3, Md=2, kknumd=2, kknums=1)
!   2. Small open-shell     (same data as test 1)
!   3. Medium closed-shell  (M=6, Md=4, kknumd=5, kknums=3)
!   4. Medium open-shell    (same data as test 3)
!   5. No single-precision  (M=4, Md=3, kknumd=3, kknums=0)
!   6. No double-precision  (M=4, Md=3, kknumd=0, kknums=4)
!   7. Duplicate kkind       (M=4, Md=3, kknumd=3 with repeated indices)
!   8. Minimal Md=1          (single fitting function)
!----------------------------------------------------------------------
program test_int3lu
    use basis_data, only: M, Md, MM, MMd, cool, cools, kkind, kkinds, &
                          kknumd, kknums, af
    use subm_int3lu, only: int3lu
    implicit none

    integer :: nfail, i
    double precision :: E2
    double precision, allocatable :: rho(:), Gmat(:), Ginv(:), Hmat(:)
    double precision, allocatable :: Fmat(:), Fmat_b(:)
    double precision, allocatable :: Fmat_ref(:), af_ref(:)
    double precision :: E2_ref

    ! Tight tolerance for double-only paths, relaxed for cools SGEMV paths
    double precision :: tol_dp, tol_sp
    tol_dp = 1.0d-10
    tol_sp = 1.0d-5

    nfail = 0
    write(*,*) '=== Testing int3lu ==='

    ! -------------------------------------------------------------------
    ! Test 1: Small closed-shell MEMO case
    !   M=3, Md=2, kknumd=2, kknums=1
    !   Hand-computed expected values verified in comments.
    ! -------------------------------------------------------------------
    call setup_small_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)

    ! Rc(1) = rho(1)*cool(1) + rho(3)*cool(3) + rho(2)*cools(1)
    !       = 2.0*1.0 + 1.0*3.0 + 0.5*0.5 = 5.25
    ! Rc(2) = 2.0*2.0 + 1.0*4.0 + 0.5*1.5 = 8.75
    ! af = Ginv * Rc:  Ginv = [[1.0, 0.5], [0.5, 2.0]]
    !   af(1) = 1.0*5.25 + 0.5*8.75 = 9.625
    !   af(2) = 0.5*5.25 + 2.0*8.75 = 20.125
    allocate(af_ref(2))
    af_ref = (/ 9.625d0, 20.125d0 /)
    call check_vec('T1 af closed-shell', af, af_ref, 2, tol_sp, nfail)
    deallocate(af_ref)

    ! E2 = Ea - Eb/2 = 226.625 - 2285.390625/2 = -916.0703125
    E2_ref = -916.0703125d0
    call check_scalar('T1 E2 closed-shell', E2, E2_ref, tol_sp, nfail)

    ! Fmat = Hmat + Coulomb contributions from cool and cools scatter
    allocate(Fmat_ref(6))
    Fmat_ref = (/ 49.975d0, 35.2d0, 109.675d0, 0.4d0, 0.5d0, 0.6d0 /)
    call check_vec('T1 Fmat closed-shell', Fmat, Fmat_ref, 6, tol_sp, nfail)
    deallocate(Fmat_ref)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 2: Small open-shell (same data, both Fmat and Fmat_b updated)
    ! -------------------------------------------------------------------
    call setup_small_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .true., .true.)

    call check_scalar('T2 E2 open-shell', E2, E2_ref, tol_sp, nfail)
    allocate(Fmat_ref(6))
    Fmat_ref = (/ 49.975d0, 35.2d0, 109.675d0, 0.4d0, 0.5d0, 0.6d0 /)
    call check_vec('T2 Fmat alpha', Fmat, Fmat_ref, 6, tol_sp, nfail)
    call check_vec('T2 Fmat_b beta', Fmat_b, Fmat_ref, 6, tol_sp, nfail)
    deallocate(Fmat_ref)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 3: Medium closed-shell (M=6, Md=4, kknumd=5, kknums=3)
    !   Uses generated data; verified against reference scalar loops.
    ! -------------------------------------------------------------------
    call setup_medium_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)
    call verify_generic('T3 medium closed', E2, Fmat, af, .false., &
                        tol_sp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 4: Medium open-shell (same data, verify Fmat_b == Fmat)
    ! -------------------------------------------------------------------
    call setup_medium_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .true., .true.)
    call verify_generic('T4 medium open alpha', E2, Fmat, af, .false., &
                        tol_sp, nfail)
    call check_vec('T4 Fmat_b == Fmat', Fmat_b, Fmat, MM, tol_dp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 5: No single-precision integrals (kknums=0)
    !   Exercises the kknums=0 guard (SGEMV skipped).
    !   Tighter tolerance since everything is double precision.
    ! -------------------------------------------------------------------
    call setup_no_single_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)
    call verify_generic('T5 no-single', E2, Fmat, af, .false., &
                        tol_dp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 6: No double-precision integrals (kknumd=0)
    !   Exercises the kknumd=0 guard (DGEMV skipped).
    !   All Rc comes from single-precision cools path.
    ! -------------------------------------------------------------------
    call setup_no_double_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)
    call verify_generic('T6 no-double', E2, Fmat, af, .true., &
                        tol_sp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 7: Duplicate kkind indices
    !   Two kkind entries map to the same Fmat element. Verifies that
    !   the scatter-add correctly accumulates both contributions.
    ! -------------------------------------------------------------------
    call setup_duplicate_indices_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)
    call verify_generic('T7 dup indices', E2, Fmat, af, .false., &
                        tol_sp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Test 8: Minimal Md=1 (single fitting function)
    !   Edge case: all BLAS calls operate on scalars/1-element vectors.
    ! -------------------------------------------------------------------
    call setup_minimal_md_case()
    E2 = 0.0d0
    call int3lu(E2, rho, Fmat_b, Fmat, Gmat, Ginv, Hmat, .false., .true.)
    call verify_generic('T8 Md=1', E2, Fmat, af, .false., tol_sp, nfail)
    call cleanup()

    ! -------------------------------------------------------------------
    ! Summary
    ! -------------------------------------------------------------------
    write(*,*)
    if (nfail > 0) then
        write(*,'(A,I3,A)') ' FAILED: ', nfail, ' test(s) failed.'
        error stop 1
    else
        write(*,'(A)') ' All int3lu tests passed.'
    end if

contains

    !==================================================================
    ! Assertion helpers
    !==================================================================
    ! Relative tolerance check: uses max(|val|, |ref|, 1) as denominator
    ! to avoid division by zero and handle values near zero gracefully.
    subroutine check_scalar(label, val, ref, tol, nf)
        character(*), intent(in) :: label
        double precision, intent(in) :: val, ref, tol
        integer, intent(inout) :: nf
        double precision :: rel_err, scale
        scale = max(abs(val), abs(ref), 1.0d0)
        rel_err = abs(val - ref) / scale
        if (rel_err < tol) then
            write(*,'(A,A)') ' PASSED - ', label
        else
            write(*,'(A,A,ES20.12,A,ES20.12,A,ES10.2)') ' FAILED - ', &
                label, val, ' expected ', ref, ' rel_err=', rel_err
            nf = nf + 1
        end if
    end subroutine

    ! Relative tolerance check per element.
    subroutine check_vec(label, v, vref, n, tol, nf)
        character(*), intent(in) :: label
        integer, intent(in) :: n
        double precision, intent(in) :: v(n), vref(n), tol
        integer, intent(inout) :: nf
        double precision :: rel_err, scale, max_rel
        integer :: i_max, ii
        max_rel = 0.0d0
        i_max = 1
        do ii = 1, n
            scale = max(abs(v(ii)), abs(vref(ii)), 1.0d0)
            rel_err = abs(v(ii) - vref(ii)) / scale
            if (rel_err > max_rel) then
                max_rel = rel_err
                i_max = ii
            end if
        end do
        if (max_rel < tol) then
            write(*,'(A,A)') ' PASSED - ', label
        else
            write(*,'(A,A,A,I4,A,ES20.12,A,ES20.12)') ' FAILED - ', label, &
                ' at index ', i_max, ': got ', v(i_max), ' expected ', vref(i_max)
            nf = nf + 1
        end if
    end subroutine

    !==================================================================
    ! Generic verification: recomputes expected values using original
    ! scalar loop formulas (the "reference implementation") and compares
    ! against the BLAS output from int3lu.
    !
    ! If sp_only is .true., the Rc reference also uses single-precision
    ! promotion (matching SGEMV behavior) for the cools path.
    !==================================================================
    subroutine verify_generic(label, E2_val, Fmat_val, af_val, sp_only, &
                              tol, nf)
        character(*), intent(in) :: label
        double precision, intent(in) :: E2_val, Fmat_val(:), af_val(:), tol
        logical, intent(in) :: sp_only
        integer, intent(inout) :: nf

        double precision, allocatable :: Rc(:), af_exp(:), Fmat_exp(:)
        double precision :: Ea, Eb, E2_exp, term
        integer :: m_ind, k_ind, kk_ind, iikk, pk_idx

        allocate(Rc(Md), af_exp(Md), Fmat_exp(MM))

        ! Recompute Rc using original scalar loops
        Rc = 0.0d0
        do kk_ind = 1, kknumd
            iikk = (kk_ind - 1) * Md
            do k_ind = 1, Md
                Rc(k_ind) = Rc(k_ind) + rho(kkind(kk_ind)) * cool(iikk + k_ind)
            end do
        end do
        do kk_ind = 1, kknums
            iikk = (kk_ind - 1) * Md
            do k_ind = 1, Md
                Rc(k_ind) = Rc(k_ind) + rho(kkinds(kk_ind)) * &
                            dble(cools(iikk + k_ind))
            end do
        end do

        ! Recompute af = Ginv * Rc using original packed symmetric loops
        do m_ind = 1, Md
            af_exp(m_ind) = 0.0d0
            do k_ind = 1, m_ind - 1
                pk_idx = m_ind + (2*Md - k_ind) * (k_ind - 1) / 2
                af_exp(m_ind) = af_exp(m_ind) + Rc(k_ind) * Ginv(pk_idx)
            end do
            do k_ind = m_ind, Md
                pk_idx = k_ind + (2*Md - m_ind) * (m_ind - 1) / 2
                af_exp(m_ind) = af_exp(m_ind) + Rc(k_ind) * Ginv(pk_idx)
            end do
        end do
        call check_vec(trim(label)//' af', af_val, af_exp, Md, tol, nf)

        ! Recompute energy
        Ea = 0.0d0;  Eb = 0.0d0
        do m_ind = 1, Md
            Ea = Ea + af_exp(m_ind) * Rc(m_ind)
            do k_ind = 1, m_ind
                pk_idx = m_ind + (2*Md - k_ind) * (k_ind - 1) / 2
                Eb = Eb + af_exp(k_ind) * af_exp(m_ind) * Gmat(pk_idx)
            end do
            do k_ind = m_ind + 1, Md
                pk_idx = k_ind + (2*Md - m_ind) * (m_ind - 1) / 2
                Eb = Eb + af_exp(k_ind) * af_exp(m_ind) * Gmat(pk_idx)
            end do
        end do
        E2_exp = Ea - Eb / 2.0d0
        call check_scalar(trim(label)//' E2', E2_val, E2_exp, tol, nf)

        ! Recompute Fmat
        Fmat_exp(1:MM) = Hmat(1:MM)
        do kk_ind = 1, kknumd
            iikk = (kk_ind - 1) * Md
            term = 0.0d0
            do k_ind = 1, Md
                term = term + af_exp(k_ind) * cool(iikk + k_ind)
            end do
            Fmat_exp(kkind(kk_ind)) = Fmat_exp(kkind(kk_ind)) + term
        end do
        do kk_ind = 1, kknums
            iikk = (kk_ind - 1) * Md
            term = 0.0d0
            do k_ind = 1, Md
                term = term + af_exp(k_ind) * dble(cools(iikk + k_ind))
            end do
            Fmat_exp(kkinds(kk_ind)) = Fmat_exp(kkinds(kk_ind)) + term
        end do
        call check_vec(trim(label)//' Fmat', Fmat_val, Fmat_exp, MM, tol, nf)

        deallocate(Rc, af_exp, Fmat_exp)
    end subroutine

    !==================================================================
    ! Shared cleanup: deallocates all basis_data module arrays and
    ! local test arrays.
    !==================================================================
    subroutine cleanup()
        if (allocated(cool))   deallocate(cool)
        if (allocated(cools))  deallocate(cools)
        if (allocated(kkind))  deallocate(kkind)
        if (allocated(kkinds)) deallocate(kkinds)
        if (allocated(af))     deallocate(af)
        if (allocated(rho))    deallocate(rho)
        if (allocated(Ginv))   deallocate(Ginv)
        if (allocated(Gmat))   deallocate(Gmat)
        if (allocated(Hmat))   deallocate(Hmat)
        if (allocated(Fmat))   deallocate(Fmat)
        if (allocated(Fmat_b)) deallocate(Fmat_b)
    end subroutine

    !==================================================================
    ! Test case setups
    !==================================================================

    ! --- Test 1 & 2: Small case (M=3, Md=2, kknumd=2, kknums=1) ---
    subroutine setup_small_case()
        M = 3;  Md = 2
        MM = M * (M + 1) / 2     ! 6
        MMd = Md * (Md + 1) / 2  ! 3
        kknumd = 2;  kknums = 1

        allocate(cool(kknumd * Md))   ! 4 elements
        allocate(cools(kknums * Md))  ! 2 elements
        allocate(kkind(kknumd), kkinds(kknums), af(Md))

        ! cool as (Md=2, kknumd=2) matrix:
        !   column 1: [1.0, 2.0]   column 2: [3.0, 4.0]
        cool(1) = 1.0d0;  cool(2) = 2.0d0
        cool(3) = 3.0d0;  cool(4) = 4.0d0

        ! cools as (Md=2, kknums=1) matrix:
        !   column 1: [0.5, 1.5]
        cools(1) = 0.5;   cools(2) = 1.5

        ! kkind(kk) maps basis pair kk to packed rho/Fmat index
        kkind(1) = 1;  kkind(2) = 3
        kkinds(1) = 2

        allocate(rho(MM))
        rho = (/ 2.0d0, 0.5d0, 1.0d0, 0.3d0, 0.7d0, 1.5d0 /)

        ! Ginv as symmetric [[1.0, 0.5], [0.5, 2.0]], packed lower tri
        allocate(Ginv(MMd))
        Ginv(1) = 1.0d0;  Ginv(2) = 0.5d0;  Ginv(3) = 2.0d0

        ! Gmat as symmetric [[3.0, 1.0], [1.0, 4.0]], packed lower tri
        allocate(Gmat(MMd))
        Gmat(1) = 3.0d0;  Gmat(2) = 1.0d0;  Gmat(3) = 4.0d0

        allocate(Hmat(MM))
        Hmat = (/ 0.1d0, 0.2d0, 0.3d0, 0.4d0, 0.5d0, 0.6d0 /)

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

    ! --- Test 3 & 4: Medium case (M=6, Md=4, kknumd=5, kknums=3) ---
    subroutine setup_medium_case()
        integer :: k
        M = 6;  Md = 4
        MM = M * (M + 1) / 2     ! 21
        MMd = Md * (Md + 1) / 2  ! 10
        kknumd = 5;  kknums = 3

        allocate(cool(kknumd * Md), cools(kknums * Md))
        allocate(kkind(kknumd), kkinds(kknums), af(Md))

        do k = 1, kknumd * Md
            cool(k) = dble(k) * 0.1d0
        end do
        do k = 1, kknums * Md
            cools(k) = real(k) * 0.2
        end do

        ! Distinct indices, all valid for MM=21
        kkind(1) = 1;  kkind(2) = 3;  kkind(3) = 6
        kkind(4) = 10; kkind(5) = 15
        kkinds(1) = 2;  kkinds(2) = 5;  kkinds(3) = 9

        allocate(rho(MM))
        do k = 1, MM
            rho(k) = 1.0d0 / dble(k)
        end do

        allocate(Ginv(MMd))
        Ginv = (/ 2.0d0, 0.1d0, 2.5d0, 0.2d0, 0.15d0, 3.0d0, &
                  0.05d0, 0.1d0, 0.2d0, 3.5d0 /)

        allocate(Gmat(MMd))
        Gmat = (/ 4.0d0, 0.5d0, 3.0d0, 0.3d0, 0.4d0, 5.0d0, &
                  0.1d0, 0.2d0, 0.6d0, 6.0d0 /)

        allocate(Hmat(MM))
        do k = 1, MM
            Hmat(k) = dble(k) * 0.01d0
        end do

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

    ! --- Test 5: No single-precision (kknums=0) ---
    subroutine setup_no_single_case()
        integer :: k
        M = 4;  Md = 3
        MM = M * (M + 1) / 2     ! 10
        MMd = Md * (Md + 1) / 2  ! 6
        kknumd = 3;  kknums = 0

        allocate(cool(kknumd * Md))
        allocate(cools(1))     ! kknums=0, allocate dummy
        allocate(kkind(kknumd), kkinds(1), af(Md))

        do k = 1, kknumd * Md
            cool(k) = dble(k) * 0.5d0
        end do
        kkind(1) = 1;  kkind(2) = 4;  kkind(3) = 7

        allocate(rho(MM))
        do k = 1, MM
            rho(k) = 0.5d0 * dble(k)
        end do

        allocate(Ginv(MMd))
        Ginv = (/ 1.0d0, 0.2d0, 1.5d0, 0.1d0, 0.3d0, 2.0d0 /)

        allocate(Gmat(MMd))
        Gmat = (/ 2.0d0, 0.3d0, 3.0d0, 0.15d0, 0.25d0, 4.0d0 /)

        allocate(Hmat(MM))
        do k = 1, MM
            Hmat(k) = dble(k) * 0.1d0
        end do

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

    ! --- Test 6: No double-precision (kknumd=0) ---
    ! All Rc comes from the cools SGEMV path.
    subroutine setup_no_double_case()
        integer :: k
        M = 4;  Md = 3
        MM = M * (M + 1) / 2     ! 10
        MMd = Md * (Md + 1) / 2  ! 6
        kknumd = 0;  kknums = 4

        allocate(cool(1))      ! kknumd=0, allocate dummy
        allocate(cools(kknums * Md))
        allocate(kkind(1), kkinds(kknums), af(Md))

        do k = 1, kknums * Md
            cools(k) = real(k) * 0.3
        end do
        kkinds(1) = 1;  kkinds(2) = 3;  kkinds(3) = 6;  kkinds(4) = 10

        allocate(rho(MM))
        do k = 1, MM
            rho(k) = 0.4d0 * dble(k)
        end do

        allocate(Ginv(MMd))
        Ginv = (/ 1.5d0, 0.1d0, 2.0d0, 0.2d0, 0.15d0, 2.5d0 /)

        allocate(Gmat(MMd))
        Gmat = (/ 3.0d0, 0.2d0, 4.0d0, 0.1d0, 0.3d0, 5.0d0 /)

        allocate(Hmat(MM))
        do k = 1, MM
            Hmat(k) = dble(k) * 0.05d0
        end do

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

    ! --- Test 7: Duplicate kkind indices ---
    ! kkind(1) = kkind(3) = 1: both contributions scatter-add to Fmat(1).
    subroutine setup_duplicate_indices_case()
        integer :: k
        M = 4;  Md = 3
        MM = M * (M + 1) / 2     ! 10
        MMd = Md * (Md + 1) / 2  ! 6
        kknumd = 3;  kknums = 2

        allocate(cool(kknumd * Md), cools(kknums * Md))
        allocate(kkind(kknumd), kkinds(kknums), af(Md))

        do k = 1, kknumd * Md
            cool(k) = dble(k) * 0.3d0
        end do
        do k = 1, kknums * Md
            cools(k) = real(k) * 0.4
        end do

        ! Duplicate: kkind(1) = kkind(3) = 1 (same Fmat element)
        kkind(1) = 1;  kkind(2) = 5;  kkind(3) = 1
        ! Also duplicate in kkinds
        kkinds(1) = 2;  kkinds(2) = 2

        allocate(rho(MM))
        do k = 1, MM
            rho(k) = 1.0d0 / dble(k)
        end do

        allocate(Ginv(MMd))
        Ginv = (/ 1.0d0, 0.3d0, 1.5d0, 0.1d0, 0.2d0, 2.0d0 /)

        allocate(Gmat(MMd))
        Gmat = (/ 2.0d0, 0.4d0, 3.0d0, 0.2d0, 0.1d0, 4.0d0 /)

        allocate(Hmat(MM))
        do k = 1, MM
            Hmat(k) = dble(k) * 0.02d0
        end do

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

    ! --- Test 8: Minimal Md=1 (single fitting function) ---
    subroutine setup_minimal_md_case()
        M = 3;  Md = 1
        MM = M * (M + 1) / 2     ! 6
        MMd = 1                   ! Md*(Md+1)/2 = 1
        kknumd = 2;  kknums = 1

        allocate(cool(kknumd * Md), cools(kknums * Md))
        allocate(kkind(kknumd), kkinds(kknums), af(Md))

        ! cool is (1, 2) matrix: just 2 scalars
        cool(1) = 5.0d0;  cool(2) = 3.0d0
        cools(1) = 2.0

        kkind(1) = 1;  kkind(2) = 3
        kkinds(1) = 2

        allocate(rho(MM))
        rho = (/ 1.0d0, 0.5d0, 2.0d0, 0.3d0, 0.7d0, 1.5d0 /)

        ! Ginv and Gmat are 1x1 packed: just one element each
        allocate(Ginv(MMd), Gmat(MMd))
        Ginv(1) = 0.5d0
        Gmat(1) = 2.0d0

        allocate(Hmat(MM))
        Hmat = (/ 0.1d0, 0.2d0, 0.3d0, 0.4d0, 0.5d0, 0.6d0 /)

        allocate(Fmat(MM), Fmat_b(MM))
        Fmat = 0.0d0;  Fmat_b = 0.0d0
    end subroutine

end program test_int3lu

! Timer stubs (g2g_timer_start/stop are C functions called from Fortran)
subroutine g2g_timer_start(label)
    character(*), intent(in) :: label
end subroutine

subroutine g2g_timer_stop(label)
    character(*), intent(in) :: label
end subroutine
