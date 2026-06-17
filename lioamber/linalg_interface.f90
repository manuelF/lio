!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% LINALG_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interfaces for the external BLAS / LAPACK routines used across       !
! lioamber. Reference Netlib signatures (assumed-size arrays, so any           !
! contiguous actual argument associates by sequence). Providing the interface  !
! lets the compiler type-check the calls and silences -Wimplicit-interface.    !
! Declaration-only: the calling convention is identical to the previous        !
! implicit external calls (no codegen / FP impact).                            !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module linalg_interface
   implicit none

   interface
      ! ---- BLAS level 1 ----
      double precision function ddot(n, dx, incx, dy, incy)
         integer :: n, incx, incy
         double precision :: dx(*), dy(*)
      end function ddot

      subroutine daxpy(n, da, dx, incx, dy, incy)
         integer :: n, incx, incy
         double precision :: da, dx(*)
         double precision :: dy(*)
      end subroutine daxpy

      subroutine caxpy(n, ca, cx, incx, cy, incy)
         integer :: n, incx, incy
         complex(kind=4) :: ca, cx(*)
         complex(kind=4) :: cy(*)
      end subroutine caxpy

      ! ---- BLAS level 2 ----
      subroutine dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
         character :: trans
         integer :: m, n, lda, incx, incy
         double precision :: alpha, beta, a(lda, *), x(*)
         double precision :: y(*)
      end subroutine dgemv

      subroutine sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
         character :: trans
         integer :: m, n, lda, incx, incy
         real(kind=4) :: alpha, beta, a(lda, *), x(*)
         real(kind=4) :: y(*)
      end subroutine sgemv

      subroutine dspmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
         character :: uplo
         integer :: n, incx, incy
         double precision :: alpha, beta, ap(*), x(*)
         double precision :: y(*)
      end subroutine dspmv

      ! ---- BLAS level 3 ----
      subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
                       c, ldc)
         character :: transa, transb
         integer :: m, n, k, lda, ldb, ldc
         double precision :: alpha, beta, a(lda, *), b(ldb, *)
         double precision :: c(ldc, *)
      end subroutine dgemm

      subroutine cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
                       c, ldc)
         character :: transa, transb
         integer :: m, n, k, lda, ldb, ldc
         complex(kind=4) :: alpha, beta, a(lda, *), b(ldb, *)
         complex(kind=4) :: c(ldc, *)
      end subroutine cgemm

      subroutine zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, &
                       c, ldc)
         character :: transa, transb
         integer :: m, n, k, lda, ldb, ldc
         complex(kind=8) :: alpha, beta, a(lda, *), b(ldb, *)
         complex(kind=8) :: c(ldc, *)
      end subroutine zgemm

      ! ---- LAPACK: eigensolvers ----
      subroutine dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
         character :: jobz, uplo
         integer :: n, lda, lwork
         integer :: info
         double precision :: a(lda, *)
         double precision :: w(*)
         double precision :: work(*)
      end subroutine dsyev

      subroutine dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, &
                        info)
         character :: jobz, uplo
         integer :: n, lda, lwork, liwork
         integer :: info
         integer :: iwork(*)
         double precision :: a(lda, *)
         double precision :: w(*)
         double precision :: work(*)
      end subroutine dsyevd

      subroutine dsyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &
                        m, w, z, ldz, isuppz, work, lwork, iwork, liwork, info)
         character :: jobz, range, uplo
         integer :: n, lda, il, iu, ldz, lwork, liwork
         integer :: m, info
         integer :: isuppz(*)
         integer :: iwork(*)
         double precision :: vl, vu, abstol
         double precision :: a(lda, *)
         double precision :: w(*), z(ldz, *)
         double precision :: work(*)
      end subroutine dsyevr

      subroutine dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, &
                       info)
         character :: jobz, uplo
         integer :: itype, n, lda, ldb, lwork
         integer :: info
         double precision :: a(lda, *), b(ldb, *)
         double precision :: w(*)
         double precision :: work(*)
      end subroutine dsygv

      double precision function dlamch(cmach)
         character :: cmach
      end function dlamch

      ! ---- LAPACK: least squares / factorizations / inverses ----
      subroutine dgels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info)
         character :: trans
         integer :: m, n, nrhs, lda, ldb, lwork
         integer :: info
         double precision :: a(lda, *), b(ldb, *), work(*)
      end subroutine dgels

      subroutine dpotrf(uplo, n, a, lda, info)
         character :: uplo
         integer :: n, lda
         integer :: info
         double precision :: a(lda, *)
      end subroutine dpotrf

      subroutine dpotri(uplo, n, a, lda, info)
         character :: uplo
         integer :: n, lda
         integer :: info
         double precision :: a(lda, *)
      end subroutine dpotri

      subroutine dtrtri(uplo, diag, n, a, lda, info)
         character :: uplo, diag
         integer :: n, lda
         integer :: info
         double precision :: a(lda, *)
      end subroutine dtrtri

      subroutine dgetrf(m, n, a, lda, ipiv, info)
         integer :: m, n, lda
         integer :: ipiv(*), info
         double precision :: a(lda, *)
      end subroutine dgetrf

      subroutine dgetri(n, a, lda, ipiv, work, lwork, info)
         integer :: n, lda, lwork
         integer :: ipiv(*)
         integer :: info
         double precision :: a(lda, *), work(*)
      end subroutine dgetri

      subroutine dgeqrf(m, n, a, lda, tau, work, lwork, info)
         integer :: m, n, lda, lwork
         integer :: info
         double precision :: a(lda, *)
         double precision :: tau(*)
         double precision :: work(*)
      end subroutine dgeqrf

      subroutine dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, &
                        iwork, info)
         character :: jobz
         integer :: m, n, lda, ldu, ldvt, lwork
         integer :: iwork(*)
         integer :: info
         double precision :: a(lda, *)
         double precision :: s(*), u(ldu, *), vt(ldvt, *)
         double precision :: work(*)
      end subroutine dgesdd
   end interface

end module linalg_interface
