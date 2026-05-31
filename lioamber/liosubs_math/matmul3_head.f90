!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
function matmul3_ddd( Amat, Bmat, Cmat ) result( Dmat )
!  Dmat = Amat * Bmat * Cmat, evaluated as two DGEMM calls. gfortran's
!  intrinsic matmul is its own (non-BLAS) routine and is several times slower
!  than DGEMM at the M~1e3 sizes this is used at (the overlap orthogonalization
!  spent ~15 s here on a M=2600 case). The complex variant below keeps the
!  generic matmul body.
!  NOTE: the DGEMM leading dimensions assume contiguous arguments (LDA =
!  size(,1)); all current callers pass full arrays. A future caller passing a
!  non-contiguous array slice would need a packed copy first.
   implicit none
   LIODBLE, intent(in)  :: Amat(:,:)
   LIODBLE, intent(in)  :: Bmat(:,:)
   LIODBLE, intent(in)  :: Cmat(:,:)
   LIODBLE, allocatable :: Dmat(:,:)
   LIODBLE, allocatable :: Xmat(:,:)
   logical :: error_found
   integer :: ma, ka, kb, nb, nc, pc

   ma = size(Amat,1); ka = size(Amat,2)
   kb = size(Bmat,1); nb = size(Bmat,2)
   nc = size(Cmat,1); pc = size(Cmat,2)

   error_found = .false.
   error_found = (error_found) .or. ( ka /= kb )
   error_found = (error_found) .or. ( nb /= nc )
   if (error_found) then
      print*, 'ERROR INSIDE matmul3_ddd'
      print*, 'Wrong sizes of input/output'
      print*; stop
   endif

   allocate( Xmat(ma, nb) )
   if (allocated(Dmat)) deallocate(Dmat)
   allocate( Dmat(ma, pc) )

   call DGEMM('N', 'N', ma, nb, ka, 1.0D0, Amat, ma, Bmat, kb, 0.0D0, Xmat, ma)
   call DGEMM('N', 'N', ma, pc, nb, 1.0D0, Xmat, ma, Cmat, nc, 0.0D0, Dmat, ma)

   deallocate( Xmat )
end function matmul3_ddd

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
function matmul3_dcd( A_in, Bmat, C_in ) result( Dmat )
   implicit none
   LIODBLE, intent(in)  :: A_in(:,:)
   TDCOMPLEX   , intent(in)  :: Bmat(:,:)
   LIODBLE, intent(in)  :: C_in(:,:)
   TDCOMPLEX, allocatable :: Dmat(:,:)
   TDCOMPLEX, allocatable :: Xmat(:,:), Amat(:,:), Cmat(:,:)
   logical :: error_found
   integer :: ii, jj
   TDCOMPLEX :: liocmplx

   ! This is necessary to avoid wrong type conversions in matmul.
   allocate(Amat(size(A_in,1), size(A_in,2)))
   do ii = 1, size(A_in,1)
   do jj = 1, size(A_in,2)
      Amat(ii,jj) = liocmplx(A_in(ii,jj),0.0D0)
   enddo
   enddo

   allocate(Amat(size(C_in,1), size(C_in,2)))
   do ii = 1, size(C_in,1)
   do jj = 1, size(C_in,2)
      Cmat(ii,jj) = liocmplx(C_in(ii,jj),0.0D0)
   enddo
   enddo

#  include "matmul3_body.f90"
   deallocate(Amat,Cmat)
end function matmul3_dcd

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
