!carlos: for the moment these subroutines only conmut the ON and real type matrix
subroutine Commut_data_r(this, Bmat, AB_BAmat, Nsize)

#ifdef CUBLAS
   use cublasmath, only : commutator_cublas
#endif

   implicit none
   class(operator), intent(inout) :: this
   integer, intent(in)            :: Nsize
   LIODBLE, intent(in)             :: Bmat(Nsize,Nsize)
   LIODBLE, intent(inout)           :: AB_BAmat(Nsize,Nsize)

#ifdef CUBLAS
   LIODBLE, allocatable :: Amat(:,:)
   allocate(Amat(Nsize,Nsize))
   Amat = this%data_ON
   AB_BAmat = commutator_cublas(Amat, Bmat)
#else
   ! AB_BAmat = A*B - B*A with A = F'(ON), B = P'(ON), both symmetric in the
   ! orthonormal basis. Since B*A = (A*B)^T for symmetric A,B, the second
   ! (full M^3) DGEMM is replaced by an O(M^2) antisymmetrization of the first
   ! product: AB_BAmat = AB - AB^T. Halves the commutator's BLAS-3 work.
   ! NOTE: A,B are symmetric only up to the ulp-level asymmetry left by their
   ! DGEMM base changes, so this is not bit-identical to the 2-DGEMM form.
   integer :: ii, jj
   LIODBLE :: aij, aji
   call DGEMM('N','N',Nsize,Nsize,Nsize, 1.0d0, this%data_ON, Nsize, &
              Bmat, Nsize, 0.0d0, AB_BAmat, Nsize)
   do jj = 1, Nsize
      AB_BAmat(jj,jj) = 0.0d0
      do ii = jj+1, Nsize
         aij = AB_BAmat(ii,jj)
         aji = AB_BAmat(jj,ii)
         AB_BAmat(ii,jj) = aij - aji
         AB_BAmat(jj,ii) = aji - aij
      enddo
   enddo
#endif

end subroutine Commut_data_r

subroutine Commut_data_c(this, Bmat, AB_BAmat, Nsize)

#ifdef CUBLAS
   use cublasmath, only : commutator_cublas
#else
   use mathsubs,   only: commutator
#endif

   implicit none
   class(operator), intent(inout) :: this
   integer, intent(in)            :: Nsize

   TDCOMPLEX, intent(in)       :: Bmat(Nsize,Nsize)
   TDCOMPLEX, intent(out)      :: AB_BAmat(Nsize,Nsize)
   TDCOMPLEX, allocatable      :: ABmat(:,:)
   TDCOMPLEX, allocatable      :: BAmat(:,:)

   LIODBLE, allocatable :: Amat(:,:)

   allocate(Amat(Nsize,Nsize), ABmat(Nsize,Nsize), BAmat(Nsize,Nsize))

   Amat=this%data_ON

#ifdef CUBLAS
      AB_BAmat = commutator_cublas(Amat, Bmat)
#else
      AB_BAmat = commutator (Amat, Bmat)
#endif

end subroutine Commut_data_c
