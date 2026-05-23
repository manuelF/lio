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
   ! AB_BAmat = A*B - B*A computed via two DGEMMs, avoiding the
   ! matmul-internal path and the temporary copies (Amat/ABmat/BAmat plus
   ! the function-return assignment) the previous version allocated.
   call DGEMM('N','N',Nsize,Nsize,Nsize, 1.0d0, this%data_ON, Nsize, &
              Bmat, Nsize, 0.0d0, AB_BAmat, Nsize)
   call DGEMM('N','N',Nsize,Nsize,Nsize,-1.0d0, Bmat, Nsize, &
              this%data_ON, Nsize, 1.0d0, AB_BAmat, Nsize)
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
