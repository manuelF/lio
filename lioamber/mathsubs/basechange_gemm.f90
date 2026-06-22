!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! BASECHANGE - manual version -
! BASETRANSFORM PROCEDURES
!
! (1) Initialization of Matm(nnd,ndd) and Mato(nii,ndd)
! (2) First Product Mati(nni,nnd)*Umat(nnd,ndd)
! (3) Second Product Utrp(nii,nni)*Matm(nni,ndd)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
function basechange_d_gemm(M,Mati,Umat,mode) result(Mato)
   implicit none
   integer         , intent(in)  :: M
   character(len=3), intent(in)  :: mode
   LIODBLE    , intent(in)  :: Umat(M,M)
   LIODBLE    , intent(in)  :: Mati(M,M)
   LIODBLE    , allocatable :: Mato(:,:)

   ! Persistent intermediate scratch. Avoids two M*M allocate/deallocate calls
   ! per invocation; Reallocated only when M grows.
   LIODBLE, allocatable, save :: Matm(:,:)
   integer,             save :: cached_M = 0

   if (cached_M /= M) then
      if (allocated(Matm)) deallocate(Matm)
      allocate(Matm(M,M))
      cached_M = M
   endif
   allocate(Mato(M,M))
   ! No zero-init needed: both DGEMMs below use beta=0 and overwrite the
   ! destination outright.

   if (mode == 'inv') then
      call DGEMM('N','N',M,M,M,1.0D0,Umat,M,Mati,M,0.0D0,Matm,M)
      call DGEMM('N','T',M,M,M,1.0D0,Matm,M,Umat,M,0.0D0,Mato,M)
   else
      call DGEMM('T','N',M,M,M,1.0D0,Umat,M,Mati,M,0.0D0,Matm,M)
      call DGEMM('N','N',M,M,M,1.0D0,Matm,M,Umat,M,0.0D0,Mato,M)
   endif
end function

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! In-place congruence transform: Mat <- Umat^T * Mat * Umat ('dir')
! or Mat <- Umat * Mat * Umat^T ('inv'). Bit-identical to basechange_d_gemm
! (same two DGEMM calls, same order) but writes the result straight back into
! Mat, avoiding the allocatable function-result `Mato` (one M*M allocation +
! one M*M copy at the call site) on every invocation. Valid because the second
! DGEMM's destination (Mat) is not one of its operands — A=Matm, B=Umat — so
! overwriting Mat in place cannot corrupt the product.
subroutine basechange_d_gemm_inplace(M, Mat, Umat, mode)
   implicit none
   integer         , intent(in)    :: M
   character(len=3), intent(in)    :: mode
   LIODBLE    , intent(in)    :: Umat(M,M)
   LIODBLE    , intent(inout) :: Mat(M,M)

   ! Persistent intermediate scratch, shared shape-cache with the functional
   ! form's pattern. Reallocated only when M grows.
   LIODBLE, allocatable, save :: Matm(:,:)
   integer,             save :: cached_M = 0

   if (cached_M /= M) then
      if (allocated(Matm)) deallocate(Matm)
      allocate(Matm(M,M))
      cached_M = M
   endif

   if (mode == 'inv') then
      call DGEMM('N','N',M,M,M,1.0D0,Umat,M,Mat ,M,0.0D0,Matm,M)
      call DGEMM('N','T',M,M,M,1.0D0,Matm,M,Umat,M,0.0D0,Mat ,M)
   else
      call DGEMM('T','N',M,M,M,1.0D0,Umat,M,Mat ,M,0.0D0,Matm,M)
      call DGEMM('N','N',M,M,M,1.0D0,Matm,M,Umat,M,0.0D0,Mat ,M)
   endif
end subroutine

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
function basechange_cc_gemm(M,Mati,Umat,mode) result(Mato)
   implicit none
   integer         , intent(in)  :: M
   character(len=3), intent(in)  :: mode
   complex(kind=4) , intent(in)  :: Umat(M,M)
   complex(kind=4) , intent(in)  :: Mati(M,M)
   complex(kind=4) , allocatable :: Matm(:,:)
   complex(kind=4) , allocatable :: Mato(:,:)
   complex(kind=4)               :: alpha, beta

   allocate(Matm(M,M),Mato(M,M))
   Matm  = cmplx(0.0E0,0.0E0)
   Mato  = cmplx(0.0E0,0.0E0)
   alpha = cmplx(1.0E0,0.0E0)
   beta  = cmplx(0.0E0,0.0E0)
   if (mode == 'inv') then
      call CGEMM('N','N',M,M,M,alpha,Umat,M,Mati,M,beta,Matm,M)
      call CGEMM('N','T',M,M,M,alpha,Matm,M,Umat,M,beta,Mato,M)
   else
      call CGEMM('T','N',M,M,M,alpha,Umat,M,Mati,M,beta,Matm,M)
      call CGEMM('N','N',M,M,M,alpha,Matm,M,Umat,M,beta,Mato,M)
   endif
   deallocate(Matm)
end function

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
function basechange_zz_gemm(M, Mati, Umat, mode) result(Mato)
   implicit none
   integer         , intent(in)  :: M
   character(len=3), intent(in)  :: mode
   complex(kind=8) , intent(in)  :: Umat(M,M)
   complex(kind=8) , intent(in)  :: Mati(M,M)
   complex(kind=8) , allocatable :: Matm(:,:)
   complex(kind=8) , allocatable :: Mato(:,:)
   complex(kind=8)               :: alpha, beta

   allocate(Matm(M,M),Mato(M,M))
   Matm  = dcmplx(0.0D0,0.0D0)
   Mato  = dcmplx(0.0D0,0.0D0)
   alpha = dcmplx(1.0D0,0.0D0)
   beta  = dcmplx(0.0D0,0.0D0)

   if (mode == 'inv') then
      call ZGEMM('N','N',M,M,M,alpha,Umat,M,Mati,M,beta,Matm,M)
      call ZGEMM('N','T',M,M,M,alpha,Matm,M,Umat,M,beta,Mato,M)
   else
      call ZGEMM('T','N',M,M,M,alpha,Umat,M,Mati,M,beta,Matm,M)
      call ZGEMM('N','N',M,M,M,alpha,Matm,M,Umat,M,beta,Mato,M)
   endif
   deallocate(Matm)
end function
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
