! This exchanges the data from another CUMAT with the one from the
! current CUMAT.
! REAL
subroutine exchange_r(this, bmat)
   implicit none
   type(cumat_r) , intent(inout) :: bmat
   class(cumat_r), intent(inout) :: this
   LIODBLE  , allocatable   :: tmp_array(:,:)
#ifdef CUBLAS
   CUDAPTR               :: tmp_pointer

   tmp_pointer     = bmat%cu_pointer
   bmat%cu_pointer = this%cu_pointer
   this%cu_pointer = tmp_pointer

   if ((.not. this%gpu_only) .and. (.not. bmat%gpu_only)) then
      call move_alloc(this%matrix, tmp_array)
      call move_alloc(bmat%matrix, this%matrix)
      call move_alloc(tmp_array,   bmat%matrix)
   endif

#else
   call move_alloc(this%matrix, tmp_array)
   call move_alloc(bmat%matrix, this%matrix)
   call move_alloc(tmp_array,   bmat%matrix)
#endif

end subroutine exchange_r

! COMPLEX
subroutine exchange_x(this, bmat)
   implicit none
   type(cumat_x) , intent(inout) :: bmat
   class(cumat_x), intent(inout) :: this
   TDCOMPLEX     , allocatable   :: tmp_array(:,:)
#ifdef CUBLAS
   CUDAPTR               :: tmp_pointer

   tmp_pointer     = bmat%cu_pointer
   bmat%cu_pointer = this%cu_pointer
   this%cu_pointer = tmp_pointer

   if ((.not. this%gpu_only) .and. (.not. bmat%gpu_only)) then
      call move_alloc(this%matrix, tmp_array)
      call move_alloc(bmat%matrix, this%matrix)
      call move_alloc(tmp_array,   bmat%matrix)
   endif

#else
   call move_alloc(this%matrix, tmp_array)
   call move_alloc(bmat%matrix, this%matrix)
   call move_alloc(tmp_array,   bmat%matrix)
#endif

end subroutine exchange_x
