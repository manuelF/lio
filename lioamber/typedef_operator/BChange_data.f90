!carlos: these subroutines allow the base changes of the matrix stored in
!        operator
subroutine BChange_AOtoON_r(this, Xmat, Nsize, invert_mode)
   use typedef_cumat, only: cumat_r

   implicit none
   class(operator) , intent(inout) :: this
   integer         , intent(in)    :: Nsize
   type(cumat_r)   , intent(in)    :: Xmat
   logical         , intent(in), optional :: invert_mode

   character(len=3) :: mode = 'dir'

   if (present(invert_mode)) then
      if (invert_mode) mode = 'inv'
   endif

   ! Seed data_ON with data_AO (handles (re)allocation), then transform it in
   ! place — drops the separate Dmat scratch (one M*M allocation + copy) that
   ! the old path used. Numerically identical: same DGEMMs on the same data.
   call this%Sets_data_ON(this%data_AO)
   call Xmat%change_base(this%data_ON, mode)
end subroutine BChange_AOtoON_r

subroutine BChange_ONtoAO_r(this, Xmat, Nsize, invert_mode)
   use typedef_cumat, only: cumat_r

   implicit none
   class(operator) , intent(inout) :: this
   integer         , intent(in)    :: Nsize
   type(cumat_r)   , intent(in)    :: Xmat
   logical         , intent(in), optional :: invert_mode

   character(len=3) :: mode = 'inv'

   if (present(invert_mode)) then
      if (invert_mode) mode = 'dir'
   endif

   ! Seed data_AO with data_ON, transform in place (see BChange_AOtoON_r).
   call this%Sets_data_AO(this%data_ON)
   call Xmat%change_base(this%data_AO, mode)
 end subroutine BChange_ONtoAO_r

 subroutine BChange_AOtoON_x(this, Xmat, Nsize, invert_mode)
   use typedef_cumat, only: cumat_x

   implicit none
   class(operator) , intent(inout) :: this
   integer         , intent(in)    :: Nsize
   type(cumat_x)   , intent(in)    :: Xmat
   logical         , intent(in), optional :: invert_mode

   character(len=3) :: mode = 'dir'
   TDCOMPLEX   , allocatable :: DmatC(:,:)

   if (present(invert_mode)) then
      if (invert_mode) mode = 'inv'
   endif
   
   allocate(DmatC(Nsize,Nsize))
   DmatC = this%dataC_AO

   call Xmat%change_base(DmatC, mode)
   call this%Sets_dataC_ON(DmatC)
   deallocate(DmatC)

end subroutine BChange_AOtoON_x

subroutine BChange_ONtoAO_x(this, Xmat, Nsize, invert_mode)
   use typedef_cumat, only: cumat_x

   implicit none
   class(operator) , intent(inout) :: this
   integer         , intent(in)    :: Nsize
   type(cumat_x)   , intent(in)    :: Xmat
   logical         , intent(in), optional :: invert_mode

   character(len=3) :: mode = 'inv'
   TDCOMPLEX   , allocatable :: DmatC(:,:)

   if (present(invert_mode)) then
      if (invert_mode) mode = 'dir'
   endif

   allocate(DmatC(Nsize,Nsize))
   DmatC = this%dataC_ON

   call Xmat%change_base(DmatC, mode)
   call this%Sets_dataC_AO(DmatC)
   deallocate(DmatC)
   
 end subroutine BChange_ONtoAO_x