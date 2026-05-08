!carlos: this subroutine builds the density matrix.
subroutine Dens_build(this, Msize, Nocup, Focup, coef_mat)
   implicit none
   class(operator), intent(inout) :: this
   integer, intent(in)            :: Msize
   integer, intent(in)            :: Nocup
   LIODBLE , intent(in)            :: Focup
   LIODBLE , intent(in)            :: coef_mat(Msize, Msize)

   !  Obtains data_AO as Focup * coef_mat(:,1:Nocup) * coef_mat(:,1:Nocup)^T.
   !  DGEMM reads only the first Nocup columns via the K parameter,
   !  so no temporary copy of occupied orbitals is needed.
   call DGEMM('N', 'T', Msize, Msize, Nocup, Focup, coef_mat, Msize, &
              coef_mat, Msize, 0.0D0, this%data_AO, Msize)

end subroutine Dens_build
