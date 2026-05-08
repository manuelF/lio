!carlos: this subroutine diagonalises the matrix stored in this.
subroutine Diagon_datamat (this, eigen_vecs, eigen_vals)
   use linear_algebra, only: matrix_diagon

   implicit none
   class(operator), intent(in) :: this
   LIODBLE, intent(out) :: eigen_vecs(:,:)
   LIODBLE, intent(out) :: eigen_vals(:)

   ! matrix_diagon copies input to eigen_vecs internally before calling
   ! DSYEVD (which overwrites its input). Pass data_ON directly.
   call matrix_diagon( this%data_ON, eigen_vecs, eigen_vals )

end subroutine Diagon_datamat
