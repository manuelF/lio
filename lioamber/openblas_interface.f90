!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% OPENBLAS_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interfaces for the OpenBLAS thread-control extensions (Fortran       !
! wrappers exported by libopenblas, called by reference). Not part of the       !
! standard BLAS interface, hence kept separate from linalg_interface. Only used !
! when libopenblas is the BLAS provider (see Makefile.options note on the       !
! -lopenblas / LIO_OVERLAP_INT3LU_G2G overlap path). Declaration-only.          !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module openblas_interface
   implicit none

   interface
      subroutine openblas_set_num_threads(num_threads)
         integer :: num_threads
      end subroutine openblas_set_num_threads

      integer function openblas_get_num_threads()
      end function openblas_get_num_threads
   end interface

end module openblas_interface
