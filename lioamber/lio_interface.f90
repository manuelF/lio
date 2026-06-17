!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% LIO_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interfaces for LIO-internal procedures defined as bare external      !
! subroutines/functions (i.e. not inside a module), so they cannot be reached   !
! via `use`. These stay bare externals because several are part of the public   !
! driver API called from liosolo / the MD interface; an interface block keeps   !
! the linkage symbol unchanged while silencing -Wimplicit-interface.            !
! Declaration-only; matches the existing definitions.                           !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
#include "datatypes/datatypes.fh"
module lio_interface
   implicit none

   interface
      ! convert_types.f90
      function liocmplx(r_part, i_part) result(c_num)
         LIODBLE   :: r_part, i_part
         TDCOMPLEX :: c_num
      end function liocmplx
   end interface

end module lio_interface
