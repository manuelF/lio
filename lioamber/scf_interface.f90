!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% SCF_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interface for the bare-external SCF driver, kept separate from        !
! lio_interface because it needs the typedef_operator derived type. lio_interface!
! is used by low-level modules (liosubs_math, packed_storage) whose dependency   !
! chain is reachable from typedef_operator, so importing the operator type there !
! would create a module cycle. Only SCF's callers use this module.              !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
#include "datatypes/datatypes.fh"
module scf_interface
   use typedef_operator, only: operator
   implicit none

   interface
      subroutine scf(E, fock_aop, rho_aop, fock_bop, rho_bop)
         import :: operator
         LIODBLE        :: E
         type(operator) :: rho_aop, fock_aop
         type(operator), optional :: rho_bop, fock_bop
      end subroutine scf
   end interface

end module scf_interface
