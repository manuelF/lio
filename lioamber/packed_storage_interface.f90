!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% PACKED_STORAGE_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interfaces for the packed<->square storage helpers defined as bare   !
! external subroutines in packed_storage.f90. Silences -Wimplicit-interface at  !
! the (numerous) call sites. Declaration-only; matches the existing definitions.!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
#include "datatypes/datatypes.fh"
module packed_storage_interface
   implicit none

   interface
      subroutine spunpack(UPLO, NM, Vector, Matrix)
         character(len=1) :: UPLO
         integer          :: NM
         LIODBLE          :: Vector(*)
         LIODBLE          :: Matrix(NM, *)
      end subroutine spunpack

      subroutine spunpack_rho(UPLO, NM, Vector, Matrix)
         character(len=1) :: UPLO
         integer          :: NM
         LIODBLE          :: Vector(*)
         LIODBLE          :: Matrix(NM, *)
      end subroutine spunpack_rho

      subroutine sprepack(UPLO, NM, Vector, Matrix)
         character(len=1) :: UPLO
         integer          :: NM
         LIODBLE          :: Vector(*)
         LIODBLE          :: Matrix(NM, *)
      end subroutine sprepack

      subroutine spunpack_rtc(UPLO, NM, Vector, Matrix)
         character(len=1) :: UPLO
         integer          :: NM
         LIODBLE          :: Vector(*)
         TDCOMPLEX        :: Matrix(NM, *)
      end subroutine spunpack_rtc

      subroutine sprepack_ctr(UPLO, NM, Vector, Matrix)
         character(len=1) :: UPLO
         integer          :: NM
         LIODBLE          :: Vector(*)
         TDCOMPLEX        :: Matrix(NM, *)
      end subroutine sprepack_ctr
   end interface

end module packed_storage_interface
