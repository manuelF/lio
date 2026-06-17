!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% GPU_TIMERS_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit Fortran interfaces for the g2g_timer_* timing hooks implemented in   !
! C++ (g2g/timer.cpp, extern "C"). Providing an explicit interface lets the     !
! compiler check the call signatures across the Fortran/C boundary and silences !
! -Wimplicit-interface for every timer call site.                               !
!                                                                               !
! Calling convention note: the C side takes (const char* name, unsigned length).!
! gfortran passes the hidden character length automatically for a character(*)  !
! dummy, so the explicit interface here matches the existing external calls     !
! exactly -- this is a declaration-only change with no codegen/FP impact.        !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module gpu_timers_interface
   implicit none

   interface
      subroutine g2g_timer_start(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_start

      subroutine g2g_timer_stop(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_stop

      subroutine g2g_timer_pause(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_pause

      subroutine g2g_timer_sum_start(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_sum_start

      subroutine g2g_timer_sum_stop(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_sum_stop

      subroutine g2g_timer_sum_pause(timer_name)
         character(len=*), intent(in) :: timer_name
      end subroutine g2g_timer_sum_pause

      subroutine g2g_timer_summary()
      end subroutine g2g_timer_summary
   end interface

end module gpu_timers_interface
