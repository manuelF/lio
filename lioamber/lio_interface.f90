!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% LIO_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit interfaces for LIO-internal procedures defined as bare external      !
! subroutines/functions (not inside a module), so they cannot be reached via    !
! `use`. They stay bare externals because several are the public driver API     !
! called from liosolo / the MD interface (modularising them would mangle the    !
! link symbol); an interface block keeps the symbol while silencing             !
! -Wimplicit-interface. Declaration-only; matches the existing definitions.     !
!                                                                               !
! Self-reference rule: a procedure that is BOTH defined and called within the   !
! same file imports from here with `use lio_interface, only: <callees>` so its  !
! own name is never brought into its definition scope.                          !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
#include "datatypes/datatypes.fh"
module lio_interface
   implicit none

   interface
      ! ---- convert_types.f90 ----
      function liocmplx(r_part, i_part) result(c_num)
         LIODBLE   :: r_part, i_part
         TDCOMPLEX :: c_num
      end function liocmplx

      ! ---- driver entry points / top-level subroutines ----
      ! NOTE: `scf` lives in the separate scf_interface module because it needs
      ! type(operator); keeping it out of lio_interface avoids a module cycle
      ! (lio_interface is used by low-level liosubs_math/packed_storage, while
      ! typedef_operator's chain reaches those).
      subroutine scf_in(E, qmcoords, clcoords, nsolin, dipxyz)
         integer, intent(in) :: nsolin
         LIODBLE, intent(in) :: qmcoords(3, *), clcoords(4, *)
         LIODBLE :: E, dipxyz(3)
      end subroutine scf_in

      subroutine liomain(E, dipxyz)
         LIODBLE, intent(inout) :: E, dipxyz(3)
      end subroutine liomain

      subroutine drive(iostat)
         integer :: iostat
      end subroutine drive

      subroutine do_restart(UID, rho_total)
         integer, intent(in) :: UID
         LIODBLE, intent(in) :: rho_total(*)
      end subroutine do_restart

      subroutine do_population_analysis(rho_tot, rho_a, rho_b)
         LIODBLE, intent(in) :: rho_tot(*), rho_a(*), rho_b(*)
      end subroutine do_population_analysis

      subroutine do_fukui_calc()
      end subroutine do_fukui_calc

      subroutine do_forces(uid)
         integer, intent(in) :: uid
      end subroutine do_forces

      subroutine recenter_coords(pos_qm, pos_tot, n_qm, n_sol)
         integer, intent(in)    :: n_qm, n_sol
         LIODBLE, intent(inout) :: pos_qm(n_qm, 3), pos_tot(n_qm + n_sol, 3)
      end subroutine recenter_coords

      subroutine read_options(inputFile, extern_stat)
         character(len=20), intent(in)    :: inputFile
         integer, optional, intent(inout) :: extern_stat
      end subroutine read_options

      subroutine lio_defaults()
      end subroutine lio_defaults

      subroutine init_lio_common(natomin, Izin, nclatom, callfrom)
         integer, intent(in) :: natomin, nclatom, callfrom, Izin(natomin)
      end subroutine init_lio_common

      subroutine dft_get_qm_forces(dxyzqm)
         LIODBLE :: dxyzqm(3, *)
      end subroutine dft_get_qm_forces

      subroutine dft_get_mm_forces(dxyzcl, dxyzqm)
         LIODBLE :: dxyzcl(3, *), dxyzqm(3, *)
      end subroutine dft_get_mm_forces

      subroutine get_nco(atom_Z, n_atoms, n_orbitals, n_unpaired, charge, &
                         open_shell, ext_status)
         integer, intent(in)  :: n_atoms, n_unpaired, charge, atom_Z(n_atoms)
         integer, intent(out) :: n_orbitals, ext_status
         logical, intent(in)  :: open_shell
      end subroutine get_nco

      subroutine gridlio()
      end subroutine gridlio

      ! ---- distance-restraint helpers (get_restrain_energy_forces.f90) ----
      subroutine read_restrain_params()
      end subroutine read_restrain_params

      subroutine get_restrain_forces(dxyzqm, f_r)
         LIODBLE :: dxyzqm(3, *), f_r
      end subroutine get_restrain_forces

      subroutine get_restrain_energy(E_restrain)
         LIODBLE :: E_restrain
      end subroutine get_restrain_energy

      ! ---- ECP: generalECP.f90 (generalecp + leaf helpers) ----
      subroutine generalecp(tipodecalculo)
         integer :: tipodecalculo
      end subroutine generalecp

      subroutine search_nan()
      end subroutine search_nan

      subroutine write_post(i)
         integer, intent(in) :: i
      end subroutine write_post

      subroutine write_ecp()
      end subroutine write_ecp

      subroutine write_ecp_parameters()
      end subroutine write_ecp_parameters

      subroutine write_basis()
      end subroutine write_basis

      subroutine write_ang_exp()
      end subroutine write_ang_exp

      subroutine write_distance()
      end subroutine write_distance

      subroutine write_fock_ecp_terms()
      end subroutine write_fock_ecp_terms

      subroutine write_fock_ecp()
      end subroutine write_fock_ecp

      subroutine write_dfock_ecp()
      end subroutine write_dfock_ecp

      subroutine obtaindistance()
      end subroutine obtaindistance

      subroutine obtainls()
      end subroutine obtainls

      subroutine norm_c()
      end subroutine norm_c

      subroutine reasignz()
      end subroutine reasignz

      subroutine read_ecp()
      end subroutine read_ecp

      subroutine deallocatev()
      end subroutine deallocatev

      subroutine allocate_ecp()
      end subroutine allocate_ecp

      ! ---- ECP: readECP.f90 ----
      subroutine lecturaecp()
      end subroutine lecturaecp

      subroutine dataecpelement(Z, elemento)
         integer, intent(in) :: Z
         character(len=3), intent(in) :: elemento
      end subroutine dataecpelement
   end interface

end module lio_interface
