!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% GPU_INTERFACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! Explicit Fortran interfaces for the g2g_* / aint_* / int3lu_gpu_* C hooks     !
! (extern "C", trailing-underscore convention) implemented on the C++ side in   !
! g2g/. Mapping: C `type&` -> scalar dummy, C `type*` -> assumed-size array     !
! dummy (or scalar where the caller passes a scalar by reference). int->integer,!
! unsigned int/uint->integer, double->real(8), float->real(4), bool->logical.   !
! Declaration-only: identical calling convention to the previous implicit       !
! external calls (no codegen / FP impact); silences -Wimplicit-interface.       !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module gpu_interface
   implicit none

   interface
      ! ---------------------------- aint_* --------------------------------- !
      subroutine aint_coulomb_fock(Es)
         real(kind=8) :: Es
      end subroutine aint_coulomb_fock

      subroutine aint_coulomb_forces(qm_forces)
         real(kind=8) :: qm_forces(*)
      end subroutine aint_coulomb_forces

      subroutine aint_coulomb_init()
      end subroutine aint_coulomb_init

      subroutine aint_deinit()
      end subroutine aint_deinit

      subroutine aint_new_step()
      end subroutine aint_new_step

      subroutine aint_parameter_init(Md, ncontd, nshelld, cd, ad, Nucd, af, &
                                     G_inv, Hmat_vec, str, fac, rmax, atomZ_i, &
                                     level_of_gpu)
         integer :: Md, level_of_gpu
         integer :: ncontd(*), nshelld(*), Nucd(*), atomZ_i(*)
         real(kind=8) :: cd(*), ad(*), af(*), G_inv(*), Hmat_vec(*), str(*), &
                         fac(*), rmax
      end subroutine aint_parameter_init

      subroutine aint_qmmm_fock(Es, Ens)
         real(kind=8) :: Es, Ens
      end subroutine aint_qmmm_fock

      subroutine aint_qmmm_forces(qm_forces, mm_forces)
         real(kind=8) :: qm_forces(*), mm_forces(*)
      end subroutine aint_qmmm_forces

      subroutine aint_qmmm_init(nclatom, r_all, pc)
         integer :: nclatom
         real(kind=8) :: r_all(*), pc(*)
      end subroutine aint_qmmm_init

      subroutine aint_query_gpu_level(gpu_level_out)
         integer :: gpu_level_out
      end subroutine aint_query_gpu_level

      ! ---------------------------- g2g_* ---------------------------------- !
      subroutine g2g_calcgammcou(rhoG, Zmat, gamm)
         real(kind=8) :: rhoG(*), Zmat(*), gamm(*)
      end subroutine g2g_calcgammcou

      subroutine g2g_calcgradxc(P, V, F, met)
         real(kind=8) :: P(*), V(*), F(*)
         integer :: met
      end subroutine g2g_calcgradxc

      subroutine g2g_calculate2e(tao, fock, vecdim)
         real(kind=8) :: tao(*), fock(*)
         integer :: vecdim
      end subroutine g2g_calculate2e

      subroutine g2g_calculateg(Tmat, F, DER)
         real(kind=8) :: Tmat(*), F(*)
         integer :: DER
      end subroutine g2g_calculateg

      subroutine g2g_calculatexc(Tmat, Fv)
         real(kind=8) :: Tmat(*), Fv(*)
      end subroutine g2g_calculatexc

      subroutine g2g_cdft_finalise()
      end subroutine g2g_cdft_finalise

      subroutine g2g_cdft_init(do_c, do_s, regions, max_nat, natoms, at_list)
         logical :: do_c, do_s
         integer :: regions, max_nat
         integer :: natoms(*), at_list(*)
      end subroutine g2g_cdft_init

      subroutine g2g_cdft_set_v(Vc, Vs)
         real(kind=8) :: Vc(*), Vs(*)
      end subroutine g2g_cdft_set_v

      subroutine g2g_cdft_w(fort_W)
         real(kind=8) :: fort_W(*)
      end subroutine g2g_cdft_w

      subroutine g2g_cioverlap(wfunc, wfunc_old, coef, coef_old, Sbig, sigma, &
                               kind_coupling, phases, phases_old, M, NCO, &
                               Nvirt, nstates, ndets)
         real(kind=8) :: wfunc(*), wfunc_old(*), coef(*), coef_old(*), Sbig(*), &
                         sigma(*), phases(*), phases_old(*)
         integer :: kind_coupling(*)
         integer :: M, NCO, Nvirt, nstates, ndets
      end subroutine g2g_cioverlap

      subroutine g2g_deinit()
      end subroutine g2g_deinit

      subroutine g2g_exacgrad_excited(rhoG, DiffExc, Xmat, fEE)
         real(kind=8) :: rhoG(*), DiffExc(*), Xmat(*), fEE(*)
      end subroutine g2g_exacgrad_excited

      subroutine g2g_exact_exchange(rho, fock, op)
         real(kind=8) :: rho(*), fock(*)
         integer :: op
      end subroutine g2g_exact_exchange

      subroutine g2g_exact_exchange_gradient(rho, frc, op)
         real(kind=8) :: rho(*), frc(*)
         integer :: op
      end subroutine g2g_exact_exchange_gradient

      ! Callers pass a 5th scalar (op) that the C side currently ignores; it is
      ! declared here to match the existing call sites.
      subroutine g2g_exact_exchange_open(rhoA, rhoB, fockA, fockB, op)
         real(kind=8) :: rhoA(*), rhoB(*), fockA(*), fockB(*)
         integer :: op
      end subroutine g2g_exact_exchange_open

      subroutine g2g_extern_functional(main_id, externFunc, HF, HF_fac, screen)
         integer :: main_id
         logical :: externFunc
         integer :: HF(*)
         real(kind=8) :: HF_fac(*), screen
      end subroutine g2g_extern_functional

      subroutine g2g_get_becke_dens(fort_becke)
         real(kind=8) :: fort_becke(*)
      end subroutine g2g_get_becke_dens

      subroutine g2g_get_becke_spin(fort_becke)
         real(kind=8) :: fort_becke(*)
      end subroutine g2g_get_becke_spin

      subroutine g2g_init()
      end subroutine g2g_init

      subroutine g2g_libint_init(Cbas, recalc, idd)
         real(kind=8) :: Cbas(*)
         integer :: recalc, idd
      end subroutine g2g_libint_init

      subroutine g2g_ls_energy(lambda, Ex)
         real(kind=8) :: lambda, Ex
      end subroutine g2g_ls_energy

      integer function g2g_ls_set_endpoints(rho0, rho1)
         real(kind=8) :: rho0(*), rho1(*)
      end function g2g_ls_set_endpoints

      integer function g2g_ls_set_endpoints_open(rho0a, rho0b, rho1a, rho1b)
         real(kind=8) :: rho0a(*), rho0b(*), rho1a(*), rho1b(*)
      end function g2g_ls_set_endpoints_open

      subroutine g2g_new_grid(grid_type)
         integer :: grid_type
      end subroutine g2g_new_grid

      subroutine g2g_parameter_init(norm, natom, max_atoms, ngaussians, r, Rm, &
                 Iz, Nr, Nr2, Nuc, M, ncont, nshell, c, a, rho_vec, fock_vec,  &
                 fockb_vec, rhoalpha, rhobeta, nco, OPEN, nunp, nopt, Iexch, e,&
                 e2, e3, wang, wang2, wang3, use_libxc, ex_functional_id,      &
                 ec_functional_id, becke)
         integer :: natom, max_atoms, ngaussians, M, nco, nunp, nopt
         integer :: Iexch, ex_functional_id, ec_functional_id
         integer :: Iz(*), Nr(*), Nr2(*), Nuc(*), ncont(*), nshell(*)
         logical :: norm, OPEN, use_libxc, becke
         real(kind=8) :: r(*), Rm(*), c(*), a(*), rho_vec(*), fock_vec(*)
         real(kind=8) :: fockb_vec(*), rhoalpha(*), rhobeta(*)
         real(kind=8) :: e(*), e2(*), e3(*), wang(*), wang2(*), wang3(*)
      end subroutine g2g_parameter_init

      integer function g2g_recommended_blas_threads()
      end function g2g_recommended_blas_threads

      integer function g2g_recommended_omp_threads()
      end function g2g_recommended_omp_threads

      integer function g2g_gpu_threads()
      end function g2g_gpu_threads

      subroutine g2g_reload_atom_positions(grid_type, atom_Z_in)
         integer :: grid_type
         integer :: atom_Z_in(*)
      end subroutine g2g_reload_atom_positions

      subroutine g2g_saverho()
      end subroutine g2g_saverho

      subroutine g2g_set_options(fort_fgm, fort_lcs, fort_sr, fort_aaf, &
                                 fort_eai, fort_rzw, fort_mppc, fort_mfe, &
                                 fort_time, fort_verbose)
         real(kind=8) :: fort_fgm, fort_lcs, fort_sr
         logical :: fort_aaf, fort_eai, fort_rzw
         integer :: fort_mppc, fort_mfe, fort_time, fort_verbose
      end subroutine g2g_set_options

      subroutine g2g_set_td_merge(enable)
         logical :: enable
      end subroutine g2g_set_td_merge

      ! fort_energy_ptr / fort_forces_ptr are passed by reference (C double*);
      ! callers pass either a scalar energy / forces-array base element, or a
      ! 0.0D0 sentinel when that output is unused. Declared scalar so both the
      ! sentinel and an array-element actual argument associate correctly.
      subroutine g2g_solve_groups(computation_type, fort_energy_ptr, &
                                  fort_forces_ptr)
         integer :: computation_type
         real(kind=8) :: fort_energy_ptr, fort_forces_ptr
      end subroutine g2g_solve_groups

      subroutine g2g_solve_groups_into(computation_type, fort_energy_ptr, &
                                       fort_forces_ptr, fock_buffer)
         integer :: computation_type
         real(kind=8) :: fort_energy_ptr, fort_forces_ptr, fock_buffer(*)
      end subroutine g2g_solve_groups_into

      subroutine g2g_solve_groups_into_open(computation_type, fort_energy_ptr, &
                                            fort_forces_ptr, fock_buffer_a, &
                                            fock_buffer_b)
         integer :: computation_type
         real(kind=8) :: fort_energy_ptr, fort_forces_ptr, fock_buffer_a(*), &
                         fock_buffer_b(*)
      end subroutine g2g_solve_groups_into_open

      ! ------------------------- int3lu_gpu_* ------------------------------ !
      subroutine int3lu_gpu_invalidate()
      end subroutine int3lu_gpu_invalidate

      subroutine int3lu_gpu_prefetch(cool, cools, Md, kknumd, kknums)
         real(kind=8) :: cool(*)
         real(kind=4) :: cools(*)
         integer :: Md, kknumd, kknums
      end subroutine int3lu_gpu_prefetch

      subroutine int3lu_gpu_ensure(cool, cools, Md, kknumd, kknums, ok)
         real(kind=8) :: cool(*)
         real(kind=4) :: cools(*)
         integer :: Md, kknumd, kknums, ok
      end subroutine int3lu_gpu_ensure

      subroutine int3lu_gpu_rc(cool, cools, Gmat, Gmats)
         real(kind=8) :: cool(*), Gmat(*)
         real(kind=4) :: cools(*), Gmats(*)
      end subroutine int3lu_gpu_rc

      subroutine int3lu_gpu_terms(cool, cools, term, terms)
         real(kind=8) :: cool(*), term(*)
         real(kind=4) :: cools(*), terms(*)
      end subroutine int3lu_gpu_terms
   end interface

end module gpu_interface
