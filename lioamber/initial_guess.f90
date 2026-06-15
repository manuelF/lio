#include "datatypes/datatypes.fh"
!## INITIAL_GUESS.F90 #########################################################!
! This file contains data and subroutines used to perform the initial guess in !
! SCF. Input variable "initial_guess" is cointained here.                      !
! Modules contained:                                                           !
!       (*) initial_guess_data                                                 !
!       (*) initial_guess_subs                                                 !
!                                                                              !
! Externally accessed subroutines:                                             !
!       (*) get_initial_guess                                                  !
! This subroutine receives the following variables as input:                   !
!   M:         (integer) Number of basis functions.                            !
!   MM:        (integer) The size of vectorised matrices,  M*(M+1)/2.          !
!   NCO:       (integer) Number of occupied orbitals (alpha orbitals in OS).   !
!   NCOb:      (integer) Number of occupied beta orbitals (only used in OS).   !
!   Xmat:      (dp real, size(M,M)) The basechange matrix from atomic to       !
!              canonical basis.                                                !
!   Hvec:      (dp real, size(MM)) Vectorised 1-e integrals (only for 1e       !
!              integral-type guess).                                           !
!   openshell: (logical) Indicates if the calculation is OS.                   !
!   natom:     (integer) Total number of atoms (used in aufbau-type guess).    !
!   Iz:        (integer, size(natom)) Contains the system's atomic numbers.    !
!   nshell:    (integer, size(0:2)) Contains the number of s, p, and d basis.  !
!   Nuc:       (integer, size(M)) For each i basis function, Nuc(i) indicates  !
!              the atom in which that function is centered.                    !
! The following are the outputs:                                               !
!   Rhovec:   (dp real, size(MM)) Vectorised density matrix.                   !
!   Rhoalpha: (dp real, size(MM)) Vectorised alpha density matrix. (only used  !
!             in OS).                                                          !
!   Rhobeta: (dp real, size(MM)) Vectorised beta density matrix (only used in  !
!            OS).                                                              !
!                                                                              !
! Internal subroutines:                                                        !
!        (*) initialise_eec       (initialises EEC array for aufbau-type guess)!
!        (*) initial_guess_aufbau (performs aufbau-type guess)                 !
!        (*) initial_guess_1e     (performs 1e-integral-type guess)            !
!                                                                              !
! Last modified: 28/03/18 Federico N. Pedron                                   !
!##############################################################################!
module initial_guess_data
  integer :: initial_guess = 0
  integer :: atomic_eec(0:54,3)

contains
   subroutine initialise_eec()
      implicit none

      atomic_eec(0:54,:) = 0
      atomic_eec(1,1)  = 1 ;                                               ! H
      atomic_eec(2,1)  = 2 ;                                               ! He
      atomic_eec(3,1)  = 3 ;                                               ! Li
      atomic_eec(4,1)  = 4 ;                                               ! Be
      atomic_eec(5,1)  = 4 ; atomic_eec(5,2)  = 1 ;                        ! B
      atomic_eec(6,1)  = 4 ; atomic_eec(6,2)  = 2 ;                        ! C
      atomic_eec(7,1)  = 4 ; atomic_eec(7,2)  = 3 ;                        ! N
      atomic_eec(8,1)  = 4 ; atomic_eec(8,2)  = 4 ;                        ! O
      atomic_eec(9,1)  = 4 ; atomic_eec(9,2)  = 5 ;                        ! F
      atomic_eec(10,1) = 4 ; atomic_eec(10,2) = 6 ;                        ! Ne
      atomic_eec(11,1) = 5 ; atomic_eec(11,2) = 6 ;                        ! Na
      atomic_eec(12,1) = 6 ; atomic_eec(12,2) = 6 ;                        ! Mg
      atomic_eec(13,1) = 6 ; atomic_eec(13,2) = 7 ;                        ! Al
      atomic_eec(14,1) = 6 ; atomic_eec(14,2) = 8 ;                        ! Si
      atomic_eec(15,1) = 6 ; atomic_eec(15,2) = 9 ;                        ! P
      atomic_eec(16,1) = 6 ; atomic_eec(16,2) = 10;                        ! S
      atomic_eec(17,1) = 6 ; atomic_eec(17,2) = 11;                        ! Cl
      atomic_eec(18,1) = 6 ; atomic_eec(18,2) = 12;                        ! Ar
      atomic_eec(19,1) = 7 ; atomic_eec(19,2) = 12;                        ! K
      atomic_eec(20,1) = 8 ; atomic_eec(20,2) = 12;                        ! Ca
      atomic_eec(21,1) = 8 ; atomic_eec(21,2) = 12; atomic_eec(21,3) = 1 ; ! Sc
      atomic_eec(22,1) = 8 ; atomic_eec(22,2) = 12; atomic_eec(22,3) = 2 ; ! Ti
      atomic_eec(23,1) = 8 ; atomic_eec(23,2) = 12; atomic_eec(23,3) = 3 ; ! V
      atomic_eec(24,1) = 7 ; atomic_eec(24,2) = 12; atomic_eec(24,3) = 5 ; ! Cr
      atomic_eec(25,1) = 8 ; atomic_eec(25,2) = 12; atomic_eec(25,3) = 5 ; ! Mn
      atomic_eec(26,1) = 8 ; atomic_eec(26,2) = 12; atomic_eec(26,3) = 6 ; ! Fe
      atomic_eec(27,1) = 8 ; atomic_eec(27,2) = 12; atomic_eec(27,3) = 7 ; ! Co
      atomic_eec(28,1) = 8 ; atomic_eec(28,2) = 12; atomic_eec(28,3) = 8 ; ! Ni
      atomic_eec(29,1) = 7 ; atomic_eec(29,2) = 12; atomic_eec(29,3) = 10; ! Cu
      atomic_eec(30,1) = 8 ; atomic_eec(30,2) = 12; atomic_eec(30,3) = 10; ! Zn
      atomic_eec(31,1) = 8 ; atomic_eec(31,2) = 13; atomic_eec(31,3) = 10; ! Ga
      atomic_eec(32,1) = 8 ; atomic_eec(32,2) = 14; atomic_eec(32,3) = 10; ! Ge
      atomic_eec(33,1) = 8 ; atomic_eec(33,2) = 15; atomic_eec(33,3) = 10; ! As
      atomic_eec(34,1) = 8 ; atomic_eec(34,2) = 16; atomic_eec(34,3) = 10; ! Se
      atomic_eec(35,1) = 8 ; atomic_eec(35,2) = 17; atomic_eec(35,3) = 10; ! Br
      atomic_eec(36,1) = 8 ; atomic_eec(36,2) = 18; atomic_eec(36,3) = 10; ! Kr
      atomic_eec(37,1) = 9 ; atomic_eec(37,2) = 18; atomic_eec(37,3) = 10; ! Rb
      atomic_eec(38,1) = 10; atomic_eec(38,2) = 18; atomic_eec(38,3) = 10; ! Sr
      atomic_eec(39,1) = 10; atomic_eec(39,2) = 18; atomic_eec(39,3) = 11; ! Y
      atomic_eec(40,1) = 10; atomic_eec(40,2) = 18; atomic_eec(40,3) = 12; ! Zr
      atomic_eec(41,1) = 9 ; atomic_eec(41,2) = 18; atomic_eec(41,3) = 14; ! Nb
      atomic_eec(42,1) = 9 ; atomic_eec(42,2) = 18; atomic_eec(42,3) = 15; ! Mo
      atomic_eec(43,1) = 10; atomic_eec(43,2) = 18; atomic_eec(43,3) = 15; ! Tc
      atomic_eec(44,1) = 9 ; atomic_eec(44,2) = 18; atomic_eec(44,3) = 17; ! Ru
      atomic_eec(45,1) = 9 ; atomic_eec(45,2) = 18; atomic_eec(45,3) = 18; ! Rh
      atomic_eec(46,1) = 8 ; atomic_eec(46,2) = 18; atomic_eec(46,3) = 20; ! Pd
      atomic_eec(47,1) = 9 ; atomic_eec(47,2) = 18; atomic_eec(47,3) = 20; ! Ag
      atomic_eec(48,1) = 10; atomic_eec(48,2) = 18; atomic_eec(48,3) = 20; ! Cd
      atomic_eec(49,1) = 10; atomic_eec(49,2) = 19; atomic_eec(49,3) = 20; ! In
      atomic_eec(50,1) = 10; atomic_eec(50,2) = 20; atomic_eec(50,3) = 20; ! Sn
      atomic_eec(51,1) = 10; atomic_eec(51,2) = 21; atomic_eec(51,3) = 20; ! Sb
      atomic_eec(52,1) = 10; atomic_eec(52,2) = 22; atomic_eec(52,3) = 20; ! Te
      atomic_eec(53,1) = 10; atomic_eec(53,2) = 23; atomic_eec(53,3) = 20; ! I
      atomic_eec(54,1) = 10; atomic_eec(54,2) = 24; atomic_eec(54,3) = 20; ! Xe
   end subroutine initialise_eec
end module initial_guess_data

!##############################################################################!
module initial_guess_subs

contains

! This subroutine is the interface between SCF and the initial guess choice.
subroutine get_initial_guess(M, MM, NCO, NCOb, Xmat, Hvec, Smat, Rhovec, &
                             rhoalpha, rhobeta, openshell, natom, Iz, nshell, &
                             Nuc)
   use initial_guess_data, only: initial_guess

   implicit none
   LIODBLE, intent(in) :: Xmat(:,:), Hvec(:), Smat(:,:)
   logical         , intent(in) :: openshell
   integer         , intent(in) :: M, MM, NCO, NCOb, natom, Iz(natom), Nuc(M), &
                                   nshell(0:2)

   LIODBLE, intent(inout) :: Rhovec(:), rhoalpha(:), rhobeta(:)
   LIODBLE :: ocupF

   call g2g_timer_start('initial guess')
   call g2g_timer_sum_start('initial guess')

   select case (initial_guess)
   case (0)
      ! 1e core-Hamiltonian guess (default). Empirically the best of the cheap
      ! guesses for LIO's main-group QM/MM cases: GWH (case 2) was measured to
      ! regress agua (14->16) and fosfato (24->27), helping only some
      ! point-charge cases, so it is not the default. See
      ! research/convergence/initial_guess_gwh_2026_06_14.md.
      if (.not. openshell) then
         ocupF = 2.0D0
         call initial_guess_1e(M, MM, NCO, ocupF, Hvec, Xmat, Rhovec )
      else
         ocupF = 1.0D0
         call initial_guess_1e(M, MM, NCO , ocupF, Hvec, Xmat, rhoalpha)
         call initial_guess_1e(M, MM, NCOb, ocupF, Hvec, Xmat, rhobeta)
         Rhovec   = rhoalpha + rhobeta
      end if
   case (1)
      call initial_guess_aufbau(M, MM, Rhovec, rhoalpha, rhobeta, natom, NCO,&
                                NCOb, Iz, nshell, Nuc, openshell)
   case (2)
      ! Generalized Wolfsberg-Helmholtz (GWH) guess (selectable, not default).
      ! Builds an effective Fock from the core-Hamiltonian diagonal and the
      ! overlap off-diagonals. Guess-only: the converged result is unchanged,
      ! only the iteration count moves.
      if (.not. openshell) then
         ocupF = 2.0D0
         call initial_guess_gwh(M, MM, NCO, ocupF, Hvec, Smat, Xmat, Rhovec )
      else
         ocupF = 1.0D0
         call initial_guess_gwh(M, MM, NCO , ocupF, Hvec, Smat, Xmat, rhoalpha)
         call initial_guess_gwh(M, MM, NCOb, ocupF, Hvec, Smat, Xmat, rhobeta)
         Rhovec   = rhoalpha + rhobeta
      end if
   case default
      write(*,*) "ERROR - Initial guess: Wrong value for input initial_guess."
   end select

   call g2g_timer_stop('initial guess')
   call g2g_timer_sum_stop('initial guess')

   return
end subroutine get_initial_guess


! Perfoms the initial guess using a modified aufbau principle.
subroutine initial_guess_aufbau(M, MM, rhototal, rhoalpha, rhobeta, natom, NCO, &
                                NCOb, Iz, nshell, Nuc, openshell)
   use initial_guess_data, only: atomic_eec, initialise_eec
   implicit none
   integer         , intent(in)  :: M, MM, natom, NCO, NCOb, nshell(0:2), &
                                    Iz(natom),Nuc(M)
   logical         , intent(in)  :: openshell
   LIODBLE, intent(out) :: rhototal(MM), rhoalpha(MM), rhobeta(MM)

   LIODBLE, allocatable :: start_dens(:,:), start_dens_alpha(:,:), &
                                    start_dens_beta(:,:)
   integer         , allocatable :: n_elecs(:,:)
   integer                       :: icount, total_iz, atom_id

   allocate(start_dens(M,M), start_dens_alpha(M,M), start_dens_beta(M,M), &
            n_elecs(natom,3))

   call initialise_eec()
   start_dens(:,:) = 0.0D0
   n_elecs = 0

   total_iz = 0
   do icount = 1, natom
      total_iz = total_iz + Iz(icount)
      n_elecs(icount, :) = atomic_eec(Iz(icount), :)
   enddo

   do icount = 1, nshell(0)
      atom_id = Nuc(icount)
      if (n_elecs(atom_id,1) >= 2) then
         start_dens(icount,icount) = 2.0D0
         n_elecs(atom_id,1) = n_elecs(atom_id,1) - 2
      else if (n_elecs(atom_id,1) > 0) then
         start_dens(icount,icount) = 1.0D0
         n_elecs(atom_id,1) = 0
      endif
   enddo

   do icount = nshell(0)+1, nshell(1)+nshell(0), 3
      atom_id = Nuc(icount)
      if (n_elecs(atom_id,2) >= 6) then
         start_dens(icount  ,icount  ) = 2.0D0
         start_dens(icount+1,icount+1) = 2.0D0
         start_dens(icount+2,icount+2) = 2.0D0
         n_elecs(atom_id,2) = n_elecs(atom_id,2) - 6
      else if (n_elecs(atom_id,2) > 0) then
         start_dens(icount, icount)     = dble(n_elecs(atom_id,2)) / 3.0D0
         start_dens(icount+1, icount+1) = dble(n_elecs(atom_id,2)) / 3.0D0
         start_dens(icount+2, icount+2) = dble(n_elecs(atom_id,2)) / 3.0D0
         n_elecs(atom_id,2) = 0
      endif
   enddo

   do icount = nshell(1)+nshell(0)+1, nshell(2)+nshell(0)+nshell(1), 6
      atom_id = Nuc(icount)
      if (n_elecs(atom_id,3) >= 10) then
         start_dens(icount, icount)     = 5.0D0 / 3.0D0
         start_dens(icount+1, icount+1) = 5.0D0 / 3.0D0
         start_dens(icount+2, icount+2) = 5.0D0 / 3.0D0
         start_dens(icount+3, icount+3) = 5.0D0 / 3.0D0
         start_dens(icount+4, icount+4) = 5.0D0 / 3.0D0
         start_dens(icount+5, icount+5) = 5.0D0 / 3.0D0
         n_elecs(atom_id,3) = n_elecs(atom_id,3) - 10
      else if (n_elecs(atom_id,3) > 0) then
         start_dens(icount, icount)     = dble(n_elecs(atom_id,3)) / 6.0D0
         start_dens(icount+1, icount+1) = dble(n_elecs(atom_id,3)) / 6.0D0
         start_dens(icount+2, icount+2) = dble(n_elecs(atom_id,3)) / 6.0D0
         start_dens(icount+3, icount+3) = dble(n_elecs(atom_id,3)) / 6.0D0
         start_dens(icount+4, icount+4) = dble(n_elecs(atom_id,3)) / 6.0D0
         start_dens(icount+5, icount+5) = dble(n_elecs(atom_id,3)) / 6.0D0
         n_elecs(atom_id,3) = 0
      endif
   enddo

   if (openshell) then
      start_dens_alpha(:,:) = start_dens(:,:) * dble(NCO ) / dble(total_iz)
      start_dens_beta(:,:)  = start_dens(:,:) * dble(NCOb) / dble(total_iz)
      start_dens(:,:)       = start_dens_alpha(:,:) + start_dens_beta(:,:)
      call sprepack('L', M, rhoalpha, start_dens_alpha)
      call sprepack('L', M, rhobeta , start_dens_beta)
   else
      start_dens(:,:) = start_dens(:,:) * dble(NCO*2) / dble(total_iz)
   endif

   call sprepack('L', M, rhototal, start_dens)
   deallocate(start_dens, start_dens_alpha, start_dens_beta, n_elecs)
   return
end subroutine initial_guess_aufbau

! This subroutine performs the 1e-integral guess. It takes the Hmat as a      !
! vector, transforms it into a matrix, diagonalizes it, and builds the        !
! density from the resulting orbitals.                                        !
subroutine initial_guess_1e(Nmat, Nvec, NCO, ocupF, hmat_vec, Xmat, densat_vec)
   use SCF_aux     , only: messup_densmat

   implicit none
   integer         , intent(in)    :: Nmat, Nvec, NCO
   LIODBLE, intent(in)    :: ocupF, Xmat(Nmat,Nmat), hmat_vec(Nvec)
   LIODBLE, intent(inout) :: densat_vec(Nvec)

   LIODBLE, allocatable   :: morb_energy(:), morb_coefon(:,:),   &
                                      morb_coefat(:,:),                   &
                                      hmat(:,:), dens_mao(:,:), tmp(:,:)
   LIODBLE, allocatable   :: WORK(:)
   integer, allocatable            :: IWORK(:)
   integer                         :: LWORK, LIWORK, info

   allocate( morb_coefon(Nmat, Nmat), morb_energy(Nmat), dens_mao(Nmat, Nmat) )
   allocate( morb_coefat(Nmat, Nmat), hmat(Nmat,Nmat), tmp(Nmat,Nmat) )

   call spunpack('L', Nmat, hmat_vec, hmat )

   ! Transform the 1e Hamiltonian to the orthonormal basis: F' = X^T H X.
   call DGEMM('N','N', Nmat, Nmat, Nmat, 1.0D0, hmat, Nmat, Xmat, Nmat, &
              0.0D0, tmp, Nmat)
   call DGEMM('T','N', Nmat, Nmat, Nmat, 1.0D0, Xmat, Nmat, tmp, Nmat, &
              0.0D0, morb_coefon, Nmat)
   morb_energy(:) = 0.0d0

   ! Divide-and-conquer diagonalization (dsyevd) instead of the QR-based
   ! dsyev: same eigenvectors, ~2-3x faster at these matrix sizes.
   allocate( WORK(1), IWORK(1) )
   call dsyevd('V', 'L', Nmat, morb_coefon, Nmat, morb_energy, WORK, -1, &
               IWORK, -1, info)
   LWORK  = INT( WORK(1) )
   LIWORK = IWORK(1)
   deallocate( WORK, IWORK )
   allocate( WORK(LWORK), IWORK(LIWORK) )
   call dsyevd('V', 'L', Nmat, morb_coefon, Nmat, morb_energy, WORK, LWORK, &
               IWORK, LIWORK, info)

   ! Back-transform coefficients to the AO basis and build the density from
   ! the occupied block: P = ocupF * C_occ C_occ^T.
   call DGEMM('N','N', Nmat, Nmat, Nmat, 1.0D0, Xmat, Nmat, morb_coefon, &
              Nmat, 0.0D0, morb_coefat, Nmat)
   call DGEMM('N','T', Nmat, Nmat, NCO, ocupF, morb_coefat, Nmat, &
              morb_coefat, Nmat, 0.0D0, dens_mao, Nmat)
   call messup_densmat( dens_mao )
   call sprepack( 'L', Nmat, densat_vec, dens_mao)

   deallocate( morb_coefon, morb_energy, dens_mao, morb_coefat, hmat, tmp, &
               WORK, IWORK )
   return
end subroutine initial_guess_1e

! This subroutine performs the Generalized Wolfsberg-Helmholtz (GWH) guess.   !
! It builds an effective Fock matrix from the core-Hamiltonian diagonal and   !
! the overlap matrix,                                                         !
!     F_ii = H_ii ,  F_ij = 0.5 * K * S_ij * (H_ii + H_jj)  (i /= j) ,        !
! with the standard empirical K = 1.75. The resulting Fock is diagonalized in !
! the orthonormal basis (F' = X^T F X) exactly as in the 1e guess, and the    !
! density is built from the occupied block. This is a starting guess only:    !
! the converged SCF result is unaffected, only the iteration count changes.   !
subroutine initial_guess_gwh(Nmat, Nvec, NCO, ocupF, hmat_vec, smat, Xmat, &
                             densat_vec)
   use SCF_aux     , only: messup_densmat

   implicit none
   integer         , intent(in)    :: Nmat, Nvec, NCO
   LIODBLE, intent(in)    :: ocupF, Xmat(Nmat,Nmat), hmat_vec(Nvec), &
                                      smat(Nmat,Nmat)
   LIODBLE, intent(inout) :: densat_vec(Nvec)

   LIODBLE, allocatable   :: morb_energy(:), morb_coefon(:,:),   &
                                      morb_coefat(:,:), hmat(:,:),        &
                                      fmat(:,:), dens_mao(:,:), tmp(:,:)
   LIODBLE, allocatable   :: WORK(:)
   integer, allocatable            :: IWORK(:)
   integer                         :: LWORK, LIWORK, info, ii, jj
   LIODBLE, parameter     :: gwh_k = 1.75D0

   allocate( morb_coefon(Nmat, Nmat), morb_energy(Nmat), dens_mao(Nmat, Nmat) )
   allocate( morb_coefat(Nmat, Nmat), hmat(Nmat,Nmat), fmat(Nmat,Nmat), &
             tmp(Nmat,Nmat) )

   call spunpack('L', Nmat, hmat_vec, hmat )

   ! Build the GWH effective Fock from the core-Hamiltonian diagonal and the
   ! overlap. Diagonal is the bare core element; off-diagonals are interpolated
   ! between the two on-site energies and scaled by the overlap and K.
   do jj = 1, Nmat
      fmat(jj,jj) = hmat(jj,jj)
      do ii = jj+1, Nmat
         fmat(ii,jj) = 0.5D0 * gwh_k * smat(ii,jj) * (hmat(ii,ii) + hmat(jj,jj))
         fmat(jj,ii) = fmat(ii,jj)
      enddo
   enddo

   ! Transform the GWH Fock to the orthonormal basis: F' = X^T F X.
   call DGEMM('N','N', Nmat, Nmat, Nmat, 1.0D0, fmat, Nmat, Xmat, Nmat, &
              0.0D0, tmp, Nmat)
   call DGEMM('T','N', Nmat, Nmat, Nmat, 1.0D0, Xmat, Nmat, tmp, Nmat, &
              0.0D0, morb_coefon, Nmat)
   morb_energy(:) = 0.0d0

   ! Divide-and-conquer diagonalization (dsyevd).
   allocate( WORK(1), IWORK(1) )
   call dsyevd('V', 'L', Nmat, morb_coefon, Nmat, morb_energy, WORK, -1, &
               IWORK, -1, info)
   LWORK  = INT( WORK(1) )
   LIWORK = IWORK(1)
   deallocate( WORK, IWORK )
   allocate( WORK(LWORK), IWORK(LIWORK) )
   call dsyevd('V', 'L', Nmat, morb_coefon, Nmat, morb_energy, WORK, LWORK, &
               IWORK, LIWORK, info)

   ! Back-transform coefficients to the AO basis and build the density from
   ! the occupied block: P = ocupF * C_occ C_occ^T.
   call DGEMM('N','N', Nmat, Nmat, Nmat, 1.0D0, Xmat, Nmat, morb_coefon, &
              Nmat, 0.0D0, morb_coefat, Nmat)
   call DGEMM('N','T', Nmat, Nmat, NCO, ocupF, morb_coefat, Nmat, &
              morb_coefat, Nmat, 0.0D0, dens_mao, Nmat)
   call messup_densmat( dens_mao )
   call sprepack( 'L', Nmat, densat_vec, dens_mao)

   deallocate( morb_coefon, morb_energy, dens_mao, morb_coefat, hmat, fmat, &
               tmp, WORK, IWORK )
   return
end subroutine initial_guess_gwh

end module initial_guess_subs
