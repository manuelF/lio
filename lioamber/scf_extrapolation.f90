#include "datatypes/datatypes.fh"
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! SCF_EXTRAPOLATION                                                           !
!                                                                            !
! Cross-step density extrapolation for the SCF starting guess (Kolafa 2004,  !
! "Always Stable Predictor-Corrector", ASPC; here the predictor used as an   !
! SCF guess). LIO is primarily a QM/MM MD code: along a smooth BOMD /        !
! geometry-optimization trajectory the converged density of step t is a      !
! smooth function of the nuclear coordinates, so it can be extrapolated from !
! the last few converged densities far more accurately than the plain VCINP  !
! "reuse the single last density" guess. A better guess means fewer SCF      !
! iterations every step, which multiplies over a whole trajectory.           !
!                                                                            !
! Predictor (k previous converged densities, i=1 newest):                    !
!     P_pred = sum_{i=1..k} B_i * P(t+1-i),   B_i = (-1)^(i+1) * C(k,i).      !
! These time-reversible coefficients satisfy sum_i B_i = 1, so the predicted !
! density conserves the electron count tr(P S) = N exactly (each stored P    !
! has the same trace and the coefficients sum to one). The prediction need   !
! not be exactly idempotent: it is only a *guess*; the very first SCF        !
! half-iteration rebuilds an idempotent density by diagonalization, so no    !
! McWeeny purification is required for the guess to be valid (purification   !
! only matters for XL-BOMD, where the propagated density enters the forces). !
!                                                                            !
! Safety:                                                                    !
!   * No-op on single points and on the first trajectory step (no history).  !
!   * The order ramps up as history fills (order = min(stored, max_order)),  !
!     so early steps stay conservative.                                      !
!   * Guess-only: it changes *where the SCF starts*, never the converged     !
!     fixed point, so it cannot change the physical result -- only the       !
!     iteration count moves. A poor extrapolation can only cost iterations,  !
!     which the conservative default order and the self-correcting SCF bound.!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
module scf_extrapolation
   implicit none
   private
   public :: scf_extrap_predict, scf_extrap_store, scf_extrap_reset

   ! Maximum predictor order (ring-buffer depth). Order 4 (coefficients
   ! 4,-6,4,-1) is a robust ASPC default: high enough to capture the smooth
   ! trajectory curvature, low enough to stay stable against trajectory noise.
   integer, parameter :: max_order = 4

   ! Ships unconditionally on (no-toggle rule). Compile-time switch only: flip to
   ! .false. and rebuild to A/B-benchmark the extrapolation; the dead branch is
   ! then removed by the optimizer.
   logical, parameter :: enabled = .true.
   integer      :: stored      = 0        ! number of valid history entries
   integer      :: head        = 0        ! ring slot of the newest entry (1..max_order)
   integer      :: vlen        = 0        ! length of the stored vectors (MM)
   logical      :: hist_open   = .false.  ! spin layout of the stored history

   ! Geometry fingerprint of the newest stored density. Extrapolation only fires
   ! when the nuclei actually moved between steps (MD / geometry optimization);
   ! at fixed geometry (transport / DLVN density driving, TD predictor, repeated
   ! property solves) the density is not a smooth function of the coordinates, so
   ! cross-step extrapolation is meaningless and is skipped.
   LIODBLE :: geo_fp_last = 0.0D0
   logical      :: have_geo_fp = .false.

   ! Ring buffers (vlen, max_order). For closed shell only hist_t is used; for
   ! open shell hist_a / hist_b hold the alpha / beta spin densities.
   LIODBLE, allocatable :: hist_t(:,:)
   LIODBLE, allocatable :: hist_a(:,:)
   LIODBLE, allocatable :: hist_b(:,:)

contains

! Cheap fingerprint of a nuclear configuration, used only to detect whether the
! geometry changed between SCF solves. A coordinate-index-weighted sum makes
! accidental collisions between two distinct trajectory geometries vanishingly
! unlikely (and a collision would only mean one skipped extrapolation).
pure function geo_fingerprint(r, ntatom) result(fp)
   implicit none
   integer        , intent(in) :: ntatom
   LIODBLE, intent(in) :: r(ntatom,3)
   LIODBLE :: fp
   integer        :: i, k
   fp = 0.0D0
   do k = 1, 3
      do i = 1, ntatom
         fp = fp + r(i,k) * dble(3*i + k)
      enddo
   enddo
end function geo_fingerprint

! Maps a 1-based logical offset (1 = newest) to its ring slot.
pure integer function ring_slot(offset) result(slot)
   implicit none
   integer, intent(in) :: offset
   slot = mod(head - (offset - 1) - 1 + max_order, max_order) + 1
end function ring_slot

! Drops all stored history (called when the system / spin layout changes).
subroutine scf_extrap_reset()
   implicit none
   stored = 0
   head   = 0
end subroutine scf_extrap_reset

! Builds the predicted guess density from the converged-density history and
! overwrites Pmat (closed) or rhoalpha/rhobeta + Pmat (open). No-op until at
! least two converged densities are available.
subroutine scf_extrap_predict(MM, Pmat, rhoalpha, rhobeta, openshell, r, ntatom)
   implicit none
   integer        , intent(in)    :: MM, ntatom
   logical        , intent(in)    :: openshell
   ! rhoalpha / rhobeta are size 1 for closed shell (see drive.f90), so they
   ! are assumed-shape and only ever touched on the open-shell branch.
   LIODBLE, intent(inout) :: Pmat(MM), rhoalpha(:), rhobeta(:)
   LIODBLE, intent(in)    :: r(ntatom,3)

   integer         :: order, ii, slot
   LIODBLE :: coef(max_order), fp

   if (.not. enabled)              return
   if (vlen /= MM)                 return   ! no history for this system yet
   if (hist_open .neqv. openshell) return
   if (stored < 2)                 return   ! <2 densities: nothing to extrapolate
                                            ! (also covers single points / step 1)

   ! Only extrapolate if the geometry moved since the last converged density;
   ! at fixed geometry the density history is not a smooth trajectory.
   fp = geo_fingerprint(r, ntatom)
   if (.not. have_geo_fp)          return
   if (fp == geo_fp_last)          return

   order = min(stored, max_order)
   call aspc_coeffs(order, coef)

   if (openshell) then
      rhoalpha = 0.0D0
      rhobeta  = 0.0D0
      do ii = 1, order
         slot     = ring_slot(ii)
         rhoalpha = rhoalpha + coef(ii) * hist_a(:, slot)
         rhobeta  = rhobeta  + coef(ii) * hist_b(:, slot)
      enddo
      Pmat = rhoalpha + rhobeta
   else
      Pmat = 0.0D0
      do ii = 1, order
         slot = ring_slot(ii)
         Pmat = Pmat + coef(ii) * hist_t(:, slot)
      enddo
   endif
end subroutine scf_extrap_predict

! Pushes the just-converged density onto the history ring buffer. Called once
! per SCF solve, after convergence.
subroutine scf_extrap_store(MM, Pmat, rhoalpha, rhobeta, openshell, r, ntatom)
   implicit none
   integer        , intent(in) :: MM, ntatom
   logical        , intent(in) :: openshell
   ! rhoalpha / rhobeta are size 1 for closed shell (see drive.f90), so they
   ! are assumed-shape and only ever touched on the open-shell branch.
   LIODBLE, intent(in) :: Pmat(MM), rhoalpha(:), rhobeta(:)
   LIODBLE, intent(in) :: r(ntatom,3)

   if (.not. enabled) return

   ! (Re)allocate / reset when the system size or spin layout changes. Note the
   ! parentheses around the .neqv. term: in Fortran .neqv. binds looser than
   ! .or., so it must be grouped explicitly.
   if (vlen /= MM .or. (hist_open .neqv. openshell) .or. .not. allocated(hist_t)) then
      if (allocated(hist_t)) deallocate(hist_t)
      if (allocated(hist_a)) deallocate(hist_a)
      if (allocated(hist_b)) deallocate(hist_b)
      allocate(hist_t(MM, max_order))
      if (openshell) allocate(hist_a(MM, max_order), hist_b(MM, max_order))
      vlen      = MM
      hist_open = openshell
      call scf_extrap_reset()
   endif

   head = mod(head, max_order) + 1
   if (openshell) then
      hist_a(:, head) = rhoalpha
      hist_b(:, head) = rhobeta
      hist_t(:, head) = Pmat
   else
      hist_t(:, head) = Pmat
   endif
   stored = min(stored + 1, max_order)

   ! Record the geometry of this converged density for the move-detection gate.
   geo_fp_last = geo_fingerprint(r, ntatom)
   have_geo_fp = .true.
end subroutine scf_extrap_store

! Time-reversible ASPC predictor coefficients B_i = (-1)^(i+1) * C(order,i),
! i = 1..order (i = 1 is the newest density). They sum to one.
pure subroutine aspc_coeffs(order, coef)
   implicit none
   integer        , intent(in)  :: order
   LIODBLE, intent(out) :: coef(:)
   integer :: ii
   LIODBLE :: binom

   binom = 1.0D0
   do ii = 1, order
      ! C(order,ii) = C(order,ii-1) * (order-ii+1)/ii
      binom = binom * dble(order - ii + 1) / dble(ii)
      if (mod(ii, 2) == 1) then
         coef(ii) =  binom
      else
         coef(ii) = -binom
      endif
   enddo
end subroutine aspc_coeffs

end module scf_extrapolation
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
