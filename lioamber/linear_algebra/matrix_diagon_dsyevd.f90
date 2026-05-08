!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
subroutine matrix_diagon_dsyevd( matrix_in, eigen_vecs, eigen_vals , info )
!
! DSYEVD with persistent workspace: the work/iwork arrays are allocated once
! (on first call or when M changes) and reused across iterations.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
  implicit none
  LIODBLE,  intent(in)            :: matrix_in(:,:)
  LIODBLE,  intent(out)           :: eigen_vecs(:,:)
  LIODBLE,  intent(out)           :: eigen_vals(:)
  integer, intent(out), optional :: info

  ! Persistent workspace — survives across calls (SAVE attribute).
  integer,          save :: cached_M = 0
  integer,          save :: cached_lwork = 0
  integer,          save :: cached_liwork = 0
  LIODBLE, allocatable, save :: work(:)
  integer,allocatable, save :: iwork(:)

  integer :: M
  integer :: local_stat
!
! Initial checks
!------------------------------------------------------------------------------!
  local_stat=0
  if ( present(info) ) info=0

  M=size(matrix_in,1)
  if ( M /= size(matrix_in,2) )  local_stat=1
  if ( M /= size(eigen_vecs,1) ) local_stat=2
  if ( M /= size(eigen_vecs,2) ) local_stat=3
  if ( M /= size(eigen_vals) )   local_stat=4

  if ( local_stat /= 0 ) then
    if ( present(info) ) then
      info = 1
      return
    else
      print*,'matrix_diagon_dsyevd : incompatible size between arguments'
      print*,'local info: ', local_stat
      stop
    end if
  end if
!
! Allocate workspace on first call or if M changed
!------------------------------------------------------------------------------!
  if ( M /= cached_M ) then
    eigen_vecs = matrix_in

    ! Workspace query
    if ( allocated(work) )  deallocate(work)
    if ( allocated(iwork) ) deallocate(iwork)
    allocate( work(1), iwork(1), stat=local_stat )
    if ( local_stat /= 0 ) then
      if ( present(info) ) then; info = 1; return
      else; print*,'matrix_diagon_dsyevd : allocation error'; stop
      end if
    end if

# ifdef magma
    call magmaf_dsyevd( 'V', 'L', M, eigen_vecs, M, eigen_vals, &
                      & work, -1, iwork, -1, local_stat )
# else
    call        dsyevd( 'V', 'L', M, eigen_vecs, M, eigen_vals, &
                      & work, -1, iwork, -1, local_stat )
# endif

    cached_lwork  = int(work(1))
    cached_liwork = iwork(1)
    deallocate( work, iwork )
    allocate( work(cached_lwork), iwork(cached_liwork), stat=local_stat )
    if ( local_stat /= 0 ) then
      if ( present(info) ) then; info = 2; return
      else; print*,'matrix_diagon_dsyevd : workspace allocation error'; stop
      end if
    end if

    cached_M = M
  end if
!
! Do actual diagonalization
!------------------------------------------------------------------------------!
  eigen_vecs = matrix_in

# ifdef magma
  call magmaf_dsyevd( 'V', 'L', M, eigen_vecs, M, eigen_vals, &
                    & work, cached_lwork, iwork, cached_liwork, local_stat )
# else
  call        dsyevd( 'V', 'L', M, eigen_vecs, M, eigen_vals, &
                    & work, cached_lwork, iwork, cached_liwork, local_stat )
# endif

  if ( local_stat /= 0 ) then
    if ( present(info) ) then
      info = 3
      return
    else
      print*,'matrix_diagon_dsyevd : critical error while diagonalizing'
      print*,'local info: ', local_stat
      stop
    end if
  end if

end subroutine matrix_diagon_dsyevd
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
