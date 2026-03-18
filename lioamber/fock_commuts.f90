!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
!%% FOCK_COMMUTS.F90 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
! This file contains calc_fock_commuts, which calculates F' and [F',P'] for    !
! DIIS matrices fockm and FP_PFm.                                              !
! Input: F (fock), P (rho), X, Y                                               !
! F' = X^T * F * X   |   P' = Y^T * P * Y   | X = (Y^-1)^T                     !
! => [F',P'] = X^T * F * P * Y - Y^T * P * F * X = A - A^T                     !
! Where A = X^T * F * P * Y                                                    !
! Output: A (scratch), A^T (scratch1), F' (fock)                               !
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
subroutine calc_fock_commuts(fock, rho, X, Y, scratch, scratch1, M)
    implicit none
    integer, intent(in)    :: M
    REAL*8,  intent(in)    :: rho(M,M),X(M,M),Y(M,M)
    REAL*8,  intent(inout) :: fock(M,M)
    REAL*8,  intent(out)   :: scratch(M,M),scratch1(M,M)
    integer :: i, j

    ! Step 1: scratch = X^T * F  (reuse scratch as temporary)
    call DGEMM('T','N',M,M,M,1.0D0,X,M,fock,M,0.0D0,scratch,M)

    ! Step 2: fock = scratch * X = X^T * F * X  (F' = Fock in ON basis)
    call DGEMM('N','N',M,M,M,1.0D0,scratch,M,X,M,0.0D0,fock,M)

    ! Step 3: scratch1 = scratch * P = (X^T * F) * P
    call DGEMM('N','N',M,M,M,1.0D0,scratch,M,rho,M,0.0D0,scratch1,M)

    ! Step 4: scratch = scratch1 * Y = X^T * F * P * Y = A
    call DGEMM('N','N',M,M,M,1.0D0,scratch1,M,Y,M,0.0D0,scratch,M)

    ! Step 5: scratch1 = A^T
    do j = 1, M
    do i = 1, M
       scratch1(i,j) = scratch(j,i)
    enddo
    enddo

end subroutine calc_fock_commuts
