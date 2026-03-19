!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
       function commutator_dd(MA,MB)
     > result(MC)
       implicit none
       real*8,intent(in)      :: MA(:,:)
       real*8,intent(in)      :: MB(:,:)
       real*8,allocatable     :: MC(:,:)
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn))
       ! MC = MA*MB
       call DGEMM('N','N',nn,nn,nn,1.0D0,MA,nn,MB,nn,0.0D0,MC,nn)
       ! MC = MC - MB*MA = MA*MB - MB*MA
       call DGEMM('N','N',nn,nn,nn,-1.0D0,MB,nn,MA,nn,1.0D0,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_zd(MA,MB)
     > result(MC)
       implicit none
       complex*16,intent(in)  :: MA(:,:)
       real*8,intent(in)      :: MB(:,:)
       complex*16,allocatable :: MC(:,:)
       complex*16,allocatable :: MB_c(:,:)
       complex*16             :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn), MB_c(nn,nn))
       one = dcmplx(1.0D0, 0.0D0)
       zero = dcmplx(0.0D0, 0.0D0)
       neg_one = dcmplx(-1.0D0, 0.0D0)
       MB_c = dcmplx(MB, 0.0D0)
       ! MC = MA*MB_c
       call ZGEMM('N','N',nn,nn,nn,one,MA,nn,MB_c,nn,zero,MC,nn)
       ! MC = MC - MB_c*MA
       call ZGEMM('N','N',nn,nn,nn,neg_one,MB_c,nn,MA,nn,one,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_dz(MA,MB)
     > result(MC)
       implicit none
       real*8,intent(in)      :: MA(:,:)
       complex*16,intent(in)  :: MB(:,:)
       complex*16,allocatable :: MC(:,:)
       complex*16,allocatable :: MA_c(:,:)
       complex*16             :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn), MA_c(nn,nn))
       one = dcmplx(1.0D0, 0.0D0)
       zero = dcmplx(0.0D0, 0.0D0)
       neg_one = dcmplx(-1.0D0, 0.0D0)
       MA_c = dcmplx(MA, 0.0D0)
       ! MC = MA_c*MB
       call ZGEMM('N','N',nn,nn,nn,one,MA_c,nn,MB,nn,zero,MC,nn)
       ! MC = MC - MB*MA_c
       call ZGEMM('N','N',nn,nn,nn,neg_one,MB,nn,MA_c,nn,one,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_zz(MA,MB)
     > result(MC)
       implicit none
       complex*16,intent(in)  :: MA(:,:)
       complex*16,intent(in)  :: MB(:,:)
       complex*16,allocatable :: MC(:,:)
       complex*16             :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn))
       one = dcmplx(1.0D0, 0.0D0)
       zero = dcmplx(0.0D0, 0.0D0)
       neg_one = dcmplx(-1.0D0, 0.0D0)
       ! MC = MA*MB
       call ZGEMM('N','N',nn,nn,nn,one,MA,nn,MB,nn,zero,MC,nn)
       ! MC = MC - MB*MA
       call ZGEMM('N','N',nn,nn,nn,neg_one,MB,nn,MA,nn,one,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_cd(MA,MB)
     > result(MC)
       implicit none
       complex*8,intent(in)  :: MA(:,:)
       real*8,intent(in)      :: MB(:,:)
       complex*8,allocatable :: MC(:,:)
       complex*8,allocatable :: MB_c(:,:)
       complex*8              :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn), MB_c(nn,nn))
       one = cmplx(1.0, 0.0)
       zero = cmplx(0.0, 0.0)
       neg_one = cmplx(-1.0, 0.0)
       MB_c = cmplx(real(MB), 0.0)
       ! MC = MA*MB_c
       call CGEMM('N','N',nn,nn,nn,one,MA,nn,MB_c,nn,zero,MC,nn)
       ! MC = MC - MB_c*MA
       call CGEMM('N','N',nn,nn,nn,neg_one,MB_c,nn,MA,nn,one,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_dc(MA,MB)
     > result(MC)
       implicit none
       real*8,intent(in)      :: MA(:,:)
       complex*8,intent(in)  :: MB(:,:)
       complex*8,allocatable :: MC(:,:)
       complex*8,allocatable :: MA_c(:,:)
       complex*8              :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn), MA_c(nn,nn))
       one = cmplx(1.0, 0.0)
       zero = cmplx(0.0, 0.0)
       neg_one = cmplx(-1.0, 0.0)
       MA_c = cmplx(real(MA), 0.0)
       ! MC = MA_c*MB
       call CGEMM('N','N',nn,nn,nn,one,MA_c,nn,MB,nn,zero,MC,nn)
       ! MC = MC - MB*MA_c
       call CGEMM('N','N',nn,nn,nn,neg_one,MB,nn,MA_c,nn,one,MC,nn)
       return;end function
!
!
!--------------------------------------------------------------------!
       function commutator_cc(MA,MB)
     > result(MC)
       implicit none
       complex*8,intent(in)  :: MA(:,:)
       complex*8,intent(in)  :: MB(:,:)
       complex*8,allocatable :: MC(:,:)
       complex*8              :: one, zero, neg_one
       integer                :: nn

       nn=size(MA,1)
       allocate(MC(nn,nn))
       one = cmplx(1.0, 0.0)
       zero = cmplx(0.0, 0.0)
       neg_one = cmplx(-1.0, 0.0)
       ! MC = MA*MB
       call CGEMM('N','N',nn,nn,nn,one,MA,nn,MB,nn,zero,MC,nn)
       ! MC = MC - MB*MA
       call CGEMM('N','N',nn,nn,nn,neg_one,MB,nn,MA,nn,one,MC,nn)
       return;end function
!
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
