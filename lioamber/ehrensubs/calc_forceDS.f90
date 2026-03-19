!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
subroutine calc_forceDS &
  ( Natoms, Nbasis, nucpos, nucvel, DensMao, FockMao, Sinv, Bmat, forceDS )
!------------------------------------------------------------------------------!
!
! DESCRIPTION
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
  implicit none
  integer,intent(in)    :: Natoms
  integer,intent(in)    :: Nbasis
  real*8,intent(in)     :: nucpos(3,Natoms)
  real*8,intent(in)     :: nucvel(3,Natoms)
  complex*16,intent(in) :: DensMao(Nbasis,Nbasis)
  real*8,intent(in)     :: FockMao(Nbasis,Nbasis)
  real*8,intent(in)     :: Sinv(Nbasis,Nbasis)
  real*8,intent(out)    :: Bmat(Nbasis,Nbasis)
  real*8,intent(out)    :: forceDS(3,Natoms)


  real*8,allocatable     :: Btrp(:,:)
  complex*16,allocatable :: InputMat(:,:),MatTrp(:,:),MatDir(:,:)
  complex*16,allocatable :: fterm1(:,:),fterm2(:,:),fterm3(:,:)
  complex*16,allocatable :: FockC(:,:), SinvC(:,:), BmatC(:,:), BtrpC(:,:)
  complex*16 :: zone, zzero
  integer :: N
!
!
!------------------------------------------------------------------------------!
  call g2g_timer_start('calc_forceDS')
  N = Nbasis
  zone = dcmplx(1.0d0, 0.0d0)
  zzero = dcmplx(0.0d0, 0.0d0)
  allocate(InputMat(N,N))
  allocate(MatTrp(N,N),MatDir(N,N))
  allocate(Btrp(N,N))
  allocate(FockC(N,N), SinvC(N,N))
  allocate(fterm1(3,Natoms),fterm2(3,Natoms),fterm3(3,Natoms))

  ! Convert real matrices to complex for ZGEMM
  FockC = dcmplx(FockMao, 0.0d0)
  SinvC = dcmplx(Sinv, 0.0d0)

  fterm1=dcmplx(0.0d0,0.0d0)
  fterm2=dcmplx(0.0d0,0.0d0)
  fterm3=dcmplx(0.0d0,0.0d0)

! NOTA: El orden de las multiplicaciones afecta levemente el
! resultado obtenido
  ! MatTrp = DensMao * FockMao * Sinv
  call ZGEMM('N','N',N,N,N,zone,DensMao,N,FockC,N,zzero,MatTrp,N)
  call ZGEMM('N','N',N,N,N,zone,MatTrp,N,SinvC,N,zzero,InputMat,N)
  MatTrp = InputMat
  ! MatDir = Sinv * FockMao * DensMao
  call ZGEMM('N','N',N,N,N,zone,FockC,N,DensMao,N,zzero,MatDir,N)
  call ZGEMM('N','N',N,N,N,zone,SinvC,N,MatDir,N,zzero,InputMat,N)
  MatDir = InputMat
  InputMat=transpose(MatTrp)+MatDir
  call calc_forceDS_dss(Natoms,Nbasis,nucpos,nucvel,InputMat,Bmat,fterm1)
  Btrp=transpose(Bmat)

  ! Convert Bmat/Btrp to complex
  allocate(BmatC(N,N), BtrpC(N,N))
  BmatC = dcmplx(Bmat, 0.0d0)
  BtrpC = dcmplx(Btrp, 0.0d0)

  ! MatTrp = DensMao * Btrp * Sinv * i
  call ZGEMM('N','N',N,N,N,zone,DensMao,N,BtrpC,N,zzero,MatTrp,N)
  call ZGEMM('N','N',N,N,N,zone,MatTrp,N,SinvC,N,zzero,InputMat,N)
  MatTrp = InputMat * dcmplx(0.0d0, 1.0d0)
  ! MatDir = Sinv * Bmat * DensMao * (-i)
  call ZGEMM('N','N',N,N,N,zone,SinvC,N,BmatC,N,zzero,MatDir,N)
  call ZGEMM('N','N',N,N,N,zone,MatDir,N,DensMao,N,zzero,InputMat,N)
  MatDir = InputMat * dcmplx(0.0d0, -1.0d0)
  InputMat=transpose(MatTrp)+MatDir
  call calc_forceDS_dss(Natoms,Nbasis,nucpos,nucvel,InputMat,Bmat,fterm2)
  deallocate(BmatC, BtrpC)


  MatTrp=DensMao*dcmplx(0.0d0,-1.0d0)
  MatDir=DensMao*dcmplx(0.0d0, 1.0d0)
  InputMat=transpose(MatTrp)+MatDir
  call calc_forceDS_dds(Natoms,Nbasis,nucpos,nucvel,InputMat,fterm3)


  forceDS=dble(real(fterm1+fterm2+fterm3))

  deallocate(InputMat,MatTrp,MatDir,Btrp,FockC,SinvC)
  deallocate(fterm1,fterm2,fterm3)
  call g2g_timer_stop('calc_forceDS')
end subroutine calc_forceDS
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!
