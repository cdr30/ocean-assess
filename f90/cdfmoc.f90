PROGRAM cdfmoc
  !!-------------------------------------------------------------------
  !!               ***  PROGRAM cdfmoc  ***
  !!
  !!  **  Purpose  :  Compute the Meridional Overturning Cell (MOC)
  !!                  PARTIAL STEPS
  !!  
  !!  **  Method   :  The MOC is computed from the V velocity field, integrated
  !!                  from the bottom to the surface, then zonally averaged with
  !!                  eventual masking for oceanic basins.
  !!                  The program looks for the file "new_maskglo.nc". If it does not exist, 
  !!                  only the calculation over all the domain is performed (this is adequate 
  !!                  for a basin configuration like NATL4).
  !!                  In new_maskglo.nc the masking corresponds to the global
  !!                  configuration. MOC for Global, Atlantic, Indo-Pacific, Indian,Pacific ocean
  !!                  Results are saved on moc.nc file with variables name respectively
  !!                  zomsfglo, zomsfatl, zomsfinp, zomsfind, zomsfpac
  !!
  !!
  !! history ;
  !!  Original :  J.M. Molines  (jul.  2005) 
  !!              A.M. Treguier (april 2006) adaptation to NATL4 case 
  !!-------------------------------------------------------------------
  !!  $Rev: 233 $
  !!  $Date: 2009-04-28 19:33:08 +0200 (Tue, 28 Apr 2009) $
  !!  $Id: cdfmoc.f90 233 2009-04-28 17:33:08Z molines $
  !!--------------------------------------------------------------
  !! * Modules used
  USE cdfio

  !! * Local variables
  IMPLICIT NONE
  INTEGER   :: jpbasins
  INTEGER   :: jbasin, jj, jk ,ji, jt              !: dummy loop index
  INTEGER   :: ierr                                !: working integer
  INTEGER   :: narg, iargc                         !: command line 
  INTEGER   :: npiglo,npjglo, npk, npt             !: size of the domain
  INTEGER   :: ncout, np
  INTEGER   :: numout=10
  INTEGER, DIMENSION(:), ALLOCATABLE ::  ipk, id_varout         !
  INTEGER, DIMENSION(2)              ::  iloc

  REAL(KIND=4), DIMENSION (:,:),     ALLOCATABLE ::  e1v, e3v, gphiv, zv !:  metrics, velocity
  REAL(KIND=4), DIMENSION (:,:),     ALLOCATABLE ::  dumlon              !: dummy longitude = 0.
  REAL(KIND=4), DIMENSION (:,:),     ALLOCATABLE ::  dumlat              !: latitude for i = north pole
  REAL(KIND=4), DIMENSION (:),       ALLOCATABLE ::  gdepw               !: deptw
  REAL(KIND=4), DIMENSION (:,:,:),   ALLOCATABLE ::  zmask               !:  jpbasins x npiglo x npjglo
  REAL(KIND=4), DIMENSION (1)                    ::  tim
  REAL(KIND=4)                                   :: spval                !: missing value

  REAL(KIND=8) ,DIMENSION(:,:,:) ,   ALLOCATABLE ::  zomsf                 !: jpbasins x npjglo x npk

  CHARACTER(LEN=300) :: cfilev , cfileoutnc='moc.nc' , maskv='mask.nc' 
  CHARACTER(LEN=300) :: coordhgr='mesh_hgr.nc',  coordzgr='mesh_zgr.nc',cbasinmask='new_maskglo.nc'
  TYPE(variable)    ,DIMENSION(:), ALLOCATABLE   :: typvar                   !: structure for attribute
  LOGICAL    :: llglo = .false.                            !: indicator for presence of new_maskglo.nc file 

  INTEGER    :: istatus

  ! constants

  !!  Read command line and output usage message if not compliant.
  narg= iargc()
  IF (( narg.ne.1 ) .AND. ( narg.ne.6 )) THEN
     PRINT *,' Usage : cdfmoc  V_file [mesh_hgr mesh_zgr new_maskglo mask output] '
     PRINT *,' Computes the MOC for oceanic basins as described in new_maskglo.nc'
     PRINT *,' PARTIAL CELLS VERSION'
     PRINT *,' Files mesh_hgr.nc, mesh_zgr.nc ,new_maskglo.nc ,mask.nc '
     PRINT *,'  must be in the current directory or specified as arguments 2-5'
     PRINT *,' Output on moc.nc (or in file specified by argument 6: '
     PRINT *,'      variables zomsfglo  : Global ocean '
     PRINT *,'      variables zomsfatl  : Atlantic Ocean '
     PRINT *,'      variables zomsfinp  : Indo Pacific '
     PRINT *,'      variables zomsfind  : Indian Ocean alone'
     PRINT *,'      variables zomsfpac  : Pacific Ocean alone'
     STOP
  ENDIF 

  CALL getarg (1, cfilev)
  npiglo= getdim (cfilev,'x',ldexact=.true.)
  npjglo= getdim (cfilev,'y',ldexact=.true.)
  npk   = getdim (cfilev,'depth')
  npt   = getdim (cfilev,'time_counter')

  IF ( narg == 6) THEN
    CALL getarg (2, coordhgr)
    CALL getarg (3, coordzgr)
    CALL getarg (4, cbasinmask)
    CALL getarg (5, maskv)
    CALL getarg (6, cfileoutnc)
  ENDIF
  
 !  Detects newmaskglo file 
  INQUIRE( FILE=cbasinmask, EXIST=llglo )
  IF (llglo) THEN
     jpbasins = 5
  ELSE
     jpbasins = 1
  ENDIF 

  ALLOCATE ( typvar(jpbasins), ipk(jpbasins), id_varout(jpbasins) )

 ! define new variables for output 
  typvar(1)%name= 'zomsfglo'
  typvar%units='Sverdrup'
  typvar%missing_value=99999.
  typvar%valid_min= -1000.
  typvar%valid_max= 1000.
  typvar%scale_factor= 1.
  typvar%add_offset= 0.
  typvar%savelog10= 0.
  typvar(1)%long_name='Meridional_Overt.Cell_Global'
  typvar(1)%short_name='zomsfglo'
  typvar%online_operation='N/A'
  typvar%axis='TZY'

  ipk(1) = npk  !  2D

  IF (llglo) THEN
     typvar(2)%name= 'zomsfatl'
     typvar(2)%long_name='Meridional_Overt.Cell_Atlantic'
     typvar(2)%short_name='zomsfatl'

     typvar(3)%name= 'zomsfinp'
     typvar(3)%long_name='Meridional_Overt.Cell_IndoPacif'
     typvar(3)%short_name='zomsfinp'

     typvar(4)%name= 'zomsfind'
     typvar(4)%long_name='Meridional_Overt.Cell_Indian'
     typvar(4)%short_name='zomsfind'

     typvar(5)%name= 'zomsfpac'
     typvar(5)%long_name='Meridional_Overt.Cell_pacif'
     typvar(5)%short_name='zomspac'

     ipk(2) = npk  !  2D
     ipk(3) = npk  !  2D
     ipk(4) = npk  !  2D
     ipk(5) = npk  !  2D
  ENDIF

  PRINT *, 'npiglo=', npiglo
  PRINT *, 'npjglo=', npjglo
  PRINT *, 'npk   =', npk

  ! Allocate arrays
  ALLOCATE ( zmask(jpbasins,npiglo,npjglo) )
  ALLOCATE ( zv(npiglo,npjglo) )
  ALLOCATE ( e1v(npiglo,npjglo),e3v(npiglo,npjglo), gphiv(npiglo,npjglo) ,gdepw(npk) )
  ALLOCATE ( zomsf(jpbasins, npjglo, npk) )
  ALLOCATE ( dumlon(1,npjglo) , dumlat(1,npjglo))

  e1v(:,:) = getvar(coordhgr, 'e1v', 1,npiglo,npjglo) 
  gphiv(:,:) = getvar(coordhgr, 'gphiv', 1,npiglo,npjglo)
  gdepw(:) = getvare3(coordzgr, 'gdepw',npk)
  gdepw(:) = -1.*  gdepw(:)

  iloc=maxloc(gphiv)
  dumlat(1,:) = gphiv(iloc(1),:)
  dumlon(:,:) = 0.   ! set the dummy longitude to 0

  ! create output fileset
   ncout =create(cfileoutnc, 'none', 1,npjglo,npk,cdep='depthw')
   ierr= createvar(ncout ,typvar,jpbasins, ipk,id_varout )
   ierr= putheadervar(ncout, cfilev,1, npjglo,npk,pnavlon=dumlon,pnavlat=dumlat,pdep=gdepw)
   ierr = copy_time(ncout, cfilev)


  ! reading the masks
  ! 1 : global ; 2 : Atlantic ; 3 : Indo-Pacif ; 4 : Indian ; 5 : Pacif
  zmask(1,:,:)=getvar(maskv,'vmask',1,npiglo,npjglo)
  IF ( llglo ) THEN
     zmask(2,:,:)=getvar(cbasinmask,'tmaskatl',1,npiglo,npjglo)
     zmask(4,:,:)=getvar(cbasinmask,'tmaskind',1,npiglo,npjglo)
     zmask(5,:,:)=getvar(cbasinmask,'tmaskpac',1,npiglo,npjglo)
     zmask(3,:,:)=zmask(5,:,:)+zmask(4,:,:)
  ! ensure that there are no overlapping on the masks
     WHERE(zmask(3,:,:) > 0 ) zmask(3,:,:) = 1
  ! change global mask for GLOBAL periodic condition
    IF (( npiglo .EQ. 182) .OR. ( npiglo .EQ. 362 ) .OR. ( npiglo .EQ. 1442 )) THEN
      zmask(:,1,:) = 0.
      zmask(:,npiglo,:) = 0.
    ENDIF
  ENDIF

  DO jt = 1,npt

    ! initialize moc to 0
    zomsf(:,:,:) = 0.

    !Read missing data or _FillValue 
    spval=getatt(cfilev,'vomecrty','_FillValue')
  
    DO jk = 1,npk-1
       ! Get velocities v at jk
       zv(:,:)= getvar(cfilev, 'vomecrty',  jk ,npiglo,npjglo)

       ! get e3v at level jk ( ps...)
       e3v(:,:) = getvar(coordzgr, 'e3v_ps', jk,npiglo,npjglo,ldiom=.true.)
     
       ! integrates 'zonally' (along i-coordinate)
       DO ji=1,npiglo
         ! For all basins 
         DO jbasin = 1, jpbasins
           DO jj=1,npjglo
  ! convert missing data to zero
             if (zv(ji,jj) == spval) then
               zv(ji,jj)=0.
             endif
             zomsf(jbasin,jj,jk)=zomsf(jbasin,jj,jk) - e1v(ji,jj)*e3v(ji,jj)* zmask(jbasin,ji,jj)*zv(ji,jj)
           ENDDO
         END DO
       END DO
    END DO

  ! integrates vertically   from bottom to surface
    DO jk=npk-1 , 1 , -1
       zomsf(:,:,jk) = zomsf(:,:,jk+1) + zomsf(:,:,jk)/1.e6
    END DO  ! loop to next level

  ! netcdf output 
    DO jbasin= 1, jpbasins
      DO jk =1, npk
        ierr = putvar (ncout, id_varout(jbasin),REAL(zomsf(jbasin,:,jk)), jk,jt,npjglo)
      END DO
    END DO

  END DO !End of loop over time

  ierr = closeout(ncout)
 
  END PROGRAM cdfmoc
