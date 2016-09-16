PROGRAM cdfmeanvar
  !!-------------------------------------------------------------------
  !!               ***  PROGRAM cdfmeanvar  ***
  !!
  !!  **  Purpose  :  Compute the Mean Value and variance  over the ocean
  !!                  PARTIAL STEPS
  !!  
  !!  **  Method   :  compute the sum ( V * e1 *e2 * e3 *mask )/ sum( e1 * e2 * e3 *mask )
  !!
  !!
  !! history ;
  !!  Original :  J.M. Molines (Oct. 2005) 
  !!              J.M. Molines  Add variance Nov. 2006
  !!-------------------------------------------------------------------
  !!  $Rev: 233 $
  !!  $Date: 2009-04-28 19:33:08 +0200 (Tue, 28 Apr 2009) $
  !!  $Id: cdfmeanvar.f90 233 2009-04-28 17:33:08Z molines $
  !!--------------------------------------------------------------
  !! * Modules used
  USE cdfio

  !! * Local variables
  IMPLICIT NONE
  INTEGER   :: jk, ik, jt, jd
  INTEGER   :: kmin=0, kmax=0                      !: domain limitation for computation
  INTEGER   :: ierr                                !: working integer
  INTEGER   :: narg, iargc                         !: command line 
  INTEGER   :: npiglo,npjglo,npk,nt                !: size of the domain
  INTEGER   :: nvpk,ndom                           !: vertical levels in working variable

  REAL(KIND=4), DIMENSION (:,:),   ALLOCATABLE ::  e1, e2, e3,  zv   !:  metrics, variable
  REAL(KIND=4), DIMENSION (:,:),   ALLOCATABLE ::  zmask, dmask      !:   npiglo x npjglo
  REAL(KIND=4)                                 ::  fillvalue         !: missing value

  REAL(KIND=8), DIMENSION (:), ALLOCATABLE :: zvol, zsum, zvar
  REAL(KIND=8)      :: zsum2d, zvar2d, zvol2d
  CHARACTER(LEN=300) :: cfilev , cdum
  CHARACTER(LEN=300) :: coordhgr,  coordzgr, cmask, domain_mask
  CHARACTER(LEN=300) :: cvar, cvartype
  CHARACTER(LEN=20) :: ce1, ce2, ce3, cvmask, cvtype, cdep, cvmaskutil
  CHARACTER(LEN=20), DIMENSION (:), ALLOCATABLE :: domains

  INTEGER    :: istatus

  ! constants

  !!  Read command line and output usage message if not compliant.
  narg= iargc()
  IF ( narg == 0 ) THEN
     PRINT *,' Usage : cdfmeanvar  ncfile cdfvar T| U | V | F | W  hgr.nc zgr.nc mask.nc kmin kmax [domain_mask.nc domains]'
     PRINT *,' Computes the mean value, and the spatial variance of the field (3D, weighted) '
     PRINT *,' domain_mask.nc is an optional netcdffile containing 2D masks defining domains to average over'
     PRINT *,' domains is a list of variable names in domain_mask.nc'
     PRINT *,' PARTIAL CELLS VERSION'
     PRINT *,' Output on standard output'
     STOP 9
  ENDIF

  CALL getarg (1, cfilev)
  CALL getarg (2, cvar)
  CALL getarg (3, cvartype)
  CALL getarg (4, coordhgr)
  CALL getarg (5, coordzgr)
  CALL getarg (6, cmask)
  CALL getarg ( 7,cdum) ; READ(cdum,*) kmin
  CALL getarg ( 8,cdum) ; READ(cdum,*) kmax
  
  IF (narg > 8 ) THEN
    IF (narg == 9) THEN
       PRINT *,'If domain_masks.nc is provided you must provide a list of domains as well'
    ! input optional imin imax jmin jmax
    ELSE
       CALL getarg (9, domain_mask)
       ndom = narg - 9
       ALLOCATE( domains(ndom))
       DO jd = 1,ndom
         CALL getarg (jd+9, domains(jd) )
       END DO
    ENDIF
  ELSE
    ndom = 1
    ALLOCATE( domains(ndom))
    domains(1) = 'None'
  ENDIF

  npiglo= getdim (cfilev,'x',ldexact=.true.)
  npjglo= getdim (cfilev,'y',ldexact=.true.)
  npk   = getdim (cfilev,'depth')
  nvpk  = getvdim(cfilev,cvar)
  nt    = getdim (cfilev,'time_counter')
  
  IF (nvpk == 2 ) nvpk = 1
  IF (nvpk == 3 ) nvpk = npk

  !Deal with kmin and kmax
  IF (nvpk .gt. kmax-kmin+1) nvpk = kmax-kmin+1

  ! Allocate arrays
  ALLOCATE ( zmask(npiglo,npjglo), dmask(npiglo,npjglo) )
  ALLOCATE ( zv(npiglo,npjglo) )
  ALLOCATE ( e1(npiglo,npjglo),e2(npiglo,npjglo), e3(npiglo,npjglo) )
  ALLOCATE ( zsum(ndom), zvol(ndom), zvar(ndom) )
  SELECT CASE (TRIM(cvartype))
  CASE ( 'T' )
     ce1='e1t'
     ce2='e2t'
     ce3='e3t_ps'
     cvmask='tmask'
     cvmaskutil='tmaskutil'
  CASE ( 'U' )
     ce1='e1u'
     ce2='e2u'
     ce3='e3t_ps'
     cvmask='umask'
     cvmaskutil='umaskutil'
  CASE ( 'V' )
     ce1='e1v'
     ce2='e2v'
     ce3='e3t_ps'
     cvmask='vmask'
     cvmaskutil='vmaskutil'
  CASE ( 'F' )
     ce1='e1f'
     ce2='e2f'
     ce3='e3t_ps'
     cvmask='fmask'
     cvmaskutil='fmaskutil'
  CASE ( 'W' )
     ce1='e1t'
     ce2='e2t'
     ce3='e3w_ps'
     cvmask='tmask'
     cvmaskutil='tmaskutil'
  CASE DEFAULT
      PRINT *, 'this type of variable is not known :', trim(cvartype)
      STOP
  END SELECT
  
  e1(:,:) = getvar(coordhgr, ce1, 1,npiglo,npjglo)
  e2(:,:) = getvar(coordhgr, ce2, 1,npiglo,npjglo)

  DO jt = 1,nt
     zvol=0.d0
     zsum=0.d0
     DO jk = 1,nvpk
        ik = jk+kmin-1
        IF ( nvpk /= 1 .OR. jt == 1 ) THEN
           ! if there is only one level do not read mask and e3 every time step ...
           zmask(:,:)=getvar(cmask,cvmask,ik,npiglo,npjglo)
           ! get e3 at level ik ( ps...)
           e3(:,:) = getvar(coordzgr, ce3, ik,npiglo,npjglo, ldiom=.true.)
        END IF
        DO jd = 1,ndom
           IF (domains(jd) /= 'None') THEN
              dmask = getvar(domain_mask, domains(jd), 1, npiglo, npjglo)
           ELSE
              dmask = getvar(cmask, cvmaskutil, 1, npiglo, npjglo )
           ENDIF
           ! Get velocities v at ik
           zv(:,:)= getvar(cfilev, cvar,  ik ,npiglo,npjglo, ktime=jt)
           fillvalue=getatt(cfilev,cvar,'_FillValue')
           where(zv == fillvalue) zv = 0.0

           ! 
           zvol2d=sum(e1 * e2 * e3 * zmask * dmask )
           zvol(jd)=zvol(jd)+zvol2d
           zsum2d=sum(zv*e1*e2*e3*zmask*dmask)
           zvar2d=sum(zv*zv*e1*e2*e3*zmask*dmask)
           zsum(jd)=zsum(jd)+zsum2d
           zvar(jd)=zvar(jd)+zvar2d
        END DO
     END DO
  END DO

  PRINT *,'Domain ','Mean ','Variance '
  DO jd = 1,ndom
     PRINT *, domains(jd), zsum(jd)/zvol(jd), zvar(jd)/zvol(jd) - (zsum(jd)/zvol(jd))*(zsum(jd)/zvol(jd))
  END DO
  
END PROGRAM cdfmeanvar
