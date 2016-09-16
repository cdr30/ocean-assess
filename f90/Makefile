# Makefile for CDFTOOLS 

# ( make.macro is a link that points to the file macro.xxx where 
#   xxx is representative of your machine )
# !!  $Rev: 249 $
# !!  $Date: 2009-05-26 18:53:30 +0200 (Tue, 26 May 2009) $
# !!  $Id: Makefile 249 2009-05-26 16:53:30Z forge $
# !!--------------------------------------------------------------


include make.macro

CDFTOOLS=CDFTOOLS-2.1

EXEC = cdfmoc cdfmeanvar

all: $(EXEC)

## Transport programs

cdfmoc: cdfio.o  cdfmoc.f90
	$(F90) cdfmoc.f90 -o cdfmoc cdfio.o $(FFLAGS)

## Averaging programs
cdfmeanvar: cdfio.o cdfmeanvar.f90
	$(F90) cdfmeanvar.f90 -o cdfmeanvar cdfio.o $(FFLAGS)

## Modules

cdfio.o: cdfio.f90
	$(F90) -c  cdfio.f90 $(FFLAGS)

eos.o: eos.f90
	$(F90) -c eos.f90 $(FFLAGS)

modpoly.o: modpoly.f90
	$(F90) -c modpoly.f90 $(FFLAGS)

## Utilities
tar:
	( cd ../ ; tar cf cdftools-2.1.tar $(CDFTOOLS)/*90 $(CDFTOOLS)/Make* \
          $(CDFTOOLS)/section.dat $(CDFTOOLS)/JOBS $(CDFTOOLS)/DOC \
          $(CDFTOOLS)/macro.* )

clean:
	\rm -f *.mod *.o  *~

cleanexe: clean
	\rm -f $(EXEC)

install:
	\cp $(EXEC)  $(INSTALL)
