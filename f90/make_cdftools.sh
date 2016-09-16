#! /bin/bash

ARCH=$1
ARCH_BASE='arch_files/make.macro.'

# Platform-specific include file
arch_file=$ARCH_BASE$ARCH
arch_avail=()

# Available files
for a in $(ls $ARCH_BASE*); do
    arch_avail+='  '$(echo $a | cut -c$(($(expr length $ARCH_BASE) + 1))-)
done

# Attempt to find and link include file
if [[ $ARCH == '' ]] ; then
    echo USAGE: make_cdftools.sh arch_name
    echo where arch_name is one of: ${arch_avail[*]}
    exit 1
elif [[ $(ls $arch_file 2> /dev/null) == '' ]] ; then
    echo Arch file \"$arch_file\" does not exist.
    echo Create this file or use one of: ${arch_avail[*]}
    exit 1
else
    ln -sf $arch_file make.macro
fi

# Platform-specific setup
case $ARCH in
    JASMIN*)
        module load intel/14.0
        module load netcdff/intel/14.0/4.2
        ;;
esac

# Run make
make -B
