# meshFieldsDist
testing integration of omegah and meshFields

## build instructions

## dependencies

The following commands were tested on a SCOREC workstation running RHEL7 with a
Nvidia Turing GPU.

`cd` to a working directory that will contain *all* your source code (including
this directory) and build directories.  That directory is referred to as `root`
in the following bash scripts.

Create a file named `envRhel7_turing.sh` with the following contents:

```
export root=$PWD 
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core 
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core 
module load gcc/10.1.0 cmake cuda/11.4

function getname() {
  name=$1
  machine=`hostname -s`
  buildSuffix=${machine}-cuda
  echo "build-${name}-${buildSuffix}"
}
export kk=$root/`getname kokkos`/install
export oh=$root/`getname omegah`/install
export cab=$root/`getname cabana`/install
export mf=$root/`getname meshFields`/install
CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$cab:$mf:$CMAKE_PREFIX_PATH

cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kk"
```

Create a file named `buildAll_turing.sh` with the following contents:

```
#!/bin/bash -e

#kokkos
cd $root
#tested with kokkos develop@9dff8cc
git clone -b develop git@github.com:kokkos/kokkos.git
cmake -S kokkos -B ${kk%%install} \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DCMAKE_INSTALL_PREFIX=$kk
cmake --build ${kk%%install} -j8 --target install

#omegah
cd $root
git clone git@github.com:SCOREC/omega_h.git
cmake -S omega_h -B ${oh%%install} \
  -DCMAKE_INSTALL_PREFIX=$oh \
  -DBUILD_SHARED_LIBS=OFF \
  -DOmega_h_USE_Kokkos=ON \
  -DOmega_h_USE_CUDA=ON \
  -DOmega_h_CUDA_ARCH=75 \
  -DOmega_h_USE_MPI=ON  \
  -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DKokkos_PREFIX=$kk/lib64/cmake
cmake --build ${oh%%install} -j8 --target install

#cabana
cd $root
git clone git@github.com:ECP-copa/Cabana.git cabana
cmake -S cabana -B ${cab%%install} \
  -DCMAKE_BUILD_TYPE="Debug" \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DCabana_ENABLE_MPI=OFF \
  -DCabana_ENABLE_CAJITA=OFF \
  -DCabana_ENABLE_TESTING=OFF \
  -DCabana_ENABLE_EXAMPLES=OFF \
  -DCabana_ENABLE_Cuda=ON \
  -DCMAKE_INSTALL_PREFIX=$cab
cmake --build ${cab%%install} -j8 --target install

#meshfields
cd $root
git clone -b kokkosController git@github.com:SCOREC/meshFields
cmake -S meshFields -B ${mf%%install} -DCMAKE_INSTALL_PREFIX=$mf
cmake --build ${mf%%install} -j8 --target install
```

Make the script executable:

```
chmod +x buildAll_turing.sh
```


Source the environment script from this work directory:

```
source envRhel7_turing.sh
```

Run the build script:

```
./buildAll_turing.sh
```

## build meshFieldsDist

**The following assumes that the environment file described in the ##dependencies section has been `source`d.**

```
cd $root
git clone git@github.com:SCOREC/meshFieldsDist.git
cmake -S meshFieldsDist -B build-meshFieldsDist-cuda
cmake --build build-meshFieldsDist-cuda
```

## run tests

```
cd $root
ctest --test-dir build-meshFieldsDist-cuda
```
