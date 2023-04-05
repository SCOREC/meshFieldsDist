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
CMAKE_PREFIX_PATH=$kk:$kk/lib64/cmake:$oh:$cab:$CMAKE_PREFIX_PATH

cm=`which cmake`
echo "cmake: $cm"
echo "kokkos install dir: $kk"
```

Create a file named `buildAll_turing.sh` with the following contents:

```
asdjakdh
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

We need a branch of `meshFields` that is currently under development so we are going to run a different clone command then the one listed on the above wiki page:

```
cd $root
git clone -b kokkosController git@github.com:SCOREC/meshFields
cmake -S meshFields -B build-meshFields-cuda
cmake --build build-meshFields-cuda 
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
