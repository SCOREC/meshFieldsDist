# meshFieldsDist
testing integration of omegah and meshFields

## build instructions

## dependencies

These only need to be installed once.

See https://github.com/SCOREC/meshFields#build-dependencies

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
