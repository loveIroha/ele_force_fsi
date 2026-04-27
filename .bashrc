

export NPUHEART_ENV=kokkos


. ~/spack/share/spack/setup-env.sh
spack env deactivate
spack load gcc@11.4.0
spack env activate -p $NPUHEART_ENV
# spack load cmake
# spack load /t5z6gjb # cmake
spack load cmake
spack load py-pip


export SPACK_VIEW=~/spack/var/spack/environments/$NPUHEART_ENV/.spack-env/view
export SPACK_CXX=$SPACK_VIEW/bin/nvcc_wrapper
# export NPUHEART_GEOMETRY_PATH=~/geometry-tiny/
export NPUHEART_GEOMETRY_PATH=~/modelscope/MV-geometry/


# OPENMP的环境变量
export OMP_NUM_THREADS=20
export OMP_PROC_BIND=spread OMP_PLACES=threads

# CUDA的环境变量(Kokkos使用)
export CUDA_VISIBLE_DEVICES=0