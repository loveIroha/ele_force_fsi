[![](https://github.com/npuheart/npuheart/actions/workflows/compile.yml/badge.svg)](https://github.com/npuheart/npuheart/actions/workflows/compile.yml)

# NPUHEART

An implementation of the immersed boundary (IB) method with GPU accelerated.

## Environment

1. UBUNTU 20.04
1. CUDA 12.3
2. GCC 11.4.0
3. PYTHON 3.10.10

## Dependents

* FEniCS version 2019.1.0
* Kokkos version 4.0.0
* Spack version developer

```bash
git clone https://github.com/npuheart/npuheart.git
cd npuheart && source ./features/.bashrc
mkdir -p build && cd build && cmake .. && make fsi_ideal_LV_systole -j32
```

cmake更改参数
```
cmake -DNPUHEART_FE_DEGREE=1 -DNPUHEART_DIMENSION=2 -DNPUHEART_ADVECTION=false ..
```


## Automated test

## Documentation



## Citing

```
@article{npuheart,
  doi = {xxxx/xxxxx},
  url = {https://doi.org/xxxx/xxxxx},
  year  = {2024},
  month = {sept},
  publisher = {XXX XXX},
  volume = {x},
  number = {xx},
  pages = {xxxx},
  author = {Ma Pengfei},
  title = {npuheart: An implementation of the immersed boundary (IB) method with GPU accelerated},
  journal = {XX XXX XXXX XXXX XX XXXXX}
}
```

## Known issues

## Contributing



## 覆盖率测试
将测试结果上传到cdash，然后查看测试结果。
```
source .bashrc
make upload
```

将性能测试结果上传到notion
```
python3 features/test_tools/upload_to_notion.py /home/kokkos/ssh/npuheart/build
```









## 多重网格

多重网格由以下几个部分组成：
1. PhysicsSolver/multigrid 文件夹下存放着多重网格类的实现
2. GPU 文件夹下有多重网格的 GPU 加速代码
目前实现的功能：
1. 1D、2D、3D的热传导方程、泊松方程、Stokes方程的数值算例
2. 零Neumann边界条件和零Dirichlet边界条件
3. 网格单元分为三种类型：INTERIOR DIRICHLET NEUMANN，定义在PhysicsSolver/multigrid/MultigridBase.hw文件中


## 单位
基本单位时统一使用厘米（cm）、克（g）、秒（s），其他单位以此派生。