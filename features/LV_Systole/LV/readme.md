

```bash
ffc -l dolfin ActiveLeftVentricle.ufl
mkdir build && cd build && cmake .. && make -j8
mpirun -np 8 ./nonlinear_solver 
```

