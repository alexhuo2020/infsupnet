# InfSupNet: a package for solving PDEs


To install, run 

```
pip install infsupnet
```

Examples are provided in the examples folder. The examples include various Poisson equation, semilinear and nonlinear elliptic equation with various boundary conditions.

The source code is provided in the "src" folder.
The Mento-Carlo numerical integration method is used and the data generation method is provided in "dataset.py". To define different differential operators, one can redefine "opA" (for the equation) and "opB" (for the boundary condition), see the semilinar/nonlinear example. 