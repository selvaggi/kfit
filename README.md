This  performs a kinematic fit on set of parameters given a set of
measured parameters with one or more analytical constraints. The derivatives of the constraint vector are computed analytically. The minimization is performed
using iteratively the Lagrange multiplier method using definitions and
conventions from : https://arxiv.org/pdf/1911.10200.pdf

## Setting up your environment

This package requires ROOT, Python v3, numpy and sympy.

```bash
pip3 install --user sympy numpy
```

## Run the examples
```bash
python3 example_simple.py
```
