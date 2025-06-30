# Quantum Approximate Multi-Objective Optimization (arXiv:2503.22797)
Code and data accompanying the paper 
[Quantum Approximate Multi-Objective Optimization](https://arxiv.org/abs/2503.22797) 
by Ayse Kotil, Elijah Pelofske, Stephanie Riedmüller, Daniel J. Egger, Stephan Eidenbenz, Thorsten Koch, and Stefan Woerner.

The repository is structured as follows:
- `classical_algorithms`: Scripts to run the exact classical algorithms (DPA-a / DCM), the code of the algorithms themselves is provided in https://github.com/stephaniehoffmann/classical-algorithms-moip.
- `data`: Problem instances and results data.
- `figures`: All figures shown in the paper.
- `notebooks`: Notebooks to run the algorithms (except DPA-a, DCM, and JuliQAOA, cf. corresponding folders) and to create figures.
- `parameter_training`: Code to run JuliQAOA to train parameters.
- `qamoo`: Python code for all algorithms presented in the paper (except DPA-a / DCM, cf. corresponding folder).

The code uses Python 3.10 and all requirements except `pygmo` can be installed via `pip install -e .`. 
`pygmo` (2.19.7) is best installed following the guidance here: https://esa.github.io/pygmo2/ (also see the comments below on the installation).

To train parameters, [JuliQAOA](https://arxiv.org/abs/2312.06451) needs to be installed:
https://github.com/lanl/JuliQAOA.jl.

The experiments on real quantum hardware require access to the [Qiskit IBM Runtime](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/runtime-service). Alternatively, Matrix Product State (MPS) simulations can be performed using [Qiskit Aer](https://qiskit.github.io/qiskit-aer/).

Some classical algorithms require access to commercial solvers like CPLEX or GUROBI, which provide academic licenses that may be applicable in certain cases.

### Installing pygmo2

1. Using Conda + Any OS

If you are using Conda to manage your Python packages, you can run the following commands to install pygmo, regardless of your operating system:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pygmo
```

2. Using pip + Linux 

If you are using pip to manage your Python packages on Linux, you can directly run `pip install pygmo` in your virtual environment. 
Note that conda is recommended to install pygmo instead of pip due to possible dependency conflicts.

3. Installation from source

Make sure to have the libraries installed in your system as listed here: https://esa.github.io/pygmo2/install.html#installation-from-source. If you are using homebrew on OSX, you can install the dependencies directly with `brew install cmake boost pagmo pybind11`.

Then, in the qamoo project folder, run the following commands to clone the pygmo repository:

```
git clone https://github.com/esa/pygmo2.git
cd pygmo2
mkdir build 
cd build
```

Then, make sure you are in your virtual Python environment and install the library:

- on Unix-like systems:

```
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make
sudo make install
```

- on Windows:

```
cmake .. -DPYTHON_EXECUTABLE=path/to/python/exe
cmake --build .
cmake --build . --target install
```

For a detailed installation guide, please refer to https://esa.github.io/pygmo2/install.html#installation-from-source.
