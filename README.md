# Overview
The purpose of this project is to create the a highly readable, well documented, well tested atmospheric dynamical core with support for variable resolution meshes with support for automatic differentiation and machine learing.
This project prioritizes code readability and maintainability over performance. 
We want to minimize external dependencies and, insofar as it is possible, create a codebase that is entirely written in python.
Given the constraints of these design decisions, it is unlikely that the resulting dynamical core will scale to hundreds or thousands of  nodes on an HPC computing system. This aligns with our stated goal of making atmospheric modeling accessible.

# Quickstart with `uv`

## Common steps
* Run `git clone git@github.com:OkayHughes/pysces.git` and navigate to the `pysces` directory.
* Install `uv`, e.g. `pip install uv` in your python environment. `uv` is a 
modern environment manager for Python that we use for development and testing.

## MPI-dependent steps
If you don't have MPI installed, run:
* Run `uv sync --group mpich --group dev --group jax --group torch`
* If you don't plan to use either jax or torch, you can omit one or both of those groups.
If you have a system MPI installed (e.g. on HPC systems), run:
* Run `which mpicc`
* Create a `.env` file containing `export MPICC=${MPICC_LOCATION}`
* Run `uv sync --env-file .env --group dev --group jax --group torch`
* If you don't plan to use either jax or torch, you can omit one or both of those groups.

## Run tests
* Navigate to the `tests` directory and run `bash run_test.sh`. These should catch if there are problems
with CPU configurations of `pysces`, and test whether there are issues with your MPI environment.


# Accelerator support
Most scientific code can be easily written in a (nearly) [purely functional](https://en.wikipedia.org/wiki/Pure_function) programming style. Consequently, this means that the codebase can be written to satisfy the requirements of the [Jax](https://github.com/jax-ml/jax) library's just-in-time compilation and automatic differentiation, while retaining the ability to run with array/tensor operations provided by [PyTorch](https://pytorch.org/) or [Numpy](https://numpy.org/). Due to Google's history of abruptly discontinuing widely used software products, we have chosen to future proof this code base by ensuring that GPU parallelism (and ideally automatic differentiation) can be sourced from any library that provides an array implementation that resembles `Numpy.ndarray`. 

# Jax support
The Jax configuration of the code is significantly more performant than either the Numpy or Torch configurations. 
This is because the allowable control flow constructs of `jax.jit` are significantly less constraining than,
e.g., `numba` or `torch.compile`. Consequently, the programming style of this project treats the idiosyncracies of the
Jax functional programming model as authorative. This means that Numpy and Torch performance suffers, as
Jax disallows assignment of slices of arrays after initial assignment, so code like 
```
x = np.eye((3, 3))
x[:, 0] = x[:, 1]
```
is entirely disallowed outside of initialization. Instead, this would be written
```
x = np.eye((3, 3))
x = np.stack((x[:, 1], x[:, 1:]), axis=1)
```
which appears almost deliberately tailored to force Numpy/Torch view semantics
to copy the data in `x`. 

# PyTorch support
PyTorch can also provide automatic differentiation capabilities and GPU parallelism. 
This is used as a fallback, as the dominance of PyTorch at so-called "AI" companies makes
it more likely to be supported in the future. Therefore, we are trying to ensure that code runs
with PyTorch as a backend, but we don't typically optimize for PyTorch performance.

# Policy on intellectual property

The view of the maintainer is that since the training data 
used to train LLMS was obtained without the consent of the people who made it, 
LLMs trained on this data are structurally incapable
of determining when they are commiting plagiarism or theft.

Imagine that a guest at your dinner party shows up with a bottle of wine
that they stole from their brother-in-law's house. You won't know
that it was stolen unless they brag to you about it.
Analogously, we cannot necessarily know whether you used an LLM
to generate code that you're trying to commit, but you should feel
about the same way about contributing LLM-generated code as you
would showing up to a dinner party with something you stole from
a relative's house.
