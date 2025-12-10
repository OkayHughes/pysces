# The [py]thon [s]pe[c]tral element homm[e]  (pysces) dynamical core.
If the lack of an [s] bothers you, you can pretend the acronym is derived from
"The [python]  [s]pe[c]tral element homm[e] i-don't-respect-acronym[s]" dynamical core.


# Design philosophy as of 11/26/2025
The design of the `jax.numpy` module 
allows us to write code that runs identically
whether or not the name `jnp` corresponds to `jax.numpy`
or just `numpy`. This is only true when our code is written
as pure functions–see [https://en.wikipedia.org/wiki/Functional_programming#Pure_functions](here, for an explanation).
This requires us to write code that does not assign to slices of existing arrays. 
Using operators like `+=` with names of arrays is fine,
as the jax code treats this as syntactic sugar for, e.g., `u = u + du`.
Consequently, the memory footprint of the non-jitted numpy code is likely larger than it needs to be,
as simple transformations/renaming that a compiler would perform easily are not possible in interpreted code.
We wish to preserve the numpy-only version of the code because it makes prototyping 
new features SO fast.

In order to have code that functions identically whether or not `jax` is enabled, 
care must be taken when doing things like applying decorators to functions or interfacing with plotting code. 
The `config.py` module is currently the place where we define `jax` configuration and handle these discrepancies.
For example, when `use_jax=True`, `config.jit` corresponds to the `jax.jit` decorator. When `use_jax=False`,
`config.jit` is a custom-defined no-op decorator. I'd like to have as few conditionals that depend on `use_jax`
as possible in the model code, and if it must be done, then the operations performed in the branches
should contain virtually identical operations (possibly with separate implementations). For the moment,
we will lean into including identical interfaces contained in `config.py` whose definitions 
vary based on `jax` configuration.


# Current sharp edges
Due to targeting the `jax` library to perform automatic differentiation, 
the code currently has several idiosyncracies.
* Due to the nature of just-in-time (JIT) compilation, 
the shapes of arguments to jitted functions cannot be used within that 
function. This induces the fortran-like need to pass data-dependent (in our case, grid decomposition dependent)
array dimensions as arguments. These arguments must be marked as `static` in the `jax` JIT decorator,
and a new version of the function will be JIT-compiled whenever the function is called with a different version
of the static parameter.
* There are several constraints on non-prognostic 
variables and the structures that hold them. As mentioned above,
tensor shapes must be passed in as function arguments that
are marked as `static`, and consequently cannot be contained in
a structure (e.g., the `grid` dictionary). The `dims` variable
is used to pass these quantities as a `frozendict` object,
which is hashable (a requirement of `static` parameters). Even though the quantities in `dims` relate directly to 
the dimensions of either the horizontal or vertical grids and errors
will result from inconsistencies between them,
the programming model necessates that `[x]_grid` and `dims` be passed as separate arguments.
In theory, this could be treated as global state
and closed-over in jitted functions instead of being passed as an argument.
However, although this sort of trick works well in pure functional _languages_,
my experience with JIT code in python is that it stymies the ability of the
compiler to provide good error messages when things go wrong. 



# remaining development priorities
* Vertical remap (8 hours of work)
  * Identify commonly used remapping configuration (<1 hour)
  * Implement column-wise remapping function (just use Zerroukat?) (4 hours)
  * Implement reference profile for subtraction (done?)
  * Develop column-wise test for remapping consistency (2 hours)
* Implement scalar advection (12 hours of work)
  * Implement separate time step for scalar advection (2 hours)
  * Implement tracer consistency tests (6 hours+??)
  * Implement one limiter method (4 hours?)
* Hyperviscosity (low priority due to split-form advection)
* IMEX jacobian (around 18 hours of work)
  * Implement shallow jacobian (2 hours)
  * Implement outer loop (4-6 hours)
  * Develop tests (5 hours?)
  * Implement split time tendencies (3 hours)
  * Implement deep Jacobian(2-3 hours)
* unstructured grids (12 hours?)
  * 
* Differentiable limiters?
* MPI (20-40 hours)
 * Refactor all top_level code into "dss_local" and "dss remote" calls, with h_grid split into chunks
 * Implement hilbert curve processor decomposition
 * Create tests for processor decomposition
 * Formulate dss extraction tuples, struct to contain them.
 * Implement DSS assembly, repopulation
 * create MPI wrapper to abstract away the ugly.
