# SparsityPattern

[![Build status](https://api.travis-ci.org/TheBB/SparsityPattern.jl.svg?branch=master)](https://travis-ci.org/TheBB/SparsityPattern.jl)

This is a small package for efficiently creating multiple sparse
vectors or matrices which differ only in their values and not their
sparsity patterns. Since most of the work involved in constructing
sparse matrices is the compression of the sparsity pattern, this lets
us save computational cost by doing it *once* instead of several times
over. It also permits saving memory by reusing the index arrays used
for storing sparsity patterns.

## Usage

```julia
using SparsityPattern
using SparseArrays

# This is the conventional way to construct sparse matrices in Julia
A1 = sparse(I, J, V1)
A2 = sparse(I, J, V2)

# With SparsityPattern, we can do this
pattern = sparsepattern(I, J)
B1 = realize(pattern, V1)
B2 = realize(pattern, V2)
```

The two calls to *sparse* waste computational resources by performing
many of the same computations twice --- it is not generally the data
array *V* that makes sparse array construction expensive.

Most of this work is done *once* in the call to *sparsepattern*, and
the two calls to *realize* are comparatively cheap. Moreover, the two
sparse matrices will share storage for the internal CSC structure.

**Warning:** Because of this, mutating operations on the resulting
sparse matrices may be unsafe!
