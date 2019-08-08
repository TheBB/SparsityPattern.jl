module SparsityPattern

using SparseArrays: SparseVector, SparseMatrixCSC

export sparsepattern, realize


abstract type AbstractSparsePattern end


"""
    SparseVectorPattern{Ti<:Integer}

Represents a sparsity pattern for a sparse vector.
"""
struct SparseVectorPattern{Ti<:Integer} <: AbstractSparsePattern
    n :: Int
    nzind :: Vector{Ti}
    accummap :: Vector{Int}
end

"""
    SparseMatrixPatternCSC{Ti<:Integer}

Represents a sparsity pattern for a matrix in CSC format.
"""
struct SparseMatrixPatternCSC{Ti<:Integer} <: AbstractSparsePattern
    m :: Int
    n :: Int
    colptr :: Vector{Ti}
    rowval :: Vector{Ti}
    accummap :: Vector{Int}
end


"""
    sparsepattern(I[, len])

Create a vector sparsity pattern with nonzero values at indices `I`,
of total length `len` (defaulting to the maximal index).
"""
sparsepattern(I::AbstractVector{<:Integer}, len::Integer) = sparsepattern!(Vector(I), len)
sparsepattern(I::AbstractVector{<:Integer}) = sparsepattern(I, maximum(I))

"""
    sparsepattern(I, J[, m, n])

Create a matrix sparsity pattern with nonzero values at indices
`(I[k], J[k])`, of total size (m,n) (defaulting to the maximal values
of `I` and `J` respectively).
"""
sparsepattern(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}) =
    sparsepattern(I, J, maximum(I), maximum(J))

function sparsepattern(I::Vector{Ti}, J::Vector{Ti}, m::Integer, n::Integer) where Ti<:Integer
    col_target = Vector{Ti}(undef, n)
    csr_rowptr = Vector{Ti}(undef, m+1)
    csr_colval = Vector{Ti}(undef, length(I))
    csr_perm = Vector{Int}(undef, length(I))
    csc_colptr = Vector{Ti}(undef, n+1)
    csc_rowval = Vector{Ti}()
    csc_perm = Vector{Int}()
    sparsepattern!(I, J, m, n, col_target, csr_rowptr, csr_colval, csr_perm, csc_colptr, csc_rowval, csc_perm)
end


"""
    sparsepattern!(I, len)

Main driver of [`sparsitypattern`](@ref) for vectors, mutating and
re-using the first argument.
"""
function sparsepattern!(I::Vector{Ti}, len::Integer) where Ti<:Integer
    perm = sortperm(I)
    permute!(I, perm)

    accummap = ones(Ti, length(I))

    peek = 1
    target = 1
    for peek in 1:length(I)
        1 <= I[peek] <= len || throw(ArgumentError("Index $(I[peek]) out of range"))
        if I[peek] != I[target]
            target += 1
            I[target] = I[peek]
        end
        accummap[peek] = target
    end

    invpermute!(accummap, perm)
    resize!(I, target)
    SparseVectorPattern(len, I, accummap)
end


"""
   sparsepattern!(I, J, m, n, col_target::Vector{Ti},
                  csr_rowptr::Vector{Ti}, csr_colval::Vector{Ti}, csr_perm::Vector{Ti},
                  csc_colptr::Vector{Ti}, csc_rowval::Vector{Ti}, csc_perm::Vector{Ti})

Main driver of [`sparsitypattern`](@ref) for matrices, mutating and
re-using its arguments.

Input arrays `col_target`, `csr_rowptr`, `csr_colval`, `csr_perm`,
`csc_colptr`, `csc_rowval` and `csc_perm` constitute storage for
intermediate forms, and require

`lenght(col_target) >= n`
`length(csr_rowptr) >= m+1`
`length(csr_colval) >= length(I)`
`length(csr_perm) == length(I)`
`length(csc_colptr) >= n+1`
`length(csc_rowval) >= nnz`
`length(csc_perm) >= nnz`

The final two vectors will be resized when the number of nonzero
entries is known, in case they are not big enough.

You may reuse `I` and `J` for `csc_colptr` and `csc_rowval`.
"""
function sparsepattern!(I::Vector{Ti}, J::Vector{Ti}, m::Integer, n::Integer, col_target::Vector{Ti},
                        csr_rowptr::Vector{Ti}, csr_colval::Vector{Ti}, csr_perm::Vector{Ti},
                        csc_colptr::Vector{Ti}, csc_rowval::Vector{Ti}, csc_perm::Vector{Ti}) where Ti<:Integer
    # This function is adapted from SparseArrays.sparse! in base Julia
    # It is identical except for the handling of nonzeros
    nvals = length(I)

    # Compute number of occurences of each row
    fill!(csr_rowptr, 0)
    for i in I
        1 <= i <= m || throw(ArgumentError("Row index $i out of range"))
        csr_rowptr[i+1] += 1
    end

    # Compute CSR row-pointers shifted forward by one
    csr_rowptr[1] = 1
    _cumsum!(@view csr_rowptr[2:end])

    # Create a CSR representation with uncombined entries
    length(csr_perm) != nvals && resize!(csr_perm, nvals)
    for (peek, (i, j)) in enumerate(zip(I, J))
        1 <= j <= n || throw(ArgumentError("Column index $j out of range"))
        target = csr_rowptr[i+1]
        csr_rowptr[i+1] += 1
        csr_colval[target] = j
        csr_perm[peek] = target
    end

    # Sweep through the CSR representation, doing a number of things at once:
    # 1: calculate the column counts shifted by one
    # 2: detect and collapse repeated entries
    fill!(csc_colptr, zero(Ti))
    fill!(col_target, zero(Ti))
    target = one(Ti)              # Next unwritten entry in the compressed form
    last_row_target = one(Ti)     # Previous row's highest target index
    rowptr = csr_rowptr[1]        # Start of next row (in the uncompressed form)
    accummap = Vector{Ti}(undef, nvals)
    for row in 1:m
        rowptr_end = csr_rowptr[row+1]
        for peek in rowptr:csr_rowptr[row+1]-1
            col = csr_colval[peek]
            if col_target[col] < last_row_target
                col_target[col] = target
                csr_colval[target] = col
                accummap[peek] = target
                target += 1
                csc_colptr[col+1] += 1
            else
                accummap[peek] = col_target[col]
            end
        end
        last_row_target = target
        rowptr = rowptr_end
        csr_rowptr[row+1] = target
    end

    # Compute the CSC col-pointers shifted forward by one
    csc_colptr[1] = 1
    _cumsum!(@view csc_colptr[2:end])

    # Finalize the CSC form
    nnz = csr_rowptr[m+1] - 1
    length(csc_rowval) < nnz && resize!(csc_rowval, nnz)
    length(csc_perm) < nnz && resize!(csc_perm, nnz)
    for row in 1:m
        for peek in csr_rowptr[row]:csr_rowptr[row+1]-1
            col = csr_colval[peek]
            target = csc_colptr[col+1]
            csc_colptr[col+1] += 1
            csc_rowval[target] = row
            csc_perm[peek] = target
        end
    end

    # Backtrack the nzval accumulation map to original indices
    for i in 1:nvals
        accummap[i] = csc_perm[accummap[i]]
    end
    permute!(accummap, csr_perm)

    SparseMatrixPatternCSC(m, n, csc_colptr, csc_rowval, accummap)
end


"""
    realize(::Union{SparseVectorPattern,SparseMatrixPatternCSC}, V[, combine])

Create a sparse vector or matrix using the given sparsity pattern and
vector `V` of nonzeros.  Repeated entries are combined with the
`combine` function, defaulting to `|` for booleans and `+` otherwise.

The vector `V` must be as long as the original index vector(s) used to
create the pattern.
"""
realize(pattern::AbstractSparsePattern, V::AbstractVector) = realize(pattern, V, +)
realize(pattern::AbstractSparsePattern, V::AbstractVector{Bool}) = realize(pattern, V, |)

function realize(pattern::SparseVectorPattern, V::AbstractVector, combine::Function)
    nzval = _realize(pattern.accummap, length(pattern.nzind), V, combine)
    SparseVector(pattern.n, pattern.nzind, nzval)
end

function realize(pattern::SparseMatrixPatternCSC, V::AbstractVector, combine::Function)
    nzval = _realize(pattern.accummap, length(pattern.rowval), V, combine)
    SparseMatrixCSC(pattern.m, pattern.n, pattern.colptr, pattern.rowval, nzval)
end


function _realize(accummap, len, V::AbstractVector{Tv}, combine::Function) where Tv
    if length(V) != length(accummap)
        throw(ArgumentError("expected vector of length $(length(accummap)), got $(length(V))"))
    end

    nzval = zeros(Tv, len)
    for (target, value) in zip(accummap, V)
        nzval[target] = combine(nzval[target], value)
    end
    nzval
end

function _cumsum!(vec::AbstractVector{Ti}, accum=one(Ti)) where Ti
    for i in eachindex(vec)
        temp = vec[i]
        vec[i] = accum
        accum += temp
    end
end


end # module
