using Test
using SparsityPattern
using SparseArrays


@testset "vector pattern" begin
    pat = @inferred(sparsepattern([1,2,3,4]))
    @test pat.n == 4
    @test pat.nzind == [1,2,3,4]
    @test pat.accummap == [1,2,3,4]

    pat = @inferred(sparsepattern([1,2,3,4], 8))
    @test pat.n == 8
    @test pat.nzind == [1,2,3,4]
    @test pat.accummap == [1,2,3,4]

    pat = @inferred(sparsepattern([1,1,1,1], 4))
    @test pat.n == 4
    @test pat.nzind == [1]
    @test pat.accummap == [1,1,1,1]

    pat = @inferred(sparsepattern([2,2,2,2], 4))
    @test pat.n == 4
    @test pat.nzind == [2]
    @test pat.accummap == [1,1,1,1]

    pat = @inferred(sparsepattern([2,3,2,3], 4))
    @test pat.n == 4
    @test pat.nzind == [2,3]
    @test pat.accummap == [1,2,1,2]

    @test_throws ArgumentError sparsepattern([0], 1)
    @test_throws ArgumentError sparsepattern([2], 1)
end


@testset "vector realize" begin
    vec = @inferred(realize(sparsepattern([1,2,3,4]), [5,2,0,3]))
    @test vec == [5,2,0,3]
    @test nnz(vec) == 4

    vec = @inferred(realize(sparsepattern([4,3,2,1]), [5,2,0,3]))
    @test vec == [3,0,2,5]
    @test nnz(vec) == 4

    vec = @inferred(realize(sparsepattern([1,1,1,1], 4), [1,2,3,4]))
    @test vec == [10,0,0,0]
    @test nnz(vec) == 1

    vec = @inferred(realize(sparsepattern([2,2,2,2], 4), [1,2,3,4]))
    @test vec == [0,10,0,0]
    @test nnz(vec) == 1

    vec = @inferred(realize(sparsepattern([2,3,2,3], 4), [1,2,3,4]))
    @test vec == [0,4,6,0]
    @test nnz(vec) == 2

    pat = sparsepattern([2,3,2,3], 4)
    @test_throws ArgumentError realize(pat, [1,2,3,4,5])
    @test_throws ArgumentError realize(pat, [1,2,3])
end


@testset "matrix pattern" begin
    pat = @inferred(sparsepattern([1,2,3,4], [1,2,3,4]))
    @test pat.m == 4
    @test pat.n == 4
    @test pat.colptr == [1,2,3,4,5]
    @test pat.rowval == [1,2,3,4]
    @test pat.accummap == [1,2,3,4]

    pat = @inferred(sparsepattern([1,2,3,4], [1,2,3,4], 8, 8))
    @test pat.m == 8
    @test pat.n == 8
    @test pat.colptr == [1,2,3,4,5,5,5,5,5]
    @test pat.rowval == [1,2,3,4]
    @test pat.accummap == [1,2,3,4]

    pat = @inferred(sparsepattern([1,1,1,1], [1,1,1,1], 4, 4))
    @test pat.m == 4
    @test pat.n == 4
    @test pat.colptr == [1,2,2,2,2]
    @test pat.rowval == [1]
    @test pat.accummap == [1,1,1,1]

    pat = @inferred(sparsepattern([1,2,1,2,1,2], [3,3,3,4,4,4], 4, 4))
    @test pat.m == 4
    @test pat.n == 4
    @test pat.colptr == [1,1,1,3,5]
    @test pat.rowval == [1,2,1,2]
    @test pat.accummap == [1,2,1,4,3,4]

    @test_throws ArgumentError sparsepattern([0], [1], 1, 1)
    @test_throws ArgumentError sparsepattern([2], [1], 1, 1)
    @test_throws ArgumentError sparsepattern([1], [0], 1, 1)
    @test_throws ArgumentError sparsepattern([1], [2], 1, 1)
end


@testset "matrix realize" begin
    mx = @inferred(realize(sparsepattern([1,2,3], [1,2,3]), [5,7,9]))
    @test mx == [5 0 0; 0 7 0; 0 0 9]
    @test nnz(mx) == 3

    mx = @inferred(realize(sparsepattern([1,1,1], [1,1,1], 3, 3), [1,2,3]))
    @test mx == [6 0 0; 0 0 0; 0 0 0]
    @test nnz(mx) == 1

    mx = @inferred(realize(sparsepattern([1,2,1,2,1,2], [3,3,3,4,4,4]), [1,2,3,4,5,6]))
    @test mx == [0 0 4 5; 0 0 2 10]
    @test nnz(mx) == 4

    pat = sparsepattern([1,2,3], [1,2,3])
    @test_throws ArgumentError realize(pat, [5,7])
    @test_throws ArgumentError realize(pat, [5,7,9,11])

    mx1 = realize(pat, [1,2,3])
    mx2 = realize(pat, [3,2,1])
    @test mx1.colptr === mx2.colptr
    @test mx1.rowval === mx2.rowval
end
