using InterpretMe
using Conda
using ScikitLearn
using RDatasets
using Test
import PyCall
try
    PyCall.pyimport_conda("sklearn", "scikit-learn")
catch
    run(`/usr/bin/python3 -m pip install sklearn`)
# Conda.add("scikit-learn")

@sk_import linear_model: LogisticRegression
@sk_import inspection: partial_dependence

@testset "InterpretMe.jl" begin
    # Write your own tests here.
    begin
        iris = RDatasets.dataset("datasets", "iris")
        x = convert(Matrix, iris[!,[:SepalLength,:SepalWidth,
                                    :PetalLength,:PetalWidth ]])
        y = convert(Array, iris.Species)
        model = fit!(LogisticRegression(), x, y)
        averaged_predictions, values = partial_dependence(model, x, [0],
                                                          grid_resolution=100)
        pd, vals = partial_dependance(x, [1], x -> predict_proba(model,x),
                                       100, false)
        @test all(pd .≈ transpose(averaged_predictions))
    end
end
