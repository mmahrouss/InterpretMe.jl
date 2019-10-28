using InterpretMe
using Test
import ScikitLearn
import RDatasets

ScikitLearn.@sk_import linear_model: LogisticRegression
ScikitLearn.@sk_import inspection: partial_dependence

@testset "InterpretMe.jl" begin
    # Write your own tests here.
    begin
        iris = RDatasets.dataset("datasets", "iris")
        x = convert(Matrix, iris[!,[:SepalLength,:SepalWidth,
                                    :PetalLength,:PetalWidth ]])
        y = convert(Array, iris.Species)
        model = ScikitLearn.fit!(LogisticRegression(), x, y)
        averaged_predictions, values = partial_dependence(model, x, [0],
                                                          grid_resolution=100)
        pd, vals = InterpretMe.partial_dependence(x, [1],
                                                 x -> ScikitLearn.predict_proba(
                                                                       model,x),
                                                  100, false)
        @test all(pd .≈ transpose(averaged_predictions))
    end
end
