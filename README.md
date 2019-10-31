# InterpretMe

[![Build Status](https://travis-ci.com/mmahrouss/InterpretMe.jl.svg?branch=master)](https://travis-ci.com/mmahrouss/InterpretMe.jl) [![Build Status](https://ci.appveyor.com/api/projects/status/github/mmahrouss/InterpretMe.jl?svg=true)](https://ci.appveyor.com/project/mmahrouss/InterpretMe-jl) [![Codecov](https://codecov.io/gh/mmahrouss/InterpretMe.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mmahrouss/InterpretMe.jl) [![Coveralls](https://coveralls.io/repos/github/mmahrouss/InterpretMe.jl/badge.svg?branch=master)](https://coveralls.io/github/mmahrouss/InterpretMe.jl?branch=master)

InterpretMe is Julia Package aimed at providing interpret-ability tools for Machine Learning models in Julia.

## Installation

```julia
julia>]
pkg> add InterpretMe
```

## Example

First import the package:

```julia
Using InterpretMe
```

Then train a model from [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)'s examples. We are going to use the [Iris Example](https://github.com/cstjean/ScikitLearn.jl/blob/master/notebooks/Iris.ipynb)

Load the packages and train the model:

```julia
import ScikitLearn
import RDatasets

ScikitLearn.@sk_import linear_model: LogisticRegression

iris = RDatasets.dataset("datasets", "iris")

X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

model = ScikitLearn.fit!(LogisticRegression(), X, y)
```

Now we can calculate the partial_dependance function from InterpretMe.

```julia
pd, vals = partial_dependence(x, [1], x -> ScikitLearn.predict_proba(model,x),
                              100, false)
```
