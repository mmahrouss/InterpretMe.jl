using LinearAlgebra
using Statistics
"""
    InterpretMe.partial_dependance(data::Array{Any,2},
        features::Vector{<:Integer}
        ,feature::T, estimate_fn, grid_size::Int, return_ice::Bool)
         where {T<:AbstractString}
Calculate the partial dependance of a machine learning model.

`estimate_fn` is a function from the model that generates a vector of predictions from a matrix of inputs.
`feature_names` should be the feature names corresponding to the features in `data`
`return_ice` whether to return the ICE lines.
"""
function partial_dependance(data::Array{<:Any,2},
                            features::Union{<:Integer,Vector{<:Integer}},
                            estimate_fn, grid_size::Int, return_ice::Bool)
    # Mainly inspired by Sk-learn's function.
    # https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
    #Step 1 Necessary checking
    if grid_size <=1
         raise(error("grid_size should be greater than 1, $grid_size was passed"))
    end
    #Step 2 Get_grid values
    grid, grid_values = get_grid(data[:,features], grid_size)
    #Step 3 evaluate partial_dependance
    probe = estimate_fn(data[1:2,:])
    pd = typeof(probe)(undef, size(grid,1), size(probe,2))
    # pd = []
    for grid_value_index in 1:size(grid,1)
        temp_data = copy(data)
        # (n,length(features)) .= (1,length(features))
        temp_data[:,features] .= grid[grid_value_index,:]
        predictions = estimate_fn(temp_data)
        pd[grid_value_index,:] = mean(predictions, dims=1)
    end
    pd, grid_values
end

function get_grid(data_features, grid_size)
    values = Nothing
    for feature in 1:size(data_features,2)
        unique_vals = unique(data_features[:,feature])
        value =
            if length(unique_vals) <= grid_size
                sort(unique_vals)
            else
                LinRange(minimum(data_features[:,feature]),
                 maximum(data_features[:,feature]), grid_size)
            end
        if values == Nothing
            values = [value]
        else
            push!(values, value)
        end
    end
    cartesian_prod_looped(values), values
end

function cartesian_prod(values)
    if length(values) == 1
        values[1]
    else
        cartesian_prod([
        hcat(repeat(values[1],inner=(size(values[2],1),
                                    [1 for i in 1:ndims(values[1])-1]...))
            ,repeat(values[2],outer=(size(values[1],1),
                                    [1 for i in 1:ndims(values[2])-1]...))
            )
         , values[3:end]...])
     end
 end

 function cartesian_prod_looped(values)
     while length(values) > 1
         values = [
         hcat(repeat(values[1],inner=(size(values[2],1),
                                     [1 for i in 1:ndims(values[1])-1]...))
             ,repeat(values[2],outer=(size(values[1],1),
                                     [1 for i in 1:ndims(values[2])-1]...))
             )
          , values[3:end]...]
      end
      values[1]
      # To return transposed.
      # res_values = typeof(values[1])(undef,size(transpose(values[1]))...)
      # using LinearAlgebra
      # transpose!(res_values, values[1])
      # or
      # typeof(values[1])(transpose(values[1]))
  end
