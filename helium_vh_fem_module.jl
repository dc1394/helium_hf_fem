module Helium_Vh_FEM_module
    using LinearAlgebra
    
    struct Helium_Vh_FEM_param
        INTEGTABLENUM::Int64
    end

    mutable struct Helium_Vh_FEM_variables
        mat_A_ele::Array{Float64, 3}
        mat_A_glo::SymTridiagonal{Float64,Array{Float64,1}}
        ug::Array{Float64, 1}
        vec_b_ele::Array{Float64, 2}
        vec_b_glo::Array{Float64, 1}
        w::Array{Float64, 1}
        x::Array{Float64, 1}
    end
end