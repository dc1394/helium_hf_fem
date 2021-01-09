module Helium_HF_FEM_Eigen_module
    using LinearAlgebra

    struct Helium_HF_FEM_Eigen_param
        ELE_TOTAL::Int64
        NODE_TOTAL::Int64
        R_MAX::Float64
        R_MIN::Float64
    end

    mutable struct Helium_HF_FEM_Eigen_variables
        hg::Symmetric{Float64,Array{Float64,2}}
        length::Array{Float64, 1}
        mat_A_ele::Array{Float64, 3}
        mat_B_ele::Array{Float64, 3}
        node_num_seg::Array{Int64, 2}
        node_r_ele::Array{Float64, 2}
        node_r_glo::Array{Float64, 1}
        phi::Array{Float64, 1}
        ug::Symmetric{Float64,Array{Float64,2}}
        vh::Array{Float64, 1}
    end
end