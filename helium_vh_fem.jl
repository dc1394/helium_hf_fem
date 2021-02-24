module Helium_Vh_FEM
    include("gausslegendre.jl")
    include("helium_vh_fem_module.jl")
    using LinearAlgebra
    using Match
    using Printf
    using .GaussLegendre
    using .Helium_Vh_FEM_module

    function construct(hfem_param)
        vh_param = Helium_Vh_FEM_module.Helium_Vh_FEM_param(100)
        vh_val = Helium_Vh_FEM_module.Helium_Vh_FEM_variables(
            Array{Float64}(undef, hfem_param.ELE_TOTAL, 2, 2),
            SymTridiagonal(Array{Float64}(undef, hfem_param.NODE_TOTAL), Array{Float64}(undef, hfem_param.NODE_TOTAL - 1)),
            Array{Float64}(undef, hfem_param.NODE_TOTAL),
            Array{Float64}(undef, hfem_param.ELE_TOTAL, 2),
            Array{Float64}(undef, hfem_param.NODE_TOTAL),
            Array{Float64}(undef, vh_param.INTEGTABLENUM),
            Array{Float64}(undef, vh_param.INTEGTABLENUM))
        
        vh_val.x, vh_val.w = GaussLegendre.gausslegendre(vh_param.INTEGTABLENUM)

        return vh_param, vh_val
    end

    phi(hfem_param, hfem_val, r) = let
        klo = 1
        max = hfem_param.NODE_TOTAL
        khi = max

        # 表の中の正しい位置を二分探索で求める
        @inbounds while khi - klo > 1
            k = (khi + klo) >> 1

            if hfem_val.node_r_glo[k] > r
                khi = k        
            else 
                klo = k
            end
        end

        # yvec_[i] = f(xvec_[i]), yvec_[i + 1] = f(xvec_[i + 1])の二点を通る直線を代入
        return (hfem_val.phi[khi] - hfem_val.phi[klo]) / (hfem_val.node_r_glo[khi] - hfem_val.node_r_glo[klo]) * (r - hfem_val.node_r_glo[klo]) + hfem_val.phi[klo]
    end

    function solvepoisson(iter, hfem_param, hfem_val, vh_val)
        @match iter begin
            # 要素行列とLocal節点ベクトルを生成
            0 => make_element_matrix_and_vector_first(hfem_param, hfem_val, vh_val)
        
            # 要素行列とLocal節点ベクトルを生成
            _ => make_element_matrix_and_vector(hfem_param, hfem_val, vh_val)
        end

        # 全体行列と全体ベクトルを生成
        tmp_dv, tmp_ev = make_global_matrix_and_vector(hfem_param, hfem_val, vh_val)

        # 境界条件処理
        boundary_conditions(vh_val, hfem_param, tmp_dv, tmp_ev)

        # 連立方程式を解く
        vh_val.ug = vh_val.mat_A_glo \ vh_val.vec_b_glo
    end
    
    function boundary_conditions(vh_val, hfem_param, tmp_dv, tmp_ev)
        a = 0.0
        tmp_dv[1] = 1.0
        vh_val.vec_b_glo[1] = a
        vh_val.vec_b_glo[2] -= a * tmp_ev[1]
        tmp_ev[1] = 0.0
    
        b = 2.0
        tmp_dv[hfem_param.NODE_TOTAL] = 1.0
        vh_val.vec_b_glo[hfem_param.NODE_TOTAL] = b
        vh_val.vec_b_glo[hfem_param.NODE_TOTAL - 1] -= b * tmp_ev[hfem_param.NODE_TOTAL - 1]
        tmp_ev[hfem_param.NODE_TOTAL - 1] = 0.0

        vh_val.mat_A_glo = SymTridiagonal(tmp_dv, tmp_ev)
    end

    function make_element_matrix_and_vector(hfem_param, hfem_val, vh_val)
        # 要素行列とLocal節点ベクトルの各成分を計算
        for e = 1:hfem_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    vh_val.mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / hfem_val.length[e]
                end

                vh_val.vec_b_ele[e, i] =
                    @match i begin
                        1 => -GaussLegendre.gl_integ(r -> -r * rho(hfem_param, hfem_val, r) * (hfem_val.node_r_ele[e, 2] - r) / hfem_val.length[e],
                                                     hfem_val.node_r_ele[e, 1],
                                                     hfem_val.node_r_ele[e, 2],
                                                     vh_val)
                        
                        2 => -GaussLegendre.gl_integ(r -> -r * rho(hfem_param, hfem_val, r) * (r - hfem_val.node_r_ele[e, 1]) / hfem_val.length[e],
                                                     hfem_val.node_r_ele[e, 1],
                                                     hfem_val.node_r_ele[e, 2],
                                                     vh_val)
                    
                        _ => 0.0
                    end
            end
        end
    end

    function make_element_matrix_and_vector_first(hfem_param, hfem_val, vh_val)
        # 要素行列とLocal節点ベクトルの各成分を計算
        for e = 1:hfem_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    vh_val.mat_A_ele[e, i, j] = (-1) ^ i * (-1) ^ j / hfem_val.length[e]
                end

                vh_val.vec_b_ele[e, i] =
                    @match i begin
                        1 => -GaussLegendre.gl_integ(r -> -4.0 * r * exp(-2.0 * r) * (hfem_val.node_r_ele[e, 2] - r) / hfem_val.length[e],
                                                     hfem_val.node_r_ele[e, 1],
                                                     hfem_val.node_r_ele[e, 2],
                                                     vh_val)
                        
                        2 => -GaussLegendre.gl_integ(r -> -4.0 * r * exp(-2.0 * r) * (r - hfem_val.node_r_ele[e, 1]) / hfem_val.length[e],
                                                     hfem_val.node_r_ele[e, 1],
                                                     hfem_val.node_r_ele[e, 2],
                                                     vh_val)
                    
                        _ => 0.0
                    end
            end
        end
    end

    function make_global_matrix_and_vector(hfem_param, hfem_val, vh_val)
        tmp_dv = zeros(hfem_param.NODE_TOTAL)
        tmp_ev = zeros(hfem_param.NODE_TOTAL - 1)

        vh_val.vec_b_glo = zeros(hfem_param.NODE_TOTAL)

        # 全体行列と全体ベクトルを生成
        for e = 1:hfem_param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    if hfem_val.node_num_seg[e, i] == hfem_val.node_num_seg[e, j]
                        tmp_dv[hfem_val.node_num_seg[e, i]] += vh_val.mat_A_ele[e, i, j]
                    elseif hfem_val.node_num_seg[e, i] + 1 == hfem_val.node_num_seg[e, j]
                        tmp_ev[hfem_val.node_num_seg[e, i]] += vh_val.mat_A_ele[e, i, j]
                    end
                end
                
                vh_val.vec_b_glo[hfem_val.node_num_seg[e, i]] += vh_val.vec_b_ele[e, i]
            end
        end

        return tmp_dv, tmp_ev
    end

    rho(hfem_param, hfem_val, r) = let
        tmp = phi(hfem_param, hfem_val, r)
        return tmp * tmp
    end
end
