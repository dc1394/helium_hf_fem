module Helium_HF_FEM_Eigen
    include("gausslegendre.jl")
    include("helium_hf_fem_eigen_module.jl")
    include("helium_vh_fem_module.jl")
    using LinearAlgebra
    using Match
    using MKL
    using .GaussLegendre
    using .Helium_HF_FEM_Eigen_module
    using .Helium_Vh_FEM_module

    const MAXR = 50.0
    const MINR = 0.0
    const NODE_TOTAL = 5000

    function construct()
        hfem_param = Helium_HF_FEM_Eigen_module.Helium_HF_FEM_Eigen_param(NODE_TOTAL - 1, NODE_TOTAL, MAXR, MINR)
        hfem_val = Helium_HF_FEM_Eigen_module.Helium_HF_FEM_Eigen_variables(
            Symmetric(zeros(hfem_param.ELE_TOTAL, hfem_param.ELE_TOTAL)),
            Array{Float64}(undef, hfem_param.ELE_TOTAL),
            Array{Float64, 3}(undef, hfem_param.ELE_TOTAL, 2, 2),
            Array{Float64, 3}(undef, hfem_param.ELE_TOTAL, 2, 2),
            Array{Int64, 2}(undef, hfem_param.ELE_TOTAL, 2),
            Array{Float64, 2}(undef, hfem_param.ELE_TOTAL, hfem_param.ELE_TOTAL),
            Array{Float64}(undef, hfem_param.NODE_TOTAL),
            Array{Float64}(undef, hfem_param.NODE_TOTAL),
            Symmetric(zeros(hfem_param.ELE_TOTAL, hfem_param.ELE_TOTAL)),
            Array{Float64}(undef, hfem_param.NODE_TOTAL))
        
        return hfem_param, hfem_val
    end

    function make_wavefunction(iter, hfem_param, hfem_val, vh_val)
        if iter == 0
            # データの生成
            make_data!(hfem_param, hfem_val)
        
            return nothing
        end

        # 要素行列の生成
        make_element_matrix!(hfem_param, hfem_val, vh_val)

        # 全体行列を生成
        hg_tmp, ug_tmp = make_global_matrix(hfem_param, hfem_val)

        # 境界条件処理を行う
        boundary_conditions!(hfem_param, hfem_val, hg_tmp, ug_tmp)

        # 一般化固有値問題を解く
        eigenval, phi = eigen!(hfem_val.hg, hfem_val.ug)
        
        # 基底状態の固有ベクトルを取り出す
        hfem_val.phi = @view(phi[:,1])

        # 固有ベクトルの要素数を増やす
        resize!(hfem_val.phi, NODE_TOTAL)

        # 端点rcでの値を0にする
        hfem_val.phi[NODE_TOTAL] = 0.0

        # 固有ベクトル（波動関数）を規格化
        normalize!(hfem_val)

        return eigenval[1]
    end
    
    rVh(hfem_param, hfem_val, vh_val, r) = let
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
        return (vh_val.ug[khi] - vh_val.ug[klo]) / (hfem_val.node_r_glo[khi] - hfem_val.node_r_glo[klo]) * (r - hfem_val.node_r_glo[klo]) + vh_val.ug[klo]
    end

    function boundary_conditions!(hfem_param, hfem_val, hg_tmp, ug_tmp)
        hfem_val.hg = Symmetric(zeros(hfem_param.ELE_TOTAL, hfem_param.ELE_TOTAL))
        hfem_val.ug = Symmetric(zeros(hfem_param.ELE_TOTAL, hfem_param.ELE_TOTAL))
    
        @inbounds for i = 1:hfem_param.ELE_TOTAL
            for j = i - 1:i + 1
                if j != 0 && j != hfem_param.NODE_TOTAL
                    hfem_val.hg.data[j, i] = hg_tmp.data[j, i]
                    hfem_val.ug.data[j, i] = ug_tmp.data[j, i]
                end
            end
        end
    end

    get_A_matrix_element(e, le, p, q, hfem_param, hfem_val, vh_val) = let
        ed = float(e - 1)
        @match p begin
            1 =>
                @match q begin
                    1 => return 0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (2.0 * ed / 3.0 + 1.0 / 6.0) + 
                                GaussLegendre.gl_integ(r -> r * (hfem_val.node_r_ele[e, 2] - r) / hfem_val.length[e] * rVh(hfem_param, hfem_val, vh_val, r) * (hfem_val.node_r_ele[e, 2] - r) / hfem_val.length[e],
                                                       hfem_val.node_r_ele[e, 1],
                                                       hfem_val.node_r_ele[e, 2],
                                                       vh_val)
                    2 => return -0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (ed / 3.0 + 1.0 / 6.0) +
                                GaussLegendre.gl_integ(r -> r * (hfem_val.node_r_ele[e, 2] - r) / hfem_val.length[e] * rVh(hfem_param, hfem_val, vh_val, r) * (r - hfem_val.node_r_ele[e, 1]) / hfem_val.length[e],
                                                       hfem_val.node_r_ele[e, 1],
                                                       hfem_val.node_r_ele[e, 2],
                                                       vh_val)
                    _ => return 0.0
                end

            2 =>
                @match q begin
                    2 => return 0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (2.0 * ed / 3.0 + 0.5) +
                                GaussLegendre.gl_integ(r -> r * (r - hfem_val.node_r_ele[e, 1]) / hfem_val.length[e] * rVh(hfem_param, hfem_val, vh_val, r) * (r - hfem_val.node_r_ele[e, 1]) / hfem_val.length[e],
                                                            hfem_val.node_r_ele[e, 1],
                                                            hfem_val.node_r_ele[e, 2],
                                                            vh_val)
                    _ => return 0.0
                end

            _ => return 0.0
        end
    end

    get_B_matrix_element(e, le, p, q) = let
        ed = float(e - 1)
        @match p begin 
            1 =>
                @match q begin
                    1 => return le * le * le * (ed * ed / 3.0 + ed / 6.0 + 1.0 / 30.0)
                    2 => return le * le * le * (ed * ed / 6.0 + ed / 6.0 + 0.05)
                    _ => return 0.0
                end

            2 =>
                @match q begin
                    2 => return le * le * le * (ed * ed / 3.0 + ed / 2.0 + 0.2)
                    _ => return 0.0
                end
            
            _ => return 0.0
        end
    end
    
    get_totalenergy(eigenenergy, hfem_param, hfem_val, vh_val) = let
        sum = 0.0
        len = length(hfem_val.phi)
        max = len - 2

        # Simpsonの公式によって数値積分する
        @inbounds @simd for i = 1:2:max
            f0 = rVh(hfem_param, hfem_val, vh_val, hfem_val.node_r_glo[i]) * hfem_val.phi[i] * hfem_val.phi[i] * hfem_val.node_r_glo[i]
            f1 = rVh(hfem_param, hfem_val, vh_val, hfem_val.node_r_glo[i + 1]) * hfem_val.phi[i + 1] * hfem_val.phi[i + 1] * hfem_val.node_r_glo[i + 1]
            f2 = rVh(hfem_param, hfem_val, vh_val, hfem_val.node_r_glo[i + 2]) * hfem_val.phi[i + 2] * hfem_val.phi[i + 2] * hfem_val.node_r_glo[i + 2]
            
            sum += (f0 + 4.0 * f1 + f2)
        end

        return 2.0 * eigenenergy - sum * hfem_val.length[1] / 3.0
    end

    function make_data!(hfem_param, hfem_val)
        # Global節点のx座標を定義(R_MIN～R_MAX）
        dr = (hfem_param.R_MAX - hfem_param.R_MIN) / float(hfem_param.ELE_TOTAL)
        @inbounds for i = 0:hfem_param.NODE_TOTAL - 1
            # 計算領域を等分割
            hfem_val.node_r_glo[i + 1] = hfem_param.R_MIN + float(i) * dr
        end

        @inbounds for e = 1:hfem_param.ELE_TOTAL
            hfem_val.node_num_seg[e, 1] = e
            hfem_val.node_num_seg[e, 2] = e + 1
        end
            
        @inbounds for e = 1:hfem_param.ELE_TOTAL
            for i = 1:2
                hfem_val.node_r_ele[e, i] = hfem_val.node_r_glo[hfem_val.node_num_seg[e, i]]
            end
        end

        # 各線分要素の長さを計算
        @inbounds for e = 1:hfem_param.ELE_TOTAL
            hfem_val.length[e] = abs(hfem_val.node_r_ele[e, 2] - hfem_val.node_r_ele[e, 1])
        end
    end

    function make_element_matrix!(hfem_param, hfem_val, vh_val)
        # 要素行列の各成分を計算
        @inbounds for e = 1:hfem_param.ELE_TOTAL
            le = hfem_val.length[e]
            for j = 1:2
                for i = 1:j
                    hfem_val.mat_A_ele[e, i, j] = get_A_matrix_element(e, le, i, j, hfem_param, hfem_val, vh_val)
                    hfem_val.mat_B_ele[e, i, j] = get_B_matrix_element(e, le, i, j)
                end
            end
        end
    end
    
    function make_global_matrix(hfem_param, hfem_val)
        hg_tmp = Symmetric(zeros(hfem_param.NODE_TOTAL, hfem_param.NODE_TOTAL))
        ug_tmp = Symmetric(zeros(hfem_param.NODE_TOTAL, hfem_param.NODE_TOTAL))

        @inbounds for e = 1:hfem_param.ELE_TOTAL
            for j = 1:2
                for i = 1:j
                    hg_tmp.data[hfem_val.node_num_seg[e, i], hfem_val.node_num_seg[e, j]] += hfem_val.mat_A_ele[e, i, j]
                    ug_tmp.data[hfem_val.node_num_seg[e, i], hfem_val.node_num_seg[e, j]] += hfem_val.mat_B_ele[e, i, j]
                end
            end
        end

        return hg_tmp, ug_tmp
    end

    function normalize!(hfem_val)
        sum = 0.0
        len = length(hfem_val.phi)
        max = len - 2

        # Simpsonの公式によって数値積分する
        @inbounds @simd for i = 1:2:max
            f0 = hfem_val.phi[i] * hfem_val.phi[i] * hfem_val.node_r_glo[i] * hfem_val.node_r_glo[i]
            f1 = hfem_val.phi[i + 1] * hfem_val.phi[i + 1] * hfem_val.node_r_glo[i + 1] * hfem_val.node_r_glo[i + 1]
            f2 = hfem_val.phi[i + 2] * hfem_val.phi[i + 2] * hfem_val.node_r_glo[i + 2] * hfem_val.node_r_glo[i + 2]
            
            sum += (f0 + 4.0 * f1 + f2)
        end
        
        hfem_val.phi = map(x -> abs(x * sqrt(sum * hfem_val.length[1] / 3.0)), hfem_val.phi)
    end
end