module Helium_HF_Eigen
    include("helium_hf_eigen_module.jl")
    using LinearAlgebra
    using Match
    using Printf
    using .Helium_HF_Eigen_module

    const NODE_TOTAL = 3000

    function construct()
        param = Helium_HF_Eigen_module.Helium_HF_Eigen_param(NODE_TOTAL - 1, NODE_TOTAL, 30.0, 0.0)
        val = Helium_HF_Eigen_module.Helium_HF_Eigen_variables(
            Symmetric(zeros(param.ELE_TOTAL, param.ELE_TOTAL)),
            Array{Float64}(undef, param.ELE_TOTAL),
            Array{Float64, 3}(undef, param.ELE_TOTAL, 2, 2),
            Array{Float64, 3}(undef, param.ELE_TOTAL, 2, 2),
            Array{Int64, 2}(undef, param.ELE_TOTAL, 2),
            Array{Float64, 2}(undef, param.ELE_TOTAL, param.ELE_TOTAL),
            Array{Float64}(undef, param.NODE_TOTAL),
            Array{Float64}(undef, param.NODE_TOTAL),
            Symmetric(zeros(param.ELE_TOTAL, param.ELE_TOTAL)),
            Array{Float64}(undef, param.NODE_TOTAL))
        
        return param, val
    end

    function construct_next(val, vh)
        val.vh = vh
    end

    function make_wavefunction(scfcount, param, val)
        # データの生成
        make_data!(param, val)

        # 要素行列の生成
        make_element_matrix!(param, val)

        if scfcount == 1
            return nothing
        end

        # 全体行列を生成
        hg_tmp, ug_tmp = make_global_matrix(param, val)

        # 境界条件処理を行う
        boundary_conditions!(param, val, hg_tmp, ug_tmp)

        # 一般化固有値問題を解く
        eigenval, phi = eigen(val.hg, val.ug)
        
        # 基底状態の固有ベクトルを取り出す
        val.phi = @view(phi[:,1])

        # 固有ベクトルの要素数を増やす
        resize!(val.phi, NODE_TOTAL)

        # 固有ベクトル（波動関数）を規格化
        normalize!(val)

        @printf("固有エネルギー = %.14f\n", eigenval[1])

        return eigenval[1]
    end

    rho(param, val, r) = let
        klo = 1;
        max = param.NODE_TOTAL;
        khi = max;

        # 表の中の正しい位置を二分探索で求める
        @inbounds while khi - klo > 1
            k = (khi + klo) >> 1

            if val.node_r_glo[k] > r
                khi = k        
            else 
                klo = k
            end
        end

        # yvec_[i] = f(xvec_[i]), yvec_[i + 1] = f(xvec_[i + 1])の二点を通る直線を代入
        tmp = (val.phi[khi] - val.phi[klo]) / (val.node_r_glo[khi] - val.node_r_glo[klo]) * (r - val.node_r_glo[klo]) + val.phi[klo]
        return tmp * tmp
    end

    function boundary_conditions!(param, val, hg_tmp, ug_tmp)
        @inbounds for i = 1:param.ELE_TOTAL
            for j = i - 1:i + 1
                if j != 0 && j != param.NODE_TOTAL
                    # 左辺の全体行列のN行とN列を削る
                    val.hg.data[j, i] = hg_tmp.data[j, i]

                    # 右辺の全体行列のN行とN列を削る    
                    val.ug.data[j, i] = ug_tmp.data[j, i]
                end
            end
        end
    end

    get_A_matrix_element(e, le, p, q) = let
        ed = float(e - 1)
        @match p begin
            1 =>
                @match q begin
                    1 => return  0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (2.0 * ed / 3.0 + 1.0 / 6.0)
                    2 => return -0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (ed / 3.0 + 1.0 / 6.0)
                    _ => return 0.0
                end

            2 =>
                @match q begin
                    1 => return -0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (ed / 3.0 + 1.0 / 6.0)
                    2 => return 0.5 * le * (ed * ed + ed + 1.0 / 3.0) - le * le * (2.0 * ed / 3.0 + 0.5)
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
                    1 => return le * le * le * (ed * ed / 6.0 + ed / 6.0 + 0.05)
                    2 => return le * le * le * (ed * ed / 3.0 + ed / 2.0 + 0.2)
                    _ => return 0.0
                end
            
            _ => return 0.0
        end
    end

    function make_data!(param, val)
        # Global節点のx座標を定義(R_MIN～R_MAX）
        dr = (param.R_MAX - param.R_MIN) / float(param.ELE_TOTAL)
        @inbounds for i = 0:param.NODE_TOTAL - 1
            # 計算領域を等分割
            val.node_r_glo[i + 1] = param.R_MIN + float(i) * dr
        end

        @inbounds for e = 1:param.ELE_TOTAL
            val.node_num_seg[e, 1] = e
            val.node_num_seg[e, 2] = e + 1
        end
            
        @inbounds for e = 1:param.ELE_TOTAL
            for i = 1:2
                val.node_r_ele[e, i] = val.node_r_glo[val.node_num_seg[e, i]]
            end
        end
    end

    function make_element_matrix!(param, val)
        # 各線分要素の長さを計算
        @inbounds for e = 1:param.ELE_TOTAL
            val.length[e] = abs(val.node_r_ele[e, 2] - val.node_r_ele[e, 1])
        end

        # 要素行列の各成分を計算
        @inbounds for e = 1:param.ELE_TOTAL
            le = val.length[e]
            for i = 1:2
                for j = 1:2
                    val.mat_A_ele[e, i, j] = get_A_matrix_element(e, le, i, j)
                    val.mat_B_ele[e, i, j] = get_B_matrix_element(e, le, i, j)
                end
            end
        end
    end
    
    function make_global_matrix(param, val)
        hg_tmp = Symmetric(zeros(param.NODE_TOTAL, param.NODE_TOTAL))
        ug_tmp = Symmetric(zeros(param.NODE_TOTAL, param.NODE_TOTAL))

        @inbounds for e = 1:param.ELE_TOTAL
            for i = 1:2
                for j = 1:2
                    hg_tmp.data[val.node_num_seg[e, i], val.node_num_seg[e, j]] += val.mat_A_ele[e, i, j]
                    ug_tmp.data[val.node_num_seg[e, i], val.node_num_seg[e, j]] += val.mat_B_ele[e, i, j]
                end
            end
        end

        return hg_tmp, ug_tmp
    end

    function normalize!(val)
        sum = 0.0
        len = length(val.phi)
        max = len - 2

        # Simpsonの公式によって数値積分する
        @inbounds @simd for i = 1:2:max
            f0 = val.phi[i] * val.phi[i] * val.node_r_glo[i] * val.node_r_glo[i]
            f1 = val.phi[i + 1] * val.phi[i + 1] * val.node_r_glo[i + 1] * val.node_r_glo[i + 1]
            f2 = val.phi[i + 2] * val.phi[i + 2] * val.node_r_glo[i + 2] * val.node_r_glo[i + 2]
            
            sum += (f0 + 4.0 * f1 + f2)
        end
        
        val.phi .*= (-1.0 / sqrt(sum * val.length[1] / 3.0))
    end
end