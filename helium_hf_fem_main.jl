include("helium_hf_fem_scf.jl")
using Printf
using .Helium_HF_FEM_SCF

const RESULT_FILENAME = "result.csv"

function main()
    res = Helium_HF_FEM_SCF.scfloop()
    if res != nothing
        hfem_param, hfem_val, energy = res
        Helium_HF_FEM_SCF.save_result(hfem_param, hfem_val, RESULT_FILENAME)
        @printf "SCF計算が収束しました: energy = %.14f (Hartree)、計算結果を%sに書き込みました\n" energy RESULT_FILENAME
    else
        @printf "SCF計算が収束しませんでした\n"
    end
end

@time main()