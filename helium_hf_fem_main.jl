include("helium_hf_fem_scf.jl")
using Printf
using .Helium_HF_FEM_SCF

function main()
    Helium_HF_FEM_SCF.scfloop()
end

main()