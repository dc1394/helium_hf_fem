include("helium_hf_scf.jl")
using Printf
using .Helium_HF_SCF

function main()
    Helium_HF_SCF.scfloop()
end

main()