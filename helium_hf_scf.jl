module Helium_HF_SCF
    include("helium_hf_eigen.jl")
    include("helium_vh.jl")
    using Printf
    using .Helium_HF_Eigen
    using .Helium_Vh

    function scfloop()
        hfem_param, hfem_val = Helium_HF_Eigen.construct()
        
        Helium_HF_Eigen.make_wavefunction(1, hfem_param, hfem_val)
        vh_param, vh_val = Helium_Vh.construct(hfem_param)
        Helium_Vh.do_run(1, hfem_param, hfem_val, vh_val)

        Helium_Vh.save_result(hfem_val, vh_param, vh_val)
    end
end

