module Helium_HF_FEM_SCF
    include("helium_hf_fem_eigen.jl")
    include("helium_vh_fem.jl")
    using Printf
    using .Helium_HF_FEM_Eigen
    using .Helium_Vh_FEM

    const SCFMAX = 1000
    const THRESHOLD = 1.0E-4

    function scfloop()
        hfem_param, hfem_val = Helium_HF_FEM_Eigen.construct()
        
        Helium_HF_FEM_Eigen.make_wavefunction(0, hfem_param, hfem_val, nothing)
        vh_param, vh_val = Helium_Vh_FEM.construct(hfem_param)
        Helium_Vh_FEM.do_run(0, hfem_param, hfem_val, vh_val)

        for scfloop in 1:SCFMAX
            eigenenergy = Helium_HF_FEM_Eigen.make_wavefunction(scfloop, hfem_param, hfem_val, vh_val)
            totalenergy = Helium_HF_FEM_Eigen.get_totalenergy(eigenenergy, hfem_param, hfem_val, vh_val)
            
            @printf("Iteration # %2d: HF eigenvalue = %.14f, energy = %.14f\n", scfloop, eigenenergy, totalenergy)
            abs(totalenergy - hfem_val.totalenergy) <= THRESHOLD && break
            hfem_val.totalenergy = totalenergy
            
            Helium_Vh_FEM.do_run(scfloop, hfem_param, hfem_val, vh_val)
        end

        # 規格化
        hfem_val.phi ./= sqrt(4.0 * pi)
        save_result(hfem_param, hfem_val)
    end

    save_result(hfem_param, hfem_val) = let
        open("result.csv", "w" ) do fp
            for i = 1:length(hfem_val.node_r_glo)
                r = hfem_val.node_r_glo[i]
                println(fp, @sprintf "%.14f, %.14f, %.14f" (r) (hfem_val.phi[i]) (Helium_HF_FEM_Eigen.rho(hfem_param, hfem_val, r)))
            end
        end
    end
end

