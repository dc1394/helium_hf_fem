module GaussLegendre
    using FastGaussQuadrature

    function constract(integtablenum)
        return gausslegendre(integtablenum)
    end

    function gl_integ(f, x1, x2, vh_val)
        xm = 0.5 * (x1 + x2)
        xr = 0.5 * (x2 - x1)
        
        s = sum(i -> vh_val.w[i] * f(xm + xr * vh_val.x[i]), eachindex(vh_val.x))
        
        return s * xr
    end
end