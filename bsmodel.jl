using BridgeStan
const BS = BridgeStan

function bsmodel_dim(bsmodel)
    return convert(Int64, BS.param_unc_num(bsmodel))::Int64
end

function bsmodel_ld(bsmodel)
    function bsm_ld(x)
        ld = try
            BS.log_density(bsmodel, x)
        catch e
            -Inf
        end
        return ld
    end
    return bsm_ld
end

function bsmodel_ldg(bsmodel)
    function bsm_ldg(x)
        ld, g = try
            BS.log_density_gradient(bsmodel, x)
        catch e
            D = BS.param_unc_num(bsmodel)
            -Inf, zeros(D)
        end
        return ld, g
    end
    return bsm_ldg
end

function bsmodel_ldgh(bsmodel)
    function bsm_ldgh(x)
        ld, g, h = try
            BS.log_density_hessian(bsmodel, x)
        catch e
            D = bsmodel_dim(bsmodel)
            -Inf, zeros(D), zeros(D, D)
        end
        return ld, g, h
    end
    return bsm_ldgh
end
