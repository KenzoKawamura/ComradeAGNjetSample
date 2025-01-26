using Pkg
Pkg.activate(".")
using Comrade
using Pyehtim
using StableRNGs
using Distributions
using VLBIImagePriors
using Optimization
using OptimizationBBO
using DisplayAs
import CairoMakie as CM
using Pigeons
using JSON
using StructArrays

rng = StableRNG(42)
obs = ehtim.obsdata.load_uvfits("hogehoge.uvf")
dlcamp, dcphase = extract_table(obs, LogClosureAmplitudes(;snrcut=3.0), ClosurePhases(;snrcut=3.0))
dvis = extract_table(obs, Visibilities(;snrcut=3.0))

beam = beamsize(dvis)
@info "beamsize: $(rad2μas(beam)) μas" 

max_abs_value = maximum(abs.(dvis.measurement))
@info "max: $(max_abs_value) Jy"

fwhmfac = 2*sqrt(2*log(2))

fovx = μas2rad(20_000.0)
fovy = μas2rad(20_000.0)
npix = 96
gauss_number = 1

function sky(θ, p)
    ftot = θ.ftot
    fluxes = θ.fg
    angles = θ.θg
    width = θ.wg
    distance = θ.dg
    core_width = θ.cw
    n = gauss_number
    
    core = modify(Gaussian(), Stretch(core_width / fwhmfac, core_width / fwhmfac))
    img = core

    if n > 0
        for i in 1:n
            x = distance[i] * sin(angles[i])
            y = distance[i] * cos(angles[i])
            img += fluxes[i]*shifted(stretched(Gaussian(), width[i]/fwhmfac, width[i]/fwhmfac), x, y)
        end
    end
    return img * ftot
end

function generate_prior(n::Int)
    return (
        ftot = truncated(Normal(max_abs_value, max_abs_value*0.1); lower=0.0), # total Jy
        cw = truncated(Normal(μas2rad(100.0), μas2rad(10.0)); lower=0.0), # width of core component
        fg = [Uniform(0.0, 1.0) for _ in 1:n], # flux ratio of each Gaussian component compare to core component
        θg = [Uniform(0, 2π) for _ in 1:n], # position angle from core component
        wg = [truncated(Normal(μas2rad(100.0), μas2rad(10.0)); lower=0.0) for _ in 1:n], # width of each Gaussian component
        dg = [truncated(Normal(0, fovx/10); lower=0.0) for _ in 1:n], # distance from core component of each Gaussian component
    )
end

prior = generate_prior(gauss_number)

g = imagepixels(fovx, fovy, npix, npix)
skym = SkyModel(sky, prior, g)
post = VLBIPosterior(skym, dvis, dlcamp, dcphase)
cpost = ascube(post)
p = prior_sample(rng, post)
@time begin
    xopt, sol = comrade_opt(post, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=50_000);
end
img = intensitymap(skymodel(post, xopt), g)
fig = imageviz(img, colormap=:afmhot, size=(500, 400));

save_fits("hogehoge.fits", img)

# chi2 check
c2_vis, c2_cphase, c2_clamp = chi2(post, xopt)
c2_cphase /= length(dcphase)
c2_clamp /= length(dlcamp)
@info "MAP chi2 check"
@info "CP  Chi2 : $(c2_cphase)"    
@info "LCA Chi2 : $(c2_clamp)"

@info "MAP result: $(xopt.sky)"

# MCMC
@time begin
    pt = pigeons(target=cpost, explorer=SliceSampler(), record=[traces, round_trip, log_sum_ratio], n_chains=16, n_rounds=8, multithreaded=true)
end
chain = sample_array(cpost, pt)

imgs = intensitymap.(skymodel.(Ref(post), sample(chain, 100)), Ref(g))
meanimg = mean(imgs)
fig = imageviz(meanimg, colormap=:afmhot)
save_fits("mtest.fits", meanimg)

# chi2 check
c2_vis, c2_cphase, c2_clamp = chi2(post, chain[end])
c2_cphase /= length(dcphase)
c2_clamp /= length(dlcamp)
@info "MCMC Chi2 check"
@info "CP  Chi2 : $(c2_cphase)"    
@info "LCA Chi2 : $(c2_clamp)"

nchain = length(chain.sky)

sp = sortperm.(chain.sky.dg)
n = gauss_number
new_chain = StructArray((;
    ftot = chain.sky.ftot,
    cw   = rad2μas(1) * chain.sky.cw,
    fg = [
        chain.sky.fg[i][sp[i]][1:n] 
        for i in 1:nchain
    ],
    θg = [
        rad2deg(1) * chain.sky.θg[i][sp[i]][1:n]
        for i in 1:nchain
    ],
    dg = [
        rad2μas(1) * chain.sky.dg[i][sp[i]][1:n]
        for i in 1:nchain
    ],
    wg = [
        rad2μas(1) * chain.sky.wg[i][sp[i]][1:n]
        for i in 1:nchain
    ]
))

samples = new_chain
flds = keys(first(samples))

function stat_field(stat::Function, vals)
    first_val = first(vals)
    if first_val isa Number
        return stat(vals)
    elseif first_val isa AbstractVector
        mat = reduce(hcat, vals)
        return vec(stat(mat, dims=2))
    else
        error("Invalid tyoe: $(typeof(first_val))")
    end
end

mean_tuple = NamedTuple{flds}(stat_field(mean, getfield.(samples, f)) for f in flds)
std_tuple = NamedTuple{flds}(stat_field(std, getfield.(samples, f)) for f in flds)

json_str_mean = JSON.json(Dict(pairs(mean_tuple)))
json_str_std = JSON.json(Dict(pairs(std_tuple)))

open("meanhoge.json", "w") do f
    write(f, json_str_mean)
end
open("stdhoge.json", "w") do f
    write(f, json_str_std)
end

function format_stat(m, s)
    if m isa Number
        return "$(m) ± $(s)"
    elseif m isa AbstractVector
        strs = ["$(m[i]) ± $(s[i])" for i in eachindex(m)]
        return "[\n  " * join(strs, ",\n  ") * "\n]"
    else
        return string(m)
    end
end

function print_stat_table(mean_tuple, std_tuple)
    println("parameter          value (mean ± std)")
    println("--------------------------------------")
    for fld in keys(mean_tuple)
        m = getfield(mean_tuple, fld)
        s = getfield(std_tuple, fld)
        formatted = format_stat(m, s)
        println("$(fld): $formatted")
    end
end

@info "MCMC result"
print_stat_table(mean_tuple, std_tuple)