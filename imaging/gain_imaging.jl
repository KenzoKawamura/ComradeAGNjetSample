using Pkg
Pkg.activate(".")
using Comrade
using Pyehtim
using LinearAlgebra
using StableRNGs
using VLBIImagePriors
using Distributions
using Enzyme
using Optimization
using OptimizationOptimisers
using Plots
using DisplayAs
import CairoMakie as CM
using AdvancedHMC
using Measurements

noise = 0.02
npix = 96
fovx = μas2rad(15_000)
fovy = μas2rad(15_000)
impix = 96
x0 = μas2rad(-5_000)
y0 = μas2rad(-5_000)

rng = StableRNG(12)

dataname = "hogehoge.uvf"
comment = ""
filename = "data/" * dataname  * "_n" * string(noise) * comment
obs = ehtim.obsdata.load_uvfits("your/directory/" * dataname)
obs = scan_average(obs).add_fractional_noise(noise)
dlcamp, dcphase = extract_table(obs, LogClosureAmplitudes(), ClosurePhases())
dvis = extract_table(obs, Visibilities())

max_abs_value = maximum(abs.(dvis.measurement))
@info "max: $(max_abs_value) Jy"

ftot = max_abs_value * 1.1

beam = beamsize(dvis)
@info "beam = $(rad2μas(beam)) μas"

rad_per_pix = fovx / impix
@info "pixel size = $(rad2μas(rad_per_pix)) μas"
if beam / 2 < rad_per_pix
    @warn "too large pixelsizes!"
elseif beam / 4 > rad_per_pix
    @warn "too small pixelsizes!"
end

function sky(θ, metadata)
    (;c, σimg, fb) = θ
    (;ftot, gauss, bkgd) = metadata
    # Apply the GMRF fluctuations to the image
    mimg = (gauss .+ fb * bkgd) ./ (1 + fb)
    rast = apply_fluctuations(CenteredLR(), mimg, σimg.*c.params)
    pimg = parent(rast)
    # @. pimg = (ftot*(1-fg))*pimg
    @. pimg = (ftot)*pimg
    m = ContinuousImage(rast, BSplinePulse{3}())
    return shifted(m, x0, y0)
end

grid = ComradeBase.imagepixels(fovx, fovy, npix, npix, x0, y0)

fwhmfac = 2*sqrt(2*log(2))

#   this is a compact gausiann at the phase center of the image
gauss_m = modify(Gaussian(), Stretch(beam/2 / fwhmfac, beam/2 / fwhmfac))
gauss = intensitymap(gauss_m, grid)
gauss ./= flux(gauss)

#   this is a background image extended over the whole image
bkgd_m = modify(Gaussian(), Stretch(fovx / 2, fovy / 2), Renormalize(1.0))
bkgd = intensitymap(bkgd_m, grid)
bkgd ./= flux(bkgd)

skymeta = (;ftot = ftot, gauss = gauss, bkgd = bkgd) # mimg = mimg./flux(mimg)

cprior = corr_image_prior(grid, dvis)

prior = (
    c = cprior, # image prior
    σimg = truncated(Normal(0.0, 0.1); lower=0.01), # std of MRF; correlation length
    fb=truncated(Normal(0.0, 1.0); lower=1e-2,)
)

skym = SkyModel(sky, prior, grid; metadata=skymeta)

"""
x.lgμ: mean of log gain amp
x.lgσ: std of log gain amp
x.lgz: random noise
"""
# single stokes gain function
# equal to self-calibration
G = SingleStokesGain() do x
    lg = x.lgμ + x.lgσ*x.lgz # amp
    # lg = 0 + x.lgσ*x.lgz
    gp = x.gp # phase
    return exp(lg + 1im*gp) # complex gain
end

# gain prior
"""
assuming that each scan/track follows the IID
"""
intpr = (
    lgμ = ArrayPrior(
        IIDSitePrior(TrackSeg(), Normal(0.0, 0.05));
        ZLC = IIDSitePrior(TrackSeg(), Normal(0.0, 4.0)),
        BDR = IIDSitePrior(TrackSeg(), Normal(0.0, 4.0)),
        SVT = IIDSitePrior(TrackSeg(), Normal(0.0, 4.0))
    ),
    lgσ = ArrayPrior(
        IIDSitePrior(TrackSeg(), Exponential(0.05));
        ZLC = IIDSitePrior(TrackSeg(), Exponential(1.0)),
        BDR = IIDSitePrior(TrackSeg(), Exponential(1.0)),
        SVT = IIDSitePrior(TrackSeg(), Exponential(1.0))
    ),
    lgz = ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.05));
        ZLC = IIDSitePrior(ScanSeg(), Normal(0.0, 4.0)),
        BDR = IIDSitePrior(ScanSeg(), Normal(0.0, 4.0)),
        SVT = IIDSitePrior(ScanSeg(), Normal(0.0, 4.0))
        ),
    gp = ArrayPrior(
        IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2)));
        ZLC = IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^4))),
        BDR = IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^4))),
        SVT = IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^4)))
    ),
)

intmodel = InstrumentModel(G, intpr)


post = VLBIPosterior(skym, intmodel, dvis, dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))

tpost = asflat(post) # transform to real dimension
ndim = dimension(tpost)
@info "posterior dimension = $(ndim)"

@time begin
    xopt, sol = comrade_opt(post, Optimisers.Adam(), initial_params=prior_sample(rng, post), maxiters=20_000, g_tol=1e-1)
end

savefig(residual(post, xopt), filename * ".residual.png")

g = imagepixels(fovx, fovy, impix, impix)
img = intensitymap(skymodel(post, xopt), g)
imageviz(img, size=(500, 400))

save_fits(filename * ".MAPimage.fits",img)

# chi2 check
c2_vis = chi2(post, xopt)[1]
n_params = length(xopt)
ν = length(dvis) - n_params
c2_vis /= ν
@info "MAP image was saved as $(filename).MAPimage.fits"
@info "visibility Chi2 2: $(c2_vis)"

intopt = instrumentmodel(post, xopt)
gt = Comrade.caltable(angle.(intopt))
savefig(plot(gt, layout=(4,4), size=(600,500)), filename * ".MAP_phase_caltable.png")

gt = Comrade.caltable(abs.(intopt))
savefig(plot(gt, layout=(4,4), size=(600,500)), filename * ".MAP_amp_caltable.png")

function hmc()
    @time begin
        chain = sample(rng, post, NUTS(0.56), 5_000; n_adapts=3_000, progress=true, initial_params=xopt, chains = 8)
    end
    chain = chain[3001:end]
    mchain = Comrade.rmap(mean, chain)
    schain = Comrade.rmap(std, chain)

    gmeas = instrumentmodel(post, (;instrument= map((x,y)->Measurements.measurement.(x,y), mchain.instrument, schain.instrument)))
    ctable_am = caltable(abs.(gmeas))
    ctable_ph = caltable(angle.(gmeas))

    savefig(plot(ctable_ph, layout=(4,3), size=(600,500)), filename * ".HMC_phase_caltable.png")
    savefig(plot(ctable_am, layout=(4,3), size=(600,500)), filename * ".HMC_amplitude_caltable.png")

    samples = skymodel.(Ref(post), chain[begin:5:end])
    imgs = intensitymap.(samples, Ref(g))

    mimg = mean(imgs)
    simg = std(imgs)./mimg
    fig = CM.Figure(;resolution=(700, 700));
    axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
    CM.image!(axs[1,1], mimg, colormap=:afmhot); axs[1, 1].title="Mean"
    CM.image!(axs[1,2], simg./mimg, colorrange=(0.0, 2.0), colormap=:afmhot);axs[1,2].title = "Std"
    CM.image!(axs[2,1], imgs[1],   colormap=:afmhot);
    CM.image!(axs[2,2], imgs[end], colormap=:afmhot);
    CM.hidedecorations!.(axs)
    save_fits(filename * ".meanimage.fits", mimg)
    save_fits(filename * ".stdimage.fits", simg)
    CM.save(filename * ".HMC_summary.png", fig)

    savefig(residual(post, mchain), filename * ".HMCresidual.png")

    # chi2 check
    c2_vis = chi2(post, chain[end])[1]
    c2_vis /= ν
    @info "HMC mean image was saved as $(filename).meanimage.fits"
    @info "visibility Chi2 2: $(c2_vis)"
end

println("Go to HMC?(y/n)")
user_input = readline()

if lowercase(user_input) == "y"
    hmc()
end
