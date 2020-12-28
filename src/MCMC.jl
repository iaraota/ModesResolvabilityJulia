# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using Quantities, FrequencyTransforms, PhysConstants

using PyCall, PyPlot, Random, LinearAlgebra, LsqFit, Optim, DelimitedFiles, Dierckx, FFTW, HDF5, DSP
using Distributed
@pyimport emcee
@pyimport scipy
@pyimport corner
function log_prob(x, μ, Σ)
    diff = x .- μ
    return -.5*dot(diff, Σ\diff)
end

function quickstart()
    ndim = 5
    Random.seed!(42)
    means = rand(ndim)
    cov = 0.5 .- reshape(rand(ndim^2), (ndim,ndim))
    cov = UpperTriangular(cov)
    cov += transpose(cov) .- Diagonal(cov)
    cov = dot(cov, cov)

    nwalkers = 32
    p0 = rand(Float64, (nwalkers,ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    sampler.run_mcmc(state, 10000)

    samples = sampler.get_chain(flat=true)
    return samples
end

function log_likelihood(m, b, log_f, x, y, yerr)
    model = m .* x .+ b
    sigma2 = yerr .^ 2 + model .^ 2 .* exp(2 .* log_f)
    return -0.5 * sum((y .- model) .^ 2 ./ sigma2 .+ log.(sigma2))
end

function log_prior(m, b, log_f)
    if -5.0 < m < 0.5 && 0.0 < b < 10.0 && -10.0 < log_f < 1.0
        return 0.0
    else
        return -Inf
    end
end

function log_probability(m, b, log_f, x, y, yerr)
    lp = log_prior(m, b, log_f) 
    if ! isfinite(lp)
        return -Inf
    else return lp .+ log_likelihood(m, b, log_f, x, y, yerr)
    end
end


function fitexample()
    Random.seed!(123)
    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # Generate some synthetic data from the model.
    N = 50
    x = sort(10 * rand(N))
    yerr = 0.1 .+ 0.5 * rand(N)
    y = m_true .* x .+ b_true
    y += abs.(f_true .* y) .* randn(N)
    y += yerr .* randn(N)

    # fit
    fit_function(t, p) = p[1]*t .+ p[2]
    p0 = [0., 0.]
    fitted_curve = curve_fit(fit_function, x, y,p0)

    # maximaze likelihood
    Random.seed!(42)
    nll(arg) = -log_likelihood(arg[1], arg[2], arg[3], x, y, yerr)
    initial = [m_true, b_true, log(f_true)] #.+ rand(3)
    soln = Optim.optimize(nll, initial)
    #soln2 = scipy.optimize.minimize(nll, initial)
    
    # mcmc
    pos = transpose(Optim.minimizer(soln)) .+ 1e-4 * randn(32, 3)
    #pos = transpose(soln2["x"]) .+ 1e-4 * randn(32, 3)
    nwalkers, ndim = size(pos)
    println(nwalkers)
    nlll(arg) = log_likelihood(arg[1], arg[2], arg[3], x, y, yerr)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, nlll)
    sampler.run_mcmc(pos, 2000, progress=true)
    
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=true)

    labels = ["m", "b", "log(f)"]
    #=
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=true)
    for i in 1:ndim
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, length(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlim(0,5000)
    end
    
    axes[end].set_xlabel("step number");
    =#

    fig = corner.corner(
        flat_samples, labels=labels, truths=[m_true, b_true, log(f_true)], levels=(1-exp(-0.5),), quantiles=(0.16, 0.84)
    );
    #=
    errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    
    x0 = 0:0.1:10
    plot(x0, m_true * x0 .+ b_true, "k", alpha=0.3, lw=3)
    plot(x0, fit_function(x0, fitted_curve.param), "--k", alpha=0.5, lw=2)
    plot(x0, fit_function(x0, Optim.minimizer(soln)), ":k", alpha=1, lw=2)
    plot(x0, fit_function(x0, soln2["x"]), "r", alpha=0.5, lw=2)
    xlim(0, 10)
    xlabel("x")
    ylabel("y")
    =#
end

function QNM_model(Θ, frequency, time_unit, strain_unit)
    A, φ, ωr, ωi = Θ
    ft_Re_FH =  time_unit*strain_unit*abs.(Fourier_1mode.("real", frequency.*time_unit, A, φ, ωr, ωi, "EF"))
    return ft_Re_FH
end

function LogLikelihood(Θ, frequency, data, Sh, time_unit, strain_unit)
    model = QNM_model(Θ, frequency, time_unit, strain_unit)
    Sh = Sh.^2
    return inner_product(frequency, data, model, Sh) - inner_product(frequency, model, model, Sh)/2
end

function LogPrior(Θ)
    A, φ, ωr, ωi = Θ
    if 0 < A < 1 && 0.0 < φ < 2π && 0 < ωr < 1.0 && 0 < ωi < 0.1
        return 0.0
    else
        return -Inf
    end
end
function LogProbability(Θ, frequency, data, Sh, time_unit, strain_unit)
    lp = LogPrior(Θ)
    if ! isfinite(lp)
        return -Inf
    else return lp .+ LogLikelihood(Θ, frequency, data, Sh, time_unit, strain_unit)
    end
end



function MCMC_QNM(M_f, redshift, q_mass, detector)
    detectors = ["LIGO", "ET", "CE", "LISA"]
    detector_itp = Dict()
    detector_strain = Dict()

    for k in detectors
        detector_strain[k] = ImportDetectorStrain(k, false)
    end

    # QNM
    folders = readdir("../q_change/")

    for simu_folder_name in folders
        if occursin(q_mass, simu_folder_name)
            println(simu_folder_name)
            simu_folder = "../q_change/"*simu_folder_name
            ## fitted parameters
            ratios = h5open(simu_folder*"/arrays/fits/ratios.h5", "r") do file
                read(file)
            end
            amplitudes = h5open(simu_folder*"/arrays/fits/amplitudes.h5", "r") do file
                read(file)
            end
            phases = h5open(simu_folder*"/arrays/fits/phases.h5", "r") do file
                read(file)
            end
            omega = h5open(simu_folder*"/arrays/fits/omega.h5", "r") do file
                read(file)
            end
            dphases = Dict()
            for (key,value) in phases
                dphases[key] = phases["(2,2,0)"] - value
            end
            # Consider number of expected detection in (3,3,0) mode
            for (key, value) in amplitudes
                amplitudes[key] = abs(amplitudes[key])
            end
            # final mass
            mass_f = 0.0
            open(simu_folder*"/import_data/metadata.txt") do file
                parameters = Dict()
                for line in eachline(file)
                    # final mass
                    if occursin("remnant-mass", line)
                        mass_f = parse(Float64, split(line)[3])
                    end
                end
            end

            for (key, value) in phases
                while phases[key] > 2π
                    phases[key] -= 2π
                end
            end

            freq, tau = Dict(), Dict()

            mode_1 = "(2,2,0)"

            # Source parameters
            M_final = (1+redshift)*M_f
            M_total = M_final / mass_f
            D_L = Quantities.luminosity_distance(redshift)

            time_unit = (M_total)*PhysConstants.tSun
            strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)
            for (key, value) in omega
                freq[key] = value[1]/2/pi/time_unit
                tau[key] =  time_unit/value[2]
            end

            data_freqs = detector_strain[detector]["freq"]
            s = detector_strain[detector]["psd"]
            ft_Re_FH =  time_unit*strain_unit*abs.(Fourier_1mode.("real", data_freqs.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "EF"))

            data_fft = s.*exp.(1im.*rand(length(s))*2π) .+ ft_Re_FH

            Θ_true = [amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2]]

            Sn = detector_strain[detector]["psd"]



            # maximaze likelihood
            nll(Θ) = - LogLikelihood(Θ, data_freqs, data_fft, Sn, time_unit, strain_unit)
            soln = optimize(nll, Θ_true)
            

            pos = transpose(Optim.minimizer(soln)) .+ 1e-8 * randn(32, 4)
            nwalkers, ndim = size(pos)        
            
            logP(Θ) = LogProbability(Θ, data_freqs, data_fft, Sn, time_unit, strain_unit)
            
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logP)
            sampler.run_mcmc(pos, 1000, progress = true)
            
            samples = sampler.get_chain()
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=true)
            
            labels = ["amplitude", "phase", "ωr", "ωi"]
            fig = corner.corner(flat_samples, labels=labels, truths=Θ_true)
            fig.suptitle(detector*" M = $M_f, z = $redshift")


        end
    end 

end
