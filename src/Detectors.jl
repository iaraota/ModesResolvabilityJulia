# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end
using FrequencyTransforms, PhysConstants, Quantities, Random, Distributions, FFTW, DSP

using PyPlot, PyCall, HDF5, LaTeXStrings

mpl = pyimport("matplotlib")
np = pyimport("numpy")

function trapezio(f, x)
    y = 0
    @fastmath @inbounds @simd for i in 2:length(f)
        y += (x[i] - x[i-1])*(f[i] + f[i-1])/2
    end
    return y
end

function DetectorsPlot(M_f, redshift, q_mass)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)
    
    detectors = ["LIGO", "ET", "LISA"]
    colors = Dict("LIGO" => "tab:brown", "ET" => "tab:gray", "LISA" => "tab:purple")
    detector_strain = Dict()
    close("all")
    for k in detectors
        detector_strain[k] = ImportDetectorStrain(k, false)
        loglog(detector_strain[k]["freq"], detector_strain[k]["freq"].*detector_strain[k]["psd"], label = k, lw = 3, color = colors[k])
    end

    # QNM Fourier Transform
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

            mode_1 = "(2,2,0)"

            # Source parameters
            M_final = (1+redshift)*M_f
            M_total = M_final / mass_f
            D_L = Quantities.luminosity_distance(redshift)

            time_unit = (M_total)*PhysConstants.tSun
            strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

            freq = 8e-4:1e-3:3e1

            ft_Re =  time_unit*strain_unit*abs.(Fourier_1mode.("real", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH"))
            ft_Im =  time_unit*strain_unit*abs.(Fourier_1mode.("imaginary", freq.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH"))
            loglog(freq, 2 .*freq.*ft_Im, lw = 3, color = "black", label = L"$(2,2,0), M = 1\times 10^{5}, z = 2$")

        
        end
    end 
    xlim(minimum(detector_strain["LISA"]["freq"]), maximum(detector_strain["ET"]["freq"]))

    xlabel("Frequency [Hz]")
    ylabel("Characteristic strain")
    tight_layout()

    legend()
    savefig("figs/Detector_FT.pdf")

    
end

function log_prob(x, μ, Σ)
    diff = x .- μ
    return -.5*dot(diff, Σ\diff)
end

function NoisePlots()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)
    
    detectors = ["LIGO", "ET", "CE", "LISA"]
    colors = Dict("LIGO" => "tab:brown", "ET" => "tab:gray", "LISA" => "tab:purple")
    
    detector_strain = Dict()
    close("all")

    for k in detectors
        detector_strain[k] = ImportDetectorStrain(k, false)
        #loglog(detector_strain[k]["freq"], detector_strain[k]["freq"].*detector_strain[k]["psd"], label = k, lw = 3, color = colors[k])
        #hist(detector_strain[k]["freq"])
    end


    detector = "LISA"
    itp = ImportDetectorStrain(detector, true)

    f_min = minimum(detector_strain[detector]["freq"])*0.1
    f_max = maximum(detector_strain[detector]["freq"])*2
    df = f_min*1
    f = f_min:df:f_max
    s = itp(f)
    max = maximum(s)
    for i in 1:length(s)
        if s[i] == 0
            s[i] = max
        end
    end
    noise_phases = rand(length(s))*2π
    s = s.*exp.(1im.*noise_phases)
    #s = vcat(s, reverse(conj(s[2:end])))

    dt = 1/df/length(s)

    N = length(s)
    t = (-N/2:(N-1)/2)*dt
    g = FFTW.ifft(s)/dt  |> fftshift

    #plot(t,g)
    α = 0
    # Compute FFT
    ## FFT frequencies
    freqs = FFTW.fftfreq(length(g), 1.0/dt) |> fftshift
    ## tukey window
    window = DSP.Windows.tukey(length(g), α)
    df_fft = freqs[2] .-freqs[1]
    ## FFT of real part
    fft = dt*FFTW.fft(window.*g) |> fftshift
    loglog(f,abs.(itp(f)), lw = 3)
    loglog(f, abs.(s))
    loglog(freqs .+ df_fft, abs.(fft), "-r", lw = 3)
end

function SignalNoisePlots(M_f, redshift, q_mass)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [20, 8]  # plot image size

	SMALL_SIZE = 15
	MEDIUM_SIZE = 20
	BIGGER_SIZE = 28
	
	plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
	plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc("xtick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("ytick", labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)
    
    detectors = ["LIGO", "ET", "CE", "LISA"]
    colors = Dict("LIGO" => "tab:brown", "ET" => "tab:gray", "LISA" => "tab:purple")
    
    detector_strain = Dict()
    close("all")

    for k in detectors
        detector_strain[k] = ImportDetectorStrain(k, false)
        #loglog(detector_strain[k]["freq"], detector_strain[k]["freq"].*detector_strain[k]["psd"], label = k, lw = 3, color = colors[k])
        #hist(detector_strain[k]["freq"])
    end


    detector = "ET"
    itp = ImportDetectorStrain(detector, true)

    f_min = minimum(detector_strain[detector]["freq"])*0.1
    f_max = maximum(detector_strain[detector]["freq"])*5
    df = f_min*1
    f = f_min:df:f_max
    s = itp(f)
    max = maximum(s)
    for i in 1:length(s)
        if s[i] == 0
            s[i] = max
        end
    end
    noise_phases = rand(length(s))*2π
    s = s.*exp.(1im.*noise_phases)
    #s = vcat(s, reverse(conj(s[2:end])))

    dt = 1/df/length(s)

    N = length(s)
    t = (-N/2:(N-1)/2)*dt
    t = (0:N-1)*dt
    g = FFTW.ifft(s)/dt  |> fftshift

    #plot(t,g)
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
            qnm_re = strain_unit*amplitudes[mode_1]*exp.(-t/tau[mode_1]).*cos.(2π*freq[mode_1].*t .+ phases[mode_1])
            ft_Re_EF =  time_unit*strain_unit*abs.(Fourier_1mode.("real", f.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "EF"))
            
            t_FH = (-N/2:(N-1)/2)*dt
            
            qnm_re_FH = strain_unit*amplitudes[mode_1]*exp.(-abs.(t_FH)/tau[mode_1]).*cos.(2π*freq[mode_1].*t_FH .+ phases[mode_1])/2
            ft_Re_FH =  time_unit*strain_unit*abs.(Fourier_1mode.("real", f.*time_unit, amplitudes[mode_1], phases[mode_1], omega[mode_1][1], omega[mode_1][2], "FH"))

            #plot(qnm_re)
            d = g .+ qnm_re
            close("all")
            #plot(t, d)
            #plot(t, qnm_re)
            #plot(t_FH, qnm_re_FH)

            
            α = 0
            # Compute FFT
            ## FFT frequencies
            freqs = FFTW.fftfreq(length(d), 1.0/dt) |> fftshift
            freqs = freqs .+ freqs[2] .- freqs[1]
            ## tukey window
            window = DSP.Windows.tukey(length(d), α)
            #plot(t,window.*qnm_re)
            ## FFT of real part
            fft = dt*FFTW.fft(window.*d) |> fftshift
            loglog(freqs, abs.(fft), lw = 3)
            loglog(f,abs.(itp(f)), lw = 3)
            loglog(f,ft_Re_FH, ls = "--", lw = 3)
            loglog(f,ft_Re_EF, ls = "--", lw = 3)
         end
     end 
end
