if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using PyPlot, DSP, DelimitedFiles, Dierckx, HDF5, FFTW
using PyCall

using FrequencyTransforms, PhysConstants, Quantities

sp = pyimport("scipy.signal")
np = pyimport("numpy")
function testeSpec()
    strain_H1 = readdlm("/home/iara/Doutorado/Julia/ModesResolvability/data/teste/H-H1_LOSC_4_V2-1126259446-32.txt", comments=true)[:,1]
    t0 = 1126259462.4 
    dt = 1/length(strain_H1)
    #plot(range(-16, stop = 16, length = length(strain_H1)), strain_H1)
    #grid()
    #tight_layout()
    deltat = 5
    fs = 4096
    # pick a shorter FTT time interval, like 1/8 of a second:
    NFFT = Int(fs/8)
    # and with a lot of overlap, to resolve short-time features:
    NOVL = Int(NFFT*15. /16)

    window = blackman(NFFT)
    spec_cmap="ocean"
    spec_H1, freqs, bins, im = plt.specgram(strain_H1, NFFT=NFFT, Fs=fs, window=window, noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat], mode = "psd")
#    plt.xlabel("time (s) since GW150914")
#    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
#    plt.axis([-deltat, deltat, 0, 2000])
#    plt.title("aLIGO H1 strain data near GW150914")
    # and choose a window that minimizes "spectral leakage" 

    f, t, Sxx = sp.spectrogram(strain_H1, fs, nfft = NFFT, window = "hanning")
    #pcolormesh(t,f,Sxx, shading = "gouraud", rasterized = true)
    #plt.colorbar()

end

function SXSSpecGram(M_f, redshift)
    l2m2 = readdlm("data/SXS/all_l2m2_sim_1.5.dat", comments=true)
    time = l2m2[:, 1]
    h_real = l2m2[:, 2]
    h_img= l2m2[:, 3]
    # ringdown
    amplitude_22 = sqrt.(h_real.^2)# + h_img.^2)
    max_ind = findall(amplitude_22 .== maximum(amplitude_22))[1]
    final_time_index = 1000
    interval_22 = max_ind:(final_time_index + max_ind)
    
    rd_re = h_real[interval_22]
    rd_im = h_img[interval_22]
    rd_time = time[interval_22] .- time[max_ind]
    
    time = time .- time[max_ind]
    # interpolate
    h_itp_re = Spline1D(time, h_real, bc = "zero")
    h_itp_im = Spline1D(time, h_img, bc = "zero")
    rd_itp_re = Spline1D(rd_time, rd_re, bc = "zero")
    rd_itp_im = Spline1D(rd_time, rd_im, bc = "zero")

    # QNM parameters
    simu_folder = "data/SXS/1.5"
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

    # Source parameters
    M_final = (1+redshift)*M_f
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)

    time_unit = (M_total)*PhysConstants.tSun
    strain_unit = ((M_final)*PhysConstants.tSun) / (D_L*PhysConstants.Dist)

    # Fourier transform
    fs = 2*4096 #Hz
    fs_sim_unit = fs*time_unit
    
    dt = 1/fs_sim_unit    
    
    initial_imr = -300
    initial_t_rd = 10
    final_time = 100
    times_rd = range(initial_t_rd, stop = final_time, step = dt)
    times_peak = range(0, stop = final_time, step = dt)
    times_all =  range(initial_imr, stop = final_time, step = dt)

    ## Physical units
    dt *= time_unit

    gw_time = Dict()

    gw_time["ringdown"] = times_rd * time_unit
    gw_time["peak"] = times_peak * time_unit
    gw_time["IMR"] = times_all * time_unit

    gw_strain = Dict()
    for key in keys(gw_time)
        gw_strain[key] = Dict()
    end

    gw_strain["ringdown"]["real"] = rd_itp_re(times_rd) * strain_unit
    gw_strain["ringdown"]["imaginary"] = rd_itp_im(times_rd) * strain_unit
    
    gw_strain["peak"]["real"] = rd_itp_re(times_peak) * strain_unit
    gw_strain["peak"]["imaginary"] = rd_itp_im(times_peak) * strain_unit
    
    gw_strain["IMR"]["real"] = h_itp_re(times_all) * strain_unit
    gw_strain["IMR"]["imaginary"] = h_itp_im(times_all) * strain_unit

    gw_freqs = Dict()
    gw_fft = Dict()

    ## tukey shape parameter
    α = [0, 0.1, 0.01]
    ls = Dict(0=> "-", 0.1 =>"--", 0.01 => ":")
#=

    for α_i in α
        for (key, value) in gw_strain
            gw_freqs[key] = FFTW.fftfreq(length(gw_time[key]), 1.0/dt) |> fftshift 

            gw_fft[key] = Dict()

            ## tukey window
            tukey_window = DSP.Windows.tukey(length(gw_time[key]), α_i)
            for (k, v) in value
                ## FFT 
                gw_fft[key][k] = dt*fft(tukey_window.*v) |> fftshift
            end
        end
        for (key, value) in gw_freqs
            loglog(value, 2*value.*abs.(gw_fft[key]["real"]), label = key*" α = $α_i", lw = 2, ls = ls[α_i])
        end
    end
    #ylim(1e-25,1e-21)
    xlabel("freq [Hz]")
    ylabel("Characteristic strain")
    legend()
    tight_layout()
    close("all")
   =# 

    # spectrogram
    fs = 1. /dt
    strain = gw_strain["IMR"]["real"]/strain_unit
    initial_time = initial_imr
    strain = gw_strain["ringdown"]["real"]/strain_unit
    strain_rd = gw_strain["ringdown"]["real"]/strain_unit
    initial_time = initial_t_rd
    strain = gw_strain["peak"]["real"]/strain_unit
    initial_time = 0

    # NFFT
    NFFT = Int32(round(length(strain_rd)/ 2))
    NOVL = Int32(round(NFFT*15. /16))

    window = tukey(NFFT, 1)
    spec_cmap="viridis"
    close("all")
    spec, freqs, bins, im = plt.specgram(strain, NFFT=NFFT, Fs=fs, window=window, 
    noverlap=NOVL, cmap=spec_cmap,mode = "psd", scale = "dB", xextent = [initial_time*time_unit,final_time*time_unit],
    vmin = -200)

    ylim(0,200)
    ylabel("Frequency (Hz)")
    xlabel("Time [s]")
    colorbar()
    println(0.5417325/time_unit/2/pi)
    println(time_unit/0.08587793)
#    max_freq_ind = findall(abs.(gw_fft["peak"]["real"]) .== maximum(abs.(gw_fft["peak"]["real"])))[1]
#    println("Maximum peak frequency: ", abs(maximum(gw_freqs["peak"][max_freq_ind])))

#    max_freq_ind = findall(abs.(gw_fft["ringdown"]["real"]) .== maximum(abs.(gw_fft["ringdown"]["real"])))[1]
#    println("Maximum ringdown frequency: ", abs(maximum(gw_freqs["ringdown"][max_freq_ind])))
    #plot(gw_time["ringdown"], gw_strain["ringdown"]["real"])
end