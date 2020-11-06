# Add modules to load path
if !("./src/module" in LOAD_PATH)
    push!(LOAD_PATH, "./src/module")
end

using DelimitedFiles, Distributed, HDF5, Random, Distributions, ProgressMeter, PyPlot, FFTW, DSP, PyCall

mpl = pyimport("matplotlib")

using Interpolations
using FrequencyTransforms, PhysConstants, Quantities

function trapezio(f, x)
    y = 0
    @fastmath @inbounds @simd for i in 2:length(f)
        y += (x[i] - x[i-1])*(f[i] + f[i-1])/2
    end
    return y
end
function integralSNRFH(f, M, r, A, freq, tau, Sh)
    return 1/Sh*(1/((f + freq)^2 + (2*pi*tau)^(-2))^2 + 1/((f - freq)^2 + (2*pi*tau)^(-2))^2)
end
function rho_FH(f, M, r, A, freq, tau, Sh)
    return (M/r)^2*A^2/(80*pi^5*tau^2)*trapezio(integralSNRFH.(f, M, r, A, freq, tau, Sh), f)
end

function approx_rho_FH(M, r, A, freq, tau, Sh_f)
    return (M/r)^2*pi*freq*tau*A^2/(20*pi^2*freq*Sh_f)
end

function ShNSA(f)
    return 9.18e-52*f^(-4) + 1.59e-41 + 9.18e-38*f^2
end

function EF_rho(M, r, A, freq, tau, Sh_f)
    return pi*freq*tau/(40*pi^2*freq*(1 + 4*(pi*freq*tau)^2)*Sh_f)*(M*A/r)^2*2*(1 + 4*(pi*freq*tau)^2)
end

function ft_psi(f, freq, tau)
    return (1/tau + 1im*(2*pi*freq + 2*pi*f))/(1/tau^2 + (2*pi*f + 2*pi*freq)^2)
end

function ft_psi_cc(f, M, r, A, phi, freq, tau)
    return (M*A/r)*exp(-1im*phi)*(1/tau * 1im*(-2*pi*freq + 2*pi*f))/(1/tau^2 + (2*pi*f - 2*pi*freq)^2)
end
function ft_re(f, φ, freq, tau)
    return ((1im*2*pi*f + 1/tau)*cos(φ) + 2*pi*freq*sin(φ)) / ((2*pi*freq)^2 - (2*pi*f - 1im/tau)^2)
end
function ft_im(f, φ, freq, tau)
    return -(-(1im*2*pi*f + 1/tau)*sin(φ) + 2*pi*freq*cos(φ)) / (-(2*pi*freq)^2 + (2*pi*f - 1im/tau)^2)
end
function berti_ft(tau, freq, f)
    return 1/tau/((1/tau)^2 + (2*pi*freq +2*pi*f)^2)    
end

function arg_iara(f, freq, tau, Sh)
    return (1/((1/tau)^2 +(2*pi*f + 2*pi*freq)^2) + 1/((1/tau)^2 + (2*pi*f - 2*pi*freq)^2))/Sh
end

function SNR_iara(f, M, r, A, freq, tau, Sh)
    return (M*A/r)^2/10/pi*trapezio(arg_iara.(f, freq, tau, Sh), f)
end

function arg_berti(f, freq, tau, Sh)
    return (1/(1/(2*pi*tau)^2 +(f + freq)^2)^2 + 1/(1/(2*pi*tau)^2 + (f - freq)^2)^2)/Sh
end

function SNR_berti(f, M, r, A, freq, tau, Sh)
    return (M*A/r)^2/80/π^5/tau^2*trapezio(arg_berti.(f, freq, tau, Sh), f)
end
function CompareAll(M_f, redshift)
    mass_f = 0.952032939704
    ωr = 0.55578866 
    ωi = 0.08517178
    A = 0.4118841893118968

    # Source parameters
    M_final = (1+redshift)*M_f*PhysConstants.tSun
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)*PhysConstants.Dist

    time_unit = (M_total)
    strain_unit = ((M_final)) / (D_L)

    freq = ωr/2/pi/time_unit
    tau =  time_unit/ωi

    f = 1e-5:1e-5:1e0
    Sh = ShNSA.(f)



    
    #FT_Hz = (A^2 ./ ((1 ./tau).^2 .+ (2*pi*(freq .- f)).^2)).*strain_unit^2

    #FT_QNM = sqrt.(Fourier_1mode.("psd", f*time_unit, A, 0, ωr, ωi))

    #FT_Re = Fourier_1mode.("real", f*time_unit, A, 0, ωr, ωi)
    #FT_Re = A*((1im*2*pi*f .+ 1/tau)) ./ ((2*pi*freq).^2 .- (2*pi*f .- 1im/tau).^2)

    #FT_Psi_Re = (ft_psi.(f, M_final, D_L, A, 0, freq, tau) .+ ft_psi_cc.(f, M_final, D_L, A, 0, freq, tau))./2
    ## Compute signal-to-noise-ratio 
    #SNR_numerical = sqrt(trapezio(4*abs.(time_unit*strain_unit*FT_QNM).^2 ./ Sh, f))
    norm = A*M_final/D_L

    FT_berti_re = norm*(berti_ft.(tau,freq,f) + berti_ft.(tau,-freq,f))./2
    FT_berti_im = norm*(berti_ft.(tau,freq,f) - berti_ft.(tau,-freq,f))./2*1im
    FT_QNM_re = norm*(ft_psi.(f, freq,tau) .+ ft_psi.(f, -freq,tau))./2
    FT_QNM_im = norm*(ft_psi.(f, freq,tau) .- ft_psi.(f, -freq,tau))./2im
    
    loglog(f, sqrt.(Sh), color = "tab:purple")
    axvline(x = freq, ls = ":", color = "k")
    loglog(f, abs.(FT_QNM_re), color = "tab:blue")
    loglog(f, abs.(FT_berti_re), color = "tab:red")
    title("Imaginary part")
    ylabel("Characteristic strain")
    xlabel("Frequency [Hz]")
    SNR_numerical = sqrt(1/5/2/pi*trapezio(4*abs.((FT_QNM_im)).^2 ./ Sh, f))
    SNR_berti = sqrt(1/5/2/pi*trapezio(4*abs.((FT_berti_re + FT_berti_im)).^2 ./ Sh, f))
    #SNR_Re = sqrt(trapezio(4*abs.(strain_unit*FT_Re).^2 ./ Sh, f))
    #SNR_Psi_Re = sqrt(trapezio(4*abs.(FT_Psi_Re).^2 ./ Sh, f))

    #SNR_Hz = sqrt(trapezio(4*abs.(FT_Hz) ./ Sh, f))
    SNR_FH = sqrt(rho_FH(f, M_final, D_L, A, freq, tau, Sh))

    SNR_approx = sqrt(approx_rho_FH(M_final, D_L, A, freq, tau, ShNSA(freq)))
    EF_SNR = sqrt(EF_rho(M_final, D_L, A, freq, tau, ShNSA(freq)))

    println(SNR_numerical)
    println(SNR_berti)
    #println(SNR_Re)
    #println(SNR_Psi_Re)
    #println(SNR_Hz)
    println(SNR_FH)
    println(SNR_approx)
    println(EF_SNR)
   
end

function PlotSNR(redshift)
    mass_f = 0.952032939704
    ωr = 0.55578866 
    #ωi = 0.08517178
    ωi = 1e-2
    A = 0.4118841893118968
    masses = exp10.(range(4, stop = 10, length = 20))

    f = exp10.(range(-5, stop = 0, length = Int(1e4)))
    Sh = ShNSA.(f)

    SNR_ber = zeros(0)
    SNR = zeros(0)
    SNR_approx = zeros(0)
    SNR_tukey = Dict()
    SNR_reflect = Dict()
    alphas = [1, .1, .01, .001,1e-10]
    #alphas = [.1, .05, .01]
    for i in alphas
        SNR_tukey[i] = zeros(0)
        SNR_reflect[i] = zeros(0)

    end
    t_final = 100
    dtime = 0.1
    phi = π
    for M_f in masses
        # Source parameters
        M_final = (1+redshift)*M_f*PhysConstants.tSun
        M_total = M_final / mass_f
        D_L = Quantities.luminosity_distance(redshift)*PhysConstants.Dist

        time_unit = (M_total)
        strain_unit = ((M_final)) / (D_L)

        freq = ωr/2/pi/time_unit
        tau =  time_unit/ωi

        norm = A*M_final/D_L
        FT_berti_re = norm*(exp(1im*phi)*berti_ft.(tau,freq,f) + exp(-1im*phi)*berti_ft.(tau,-freq,f))./2
        FT_berti_im = norm*(exp(1im*phi)*berti_ft.(tau,freq,f) - exp(-1im*phi)*berti_ft.(tau,-freq,f))./2/1im
        FT_QNM_re = norm*(exp(1im*phi)*ft_psi.(f, freq,tau) .+ exp(-1im*phi)*ft_psi.(f, -freq,tau))./2
        FT_QNM_im = norm*(exp(1im*phi)*ft_psi.(f, freq,tau) .- exp(-1im*phi)*ft_psi.(f, -freq,tau))./2im

        # Number of points 
        N = 2^12 - 1 
        # Start time 
        t0 = 0 
        #=
        tmax = 1/1e-5
        # Sample period
        println(M_f)
        fs = 3
        Ts = 1/fs
        =#
        tmax = t_final*time_unit

        #Ts = (tmax - t0)/N
        Ts = dtime *time_unit
        # time coordinate
        
        t1 = -tmax:Ts:0
        t2 = 0:Ts:tmax
        tall = -tmax:Ts:tmax
        t = 0:Ts:tmax

        # signal 
    
        signal_reflec_re = real.(QNM_reflec.(tall, M_final, D_L, A, phi, freq, tau)/2)
        signal_reflec_im = imag.(QNM_reflec.(tall, M_final, D_L, A, phi, freq, tau)/2)
        freqs_reflec = FFTW.fftfreq(length(signal_reflec_re), 1.0/Ts) |> fftshift

        signal_re = QNM_Re.(t, M_final, D_L, A, phi, freq, tau)
        signal_im = QNM_Im.(t, M_final, D_L, A, phi, freq, tau)
        freqs_tukey = FFTW.fftfreq(length(t), 1.0/Ts) |> fftshift
        close("all")
        for i in alphas
            ft_tukey_re = Ts*fft(DSP.Windows.tukey(length(signal_re),i).*signal_re) |> fftshift
            ft_tukey_im = Ts*fft(DSP.Windows.tukey(length(signal_im),i).*signal_im) |> fftshift
            ft_reflec_re = Ts*fft(DSP.Windows.tukey(length(signal_reflec_re),i).*(signal_reflec_re)) |> fftshift
            ft_reflec_im = Ts*fft(DSP.Windows.tukey(length(signal_reflec_im),i).*(signal_reflec_im)) |> fftshift
            append!(SNR_tukey[i], sqrt.(1/5/4/pi*trapezio(4 .*abs.(ft_tukey_re).^2 ./ ShNSA.(freqs_tukey), freqs_tukey)))
            append!(SNR_reflect[i], sqrt.(1/5/4/pi*trapezio(4 .*abs.(ft_reflec_re).^2 ./ ShNSA.(freqs_reflec), freqs_reflec)))
            #loglog(freqs_tukey, abs.(ft_tukey_re), label = "tukey α = "*string(i), ls = "--")
            #loglog(freqs_reflec, abs.(ft_reflec_re), label = "reflect α = "*string(i))
            #legend(fontsize = 10)

        end

    
        append!(SNR,sqrt(1/5/4/pi*trapezio(4*abs.((FT_QNM_re)).^2 ./ Sh, f)))
        append!(SNR_ber,sqrt(1/5/2/pi*trapezio(4*abs.((FT_berti_re)).^2 ./ Sh, f)))
  #      append!(SNR_tapper, sqrt.(1/5/4/pi*trapezio(4 .*abs.(F_tapper).^2 ./ ShNSA.(freqs), freqs)))
        #append!(SNR_itp, sqrt.(1/5/4/pi*trapezio(4 .*abs.(sitp.(f)).^2 ./ ShNSA.(f), f)))
        #append!(freqs, freq.^(-3))
        #append!(SNR, sqrt.(SNR_iara(f, M_final, D_L, A, freq, tau, Sh)))
       # append!(SNR_ber, sqrt.(SNR_berti(f, M_final, D_L, A, freq, tau, Sh)))
        append!(SNR_approx, sqrt.(approx_rho_FH(M_final, D_L, A, freq, tau, ShNSA(freq))))
    end
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
	rcParams["mathtext.fontset"] = "cm"
	rcParams["font.family"] = "STIXGeneral"
	rcParams["figure.figsize"] = [15, 8]  # plot image size

    font_size = 28
    mpl.rc("font", size=font_size)
    
    close("all")
    loglog(masses, SNR_ber, label = "FH", ls = "-", lw = 4, color = "aqua", alpha = 0.4)
    loglog(masses, SNR, ls = "-", label = "EF", lw = 4, color = "pink", alpha = 0.4)
    #loglog(masses, SNR_approx, ls = "-", label = "delta", lw = 4, color = "lime", alpha = 0.4)
   
    for i in alphas
        loglog(masses, SNR_reflect[i], ls = "-", label = "reflect α = "*string(i), lw = 2)
        loglog(masses, SNR_tukey[i], label = "tukey α = "*string(i), lw = 2, ls = "--")
    end
    
    
    #loglog(masses, SNR_tapper, color = "tab:purple", label = "tukey ") 
    #loglog(masses, SNR_itp, color = "tab:orange", label = "interp") 
    #loglog(f, sqrt.(Sh).*1e2)
    #loglog(masses, ShNSA.(freqs)*1e42, color = "tab:purple", label = L"$S_h\times 10^{42}$")
    ylabel("SNR")
    xlabel("Final mass")
    title("Real, z = "*string(redshift)*", T = "*string(t_final)*" M, dt = "*string(dtime)*", ϕ = "*string(phi))
    legend(fontsize = 10)
    savefig("figs/SNR/Re_SNR_z"*string(redshift)*"_T"*string(t_final)*"_dt"*string(dtime)*"_phi"*string(phi)*".png")
    #title("Real + Imaginary")

end

function QNM_Re(t, M, r, A, ϕ, freq, tau)
    return (M*A/r)*exp(-t/tau)*cos(2π*freq*t + ϕ)
end

function QNM_Re_reflec(t, M, r, A, ϕ, freq, tau)
    return (M*A/r)*exp(-abs(t)/tau)*cos(2π*freq*t + ϕ)
end

function QNM_Im(t, M, r, A, ϕ, freq, tau)
    return (M*A/r)*exp(-t/tau)*sin(2π*freq*t + ϕ)
end

function QNM_Im_reflec(t, M, r, A, ϕ, freq, tau)
    return (M*A/r)*exp(-abs(t)/tau)*sin(2π*freq*t + ϕ)
end

function QNM_reflec(t, M, r, A, ϕ, freq, tau)
    return (M*A/r)*exp(-abs(t)/tau)*exp(1im*(2π*freq*t + ϕ))
end

using PyCall
sp = pyimport("scipy")
function FourierWindow(M_f, redshift)
    mass_f = 0.952032939704
    ωr = 0.55578866 
    #ωi = 0.08517178
    ωi = 1e-2
    A = 0.4118841893118968

    # Source parameters
    M_final = (1+redshift)*M_f*PhysConstants.tSun
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)*PhysConstants.Dist

    time_unit = (M_total)
    strain_unit = ((M_final)) / (D_L)

    freq = ωr/2/pi/time_unit
    tau =  time_unit/ωi
    # Number of points 
    N = 2^18 - 1 
    # Start time 
    t0 = 0 
    tmax = 1000*time_unit
    tmax = 1/1e-6
    # Sample period
    fs = 3
    Ts = 1/fs
    N = (tmax - t0)/Ts


    # time coordinate
    t = t0:Ts:tmax
    #=
    # signal 
    signal = QNM_Re.(t, M_final, D_L, A, 0, freq, tau)# + 1im*QNM_Im.(t, M_final, D_L, A, 0, freq, tau)
    taper = DSP.Windows.tukey(length(signal), 0.01)
    signal_tappered = taper.*signal
    # Fourier Transform of it 
    F = Ts*fft(signal) |> fftshift
    F_tapper = Ts*fft(signal_tappered) |> fftshift
    freqs_fft = FFTW.fftfreq(length(t), 1.0/Ts) |> fftshift
    window = Ts*fft(taper) |> fftshift

    f = minimum(freqs_fft):1e-6:maximum(freqs_fft)
    itp = interpolate(abs.(F_tapper),  BSpline(Cubic(Line(OnGrid()))))
    sitp = Interpolations.scale(itp, freqs_fft)
    =#
    norm = A*M_final/D_L

    #sci_itp = sp.interpolate.interp1d(freqs_fft, F_tapper)
    freqs = 1e-5:1e-6:1
    t_final = 1000
    dtime = 0.1
    phi = π/2
    FT_berti_re = norm.*(berti_ft.(tau,freq,freqs) .+ berti_ft.(tau,-freq,freqs))./2
    FT_berti_im = norm.*(berti_ft.(tau,freq,freqs) .- berti_ft.(tau,-freq,freqs))./2im
    FT_QNM_re = norm.*(ft_psi.(freqs, freq,tau) .+ ft_psi.(freqs, -freq,tau))./2
    FT_QNM_im = norm.*(ft_psi.(freqs, freq,tau) .- ft_psi.(freqs, -freq,tau))./2im

    tmax = t_final*time_unit

    #Ts = (tmax - t0)/N
    Ts = dtime *time_unit
    # time coordinate
    
    t = t0:Ts:tmax
    t1 = -tmax:Ts:0
    t2 = 0:Ts:tmax
    tall = -tmax:Ts:tmax
    #signal_reflec_Re = QNM_Re_reflec.(tall, M_final, D_L, A, 0, freq, tau)/2
    signal_reflec_Im = real(QNM_reflec.(tall, M_final, D_L, A, phi, freq, tau)/2)
    #ft_reflec_Re = Ts*fft(signal_reflec_Re) |> fftshift
    ft_reflec_Im = Ts*fft(signal_reflec_Im) |> fftshift
    freqs_reflec = FFTW.fftfreq(length(signal_reflec_Im), 1.0/Ts) |> fftshift
    
    signal = QNM_Re_reflec.(t, M_final, D_L, A, phi, freq, tau)
    freqs_tukey = FFTW.fftfreq(length(t), 1.0/Ts) |> fftshift
    alphas = [1]#, 0.5, 0.1, 0.01, .001]
    loglog(freqs, abs.(FT_QNM_re), label = "iara")
    loglog(freqs, abs.(FT_berti_re), label = "berti")
    for i in alphas
        ft_tukey = Ts*fft(DSP.Windows.tukey(length(signal),i).*signal) |> fftshift
        ft_reflec = Ts*fft(DSP.Windows.tukey(length(signal_reflec_Im),i).*signal_reflec_Im) |> fftshift
        loglog(freqs_reflec, abs.(ft_reflec), label = "fft reflect α = "*string(i), ls = "--")
        loglog(freqs_tukey, abs.(ft_tukey), label = "tukey α = "*string(i), ls = "-.")
    end
    # plots 
    #plot(t, signal)
    #loglog(freqs_fft, abs.(F), label = "rectangular")
    #loglog(freqs_fft, abs.(F_tapper), label = "tukey α = 0.01")
    #loglog(f, abs.(sci_itp(f)), label = "scipy")
    #loglog(f, abs.(sitp.(f)), label = "interpolation")
    #loglog(f, abs.(sitp(f)), label = "interpolation")

    loglog(freqs_reflec, abs.(ft_reflec_Im), label = "reflect no win", ls = ":")
    title("Real, z = "*string(redshift)*", T = "*string(t_final)*" M, dt = "*string(dtime)*", ϕ = "*string(phi))
    legend()
    savefig("figs/SNR/Re_FT_z"*string(redshift)*"_T"*string(t_final)*"_dt"*string(dtime)*"_phi"*string(phi)*".png")
    #freq_domain = plot(freqs, abs.(F), title = "Spectrum", xlim=(-1000, +1000)) 
    #plot(time_domain, freq_domain, layout = 2)
    #savefig("Wave.pdf")

end

function FTPlots(M_f, redshift,f)
    mass_f = 0.952032939704
    ωr_220 = 0.55578866 
    ωi_220 = 0.08517178

    ωr_221 = 0.5427
    ωi_221 = 0.2564

    #ωi = 1e-2
    A_220 = 0.4118841893118968
    A_221 = 0.66*A_220
    masses = exp10.(range(4, stop = 9, length = 50))

    #f = exp10.(range(-5, stop = 0, length = Int(1e4)))
    #Sh = ShNSA.(f)

    M_final = (1+redshift)*M_f*PhysConstants.tSun
    M_total = M_final / mass_f
    D_L = Quantities.luminosity_distance(redshift)*PhysConstants.Dist

    time_unit = (M_total)
    strain_unit = ((M_final)) / (D_L)

    freq_220 = ωr_220/2/pi/time_unit
    tau_220 =  time_unit/ωi_220

    freq_221 = ωr_221/2/pi/time_unit
    tau_221 =  time_unit/ωi_221

    norm_220 = A_220*M_final/D_L

    norm_221 = A_221*M_final/D_L

    FT_berti_re_220 = norm_220*(berti_ft.(tau_220,freq_220,f) + berti_ft.(tau_220,-freq_220,f))./2
    FT_berti_im_220 = norm_220*(berti_ft.(tau_220,freq_220,f) - berti_ft.(tau_220,-freq_220,f))./2/1im
    FT_QNM_re_220 = norm_220*(ft_psi.(f, freq_220,tau_220) .+ ft_psi.(f, -freq_220,tau_220))./2
    FT_QNM_im_220 = norm_220*(ft_psi.(f, freq_220,tau_220) .- ft_psi.(f, -freq_220,tau_220))./2im
    
    FT_berti_re_221 = norm_221*(berti_ft.(tau_221,freq_221,f) + berti_ft.(tau_221,-freq_221,f))./2
    FT_berti_im_221 = norm_221*(berti_ft.(tau_221,freq_221,f) - berti_ft.(tau_221,-freq_221,f))./2/1im
    FT_QNM_re_221 = norm_221*(ft_psi.(f, freq_221,tau_221) .+ ft_psi.(f, -freq_221,tau_221))./2
    FT_QNM_im_221 = norm_221*(ft_psi.(f, freq_221,tau_221) .- ft_psi.(f, -freq_221,tau_221))./2im
    
    #loglog(f, abs.(FT_QNM_re_220), label = "EF, M = "*string(M_f))
   # loglog(f, abs.(FT_QNM_re_221), label = "EF - (2,2,1)")
    
    #loglog(f, abs.(FT_berti_re_220), label = "FH, M = "*string(M_f))
   # loglog(f, abs.(FT_berti_re_221), label = "FH - (2,2,1)")

    #loglog(f, abs.(FT_QNM_re_220 + FT_QNM_re_221), label = "EF - (2,2,0) + (2,2,1)")
    #loglog(f, abs.(FT_berti_re_220 + FT_berti_re_221), label = "FH - (2,2,0) + (2,2,1)")

    loglog(f, 2 .*f.*abs.(FT_berti_re_220), label = "FH, M = "*string(M_f))
    loglog(f, 2 .*f.*abs.(FT_QNM_re_220), label = "EF, M = "*string(M_f))
    legend()

end
