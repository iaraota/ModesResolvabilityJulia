include("src/SNR.jl")

for i in 9:10
    RunAllSXSFolders(exp10.(range(i, stop = i+1, length = 100)), "LISA", 1, 0, 1, "q10_220_1e"*string(i))
end