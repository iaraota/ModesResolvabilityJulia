using Gnuplot,Random

function gpPlot(X,Y,Z)
    x = X #log10.(X)
    y = Y
    z = Z #log10.(Z)
    #=
    @gsp x y z "w l lc palette" "set view map" "set dgrid3d" "set pm3d"
    @gsp :- "set pm3d interpolate 1,1" 
    @gsp :- "set logscale x" "set logscale cb"
    @gsp :- xrange = (minimum(X), maximum(X)) yrange =  (minimum(Y), maximum(Y))
    @gsp :- cbrange = (minimum(Z), 1000)
    =#
    @gp x y z "w p lc palette"
    @gp :- "set logscale x"  "set logscale cb"
end