using Yao,YaoExtensions,PyCall,YaoPlots, JSON, Statistics, Yao.AD, Compose, LaTeXStrings, Pickle, LinearAlgebra
using QuAlgorithmZoo: Adam, update!


# include all the useful functions for VpVQD
include("pvqd_functions.jl")
import PyPlot; const plt = PyPlot

# Now import useful subroutines from python
pushfirst!(PyVector(pyimport("sys")."path"), "")


## First of all, load data from pVQD file

## Sys data
n_spins   = 3
depth     = 1
dt        = 0.05

q_circ    = IBM_challenge_ansatz(n_spins,depth,zeros(15))
f_circ    = chain(n_spins,[put(1=>X),put(2=>X)])


data     = JSON.parse(open("data/IBM-ansatz_depth"*string(depth)*"_"*string(n_spins)*"spin_dt"*string(dt)*"_adam_julia.dat","r"))


# Now for every timestep mesure the obs and save the values
fid_values = []


times  = data["times"]
params = data["parameters"]

state_list = []



## Now calculate the fidelities
for (t,time) in enumerate(times)
	p = params[t]

	println(p)
	dispatch!(q_circ,p)

	# Fidelity wrt exact state
	value = abs(dot(statevec(zero_state(n_spins) |> q_circ)',statevec(zero_state(n_spins) |> f_circ)))^2
	push!(fid_values,value)


end



## Save the data

res = Dict("overlaps"=>fid_values,"ansatz_reps"=>[depth],"spins"=> [n_spins],"dt"=>[dt],"times"=>times)
j_res = JSON.json(res)
open("data/FIDELITIES_IBM-ansatz_depth"*string(depth)*"_"*string(n_spins)*"spin_dt"*string(dt)*"_adam_julia.dat","w") do j
	write(j,j_res)
end

# Now plot the data

plt.plot(times,fid_values,linestyle="-", marker="o",linewidth=0.7)
plt.xlabel(L"t")
plt.ylabel(L"Overlap $|110 \rangle$")
plt.ylim(ymin=-0.05,ymax=1.05)
plt.title("pVQD dt="*string(dt))
#plt.yscale("log")

plt.gcf()
