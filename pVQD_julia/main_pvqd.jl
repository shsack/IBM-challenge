# This is the main script for pVQD with Julia

#=  Algorithm flow:
	- Give a set of parameters 
	- Give an initial shift (not so important for now)
	- Create the UTU^dag circuit
	- Measure P_0
	- Optimize parameters
	- Proceed to next step
=#

using Yao,YaoExtensions,PyCall,YaoPlots, JSON, Statistics, Yao.AD, Compose
using QuAlgorithmZoo: Adam, update!


# include all the useful functions for VpVQD
include("pvqd_functions.jl")
import PyPlot; const plt = PyPlot

# Now import useful subroutines from python
pushfirst!(PyVector(pyimport("sys")."path"), "")
#sub = pyimport("_sub")


## Now we can declare all the important quantities for the system
begin
	n_qubits   = 3

	J          = 0.25
	Jx         = 1
	Jy         = 1
	Jz         = 1

	B          = 1.0
	dt         = 0.05
	n_dt       = 60
	## For ansatz
	depth      = 1
end



## Create the ansatz for the pVQD and Trotter

l_circ     = IBM_challenge_ansatz(n_qubits,depth,zeros(15))
r_circ     = IBM_challenge_ansatz(n_qubits,depth,zeros(15))
trott_circ = heisenberg_trotter_step(n_qubits,dt,Jx,Jy,Jz)

#Observable (global or local)

g_obs = projector_zero(n_qubits)

#Uncomment to Plot the ansatz circuit
#YaoPlots.plot(l_circ) |> SVG("ansatz_circ.svg")

### Main routine

## Define quantities for algorithms


t_steps      = [dt*i for i in 0:n_dt]
opt_steps    = 200
n_params     = nparameters(l_circ)
final_params = zeros(n_params)


## Container for quantities

initial_fidelities = []
final_fidelities   = []
parameters_list    = []
push!(parameters_list,final_params)
global high_fid = 0	



println("Starting the p-VQD")

for (k,t_step) in enumerate(t_steps)

	
	println("----------------------------------------------------")
	println("Time: $t_step")

	# Create the circuit with initial parameters (or final from previous run)
	# For semplicity, we impose dw_0 = 0 

	dispatch!(l_circ,final_params)
	dispatch!(r_circ,final_params)
	# modify trotter for time dependent hamiltonian
	tot_circ = chain(l_circ,NoParams(trott_circ),NoParams(r_circ')) # Creation of total circuit
	
	#params   = parameters(tot_circ)

	global final_params = parameters(tot_circ)
	global high_fid = 0

	# Calculate initial fidelity and load it into a vector
	local init_fid = real.(expect(g_obs, zero_state(n_qubits) |> tot_circ))
	push!(initial_fidelities,init_fid)


	# Now optimise the circuit
	optimizer    = Adam(lr=0.01) # initialise the optimiser

	for i = 1:opt_steps
		## `expect'` gives the gradient of an observable.

		#params = parameters(tot_circ)
		grad_input, grad_params = expect'(g_obs, zero_state(n_qubits) => tot_circ)

		## feed the gradients into the circuit.
		## ADAM
		dispatch!(tot_circ, update!(final_params, -grad_params, optimizer))
		## SGD
		#final_params += grad_params
		#dispatch!(tot_circ, final_params)
	
		# Save the new fidelity
		local new_fid = real.(expect(g_obs, zero_state(n_qubits) |> tot_circ))

		println("Step $i, fidelity = $new_fid")

		if new_fid > high_fid
			#global final_params = parameters(tot_circ)
			global high_fid = new_fid
		end
	end

	println("Final parameters: $final_params")

	# Now save the final results into a vector
	push!(final_fidelities,high_fid)
	push!(parameters_list,final_params)

	#### Here you can measure observables on the final state...
end



### Now save the data on external file

save_data = true


if save_data
	res = Dict("parameters"=>parameters_list,"initial_fidelities"=>initial_fidelities,"final_fidelities"=>[final_fidelities],"ansatz_reps"=>[depth],"spins"=> [n_qubits],"dt"=>[dt],"times"=>t_steps)
	j_res = JSON.json(res)
	open("data/IBM-ansatz_depth"*string(depth)*"_"*string(n_qubits)*"spin_dt"*string(dt)*"_adam_julia.dat","w") do j
		write(j,j_res)
	end
end




