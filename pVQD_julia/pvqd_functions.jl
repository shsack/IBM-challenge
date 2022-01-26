## pVQD meets Julia! 

# This file will contain all the useful functions for pVQD and variational circuit in general

###################################################################################################
## GATES

function Rzz(n,i,j,theta)
	circ = chain(n,[cnot(i,j),
		            put(j=>Rz(theta)),
		            cnot(i,j)
		            ])
	return circ
end

function Rxx(n,i,j,theta)
	circ = chain(n,[put(i=>Ry(pi/2)),
		            cnot(i,j),
		            put(j=>Rz(theta)),
		            cnot(i,j),
		            put(i=>Ry(-pi/2))
		            ])
	return circ
end


function Ryy(n,i,j,theta)
	circ = chain(n,[put(i=>Rx(pi/2)),
		            put(j=>Rx(pi/2)),
		            cnot(i,j),
		            put(j=>Rz(theta)),
		            cnot(i,j),
		            put(i=>Rx(-pi/2)),
		            put(j=>Rx(-pi/2))
		            ])
	return circ
end

function Rxyz(n,i,j,theta)

	circ = chain(n,[cnot(i,j),
		            put(i=>Rx(theta[1])),
		            put(i=>NoParams(Rx(pi/2)')),
		            put(j=>Rz(theta[2])),
		            put(i=>H),
		            cnot(i,j),
		            put(i=>H),
		            put(j=>Rz(theta[3])'),
		            cnot(i,j),
		            put(i=>NoParams(Rx(pi/2))),
		            put(j=>NoParams(Rx(pi/2)'))
		            ])
	return circ 

end


###################################################################################################
# ANSATZE



function IBM_challenge_ansatz(n,depth,params)

	count = 1

	circ = chain(n,[put(1=>X),put(2=>X)])

	## Push the gate in order
	push!(circ,Rxyz(n,1,2,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,2,3,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,1,2,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,2,3,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,1,2,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,2,3,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,1,2,params[count:count+2]))
	count += 3

	push!(circ,Rxyz(n,2,3,params[count:count+2]))
	count += 3


	return circ
end


###################################################################################################
# PROJECTORS AND OBSERVABLES

function projector_zero(n)
	prj    = kron([0.5*(I2+Z) for i in 1:n]...)
	#id 	   = kron([I2 for i in 1:n]...)
	return  prj
end	

function projector_site(n,i)
	op_list = [0.5*(I2+Z) for i in 1:n]
	op_list[i] = 0.5*(I2-Z)

	prj = kron(op_list...)

	return prj
end

function spin_x(n,site)
	op_list = [I2 for i in 1:n]
	op_list[site] = X 

	obs = kron(op_list...)

	return obs
end	

function spin_y(n,site)
	op_list = [I2 for i in 1:n]
	op_list[site] = Y 

	obs = kron(op_list...)

	return obs
end	

function spin_z(n,site)
	op_list = []

	for i in 1:n
		if i==site 
			push!(op_list,Z)
		else 
			push!(op_list,I2)
		end
	end
	
	obs = kron(op_list...)

	return obs
end	

###################################################################################################
# TROTTER

function ising_trotter_step(n,dt,J,B)

	circ = chain(n, put(i=> I2) for i in 1:n)
	# Rx layer
	for i in 1:n
		#push!(circ, Rx(B*dt))
		push!(circ,chain(n,put(i=>Rx(2*B*dt))))
		
	end

	# Rzz layer
	for j in 1:n-1
		push!(circ,Rzz(n,j,j+1,2*J*dt))	
	end

	return circ
end

function heisenberg_trotter_step(n,dt,Jx,Jy,Jz)

	circ = chain(n)

	for j in 1:n-1
		push!(circ,Rxx(n,j,j+1,2*Jz*dt))	
	end

	for j in 1:n-1
		push!(circ,Ryy(n,j,j+1,2*Jz*dt))	
	end

	for j in 1:n-1
		push!(circ,Rzz(n,j,j+1,2*Jz*dt))	
	end

	return circ 
end 


## Functions to create Trotter circuit with multiple steps

function trotter_ising(n,t_step,t,J,B)
	circ = chain(n)
	if t_step == 0
		return circ

	else
		for i in 1:t_step
			for j in 1:n
				push!(circ,put(j=>Rx(2*B*t/t_step)))
			end
			for j in 1:n-1
				push!(circ,Rzz(n,j,j+1,2*J*t/t_step))
			end
		end

		return circ
	end
end


function trotter_heisenberg(n,t_step,t,Jx,Jy,Jz)
	circ = chain(n)
	if t_step == 0
		return circ

	else
		for i in 1:t_step
			push!(circ,heisenberg_trotter_step(n,t/t_step,Jx,Jy,Jz))
		end

		return circ
	end
end









