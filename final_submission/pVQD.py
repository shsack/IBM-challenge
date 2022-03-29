import numpy as np 
import json
import functools
import itertools
import matplotlib.pyplot as plt 
from scipy   import  linalg as LA 


import qiskit
from qiskit                               import Aer, execute
from qiskit.quantum_info       import Pauli
from qiskit.utils                         import QuantumInstance
from qiskit.opflow             import PauliOp, SummedOp, CircuitSampler, StateFn
from qiskit.circuit                       import ParameterVector
from qiskit.opflow.evolutions             import Trotter, PauliTrotterEvolution

from qiskit.opflow.state_fns      import CircuitStateFn
from qiskit.opflow.expectations   import PauliExpectation, AerPauliExpectation
from qiskit.opflow.primitive_ops  import CircuitOp
from qiskit.opflow                import Z, I


# Useful functions

def projector_zero(n_qubits):
    # This function create the global projector |00...0><00...0|
    from qiskit.opflow            import Z, I

    prj_list = [0.5*(I+Z) for i in range(n_qubits)]
    prj = prj_list[0]

    for a in range(1,len(prj_list)):
        prj = prj^prj_list[a]

    return prj

def projector_zero_local(n_qubits):
    # This function creates the local version of the cost function 
    # proposed by Cerezo et al: https://www.nature.com/articles/s41467-021-21728-w
    from qiskit.opflow          import Z, I

    tot_prj = 0

    for k in range(n_qubits):
        prj_list = [I for i in range(n_qubits)]
        prj_list[k] = 0.5*(I+Z)
        prj = prj_list[0]

        for a in range(1,len(prj_list)):
            prj = prj^prj_list[a]

        tot_prj += prj

    tot_prj = (1/n_qubits)*tot_prj
    
    return tot_prj


def ei(i,n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]


def construct_total_circuit(U_dt,time_step,ansatz,params_vec,left,right):
    ## This function creates the circuit that will be used to evaluate overlap and its gradient

    l_circ  = ansatz.assign_parameters({params_vec: left})
    r_circ  = ansatz.assign_parameters({params_vec: right})
    zero_prj = StateFn(projector_zero(U_dt.num_qubits),is_measurement = True)
    state_wfn = zero_prj @ StateFn(r_circ.combine(U_dt)+ l_circ.inverse())

    return state_wfn


def compute_overlap_and_gradient(left,right,state_wfn,parameters,shift,expectator,sampler):

	nparameters = len(parameters)
	# build dictionary of parameters to values
	# {left[0]: parameters[0], .. ., right[0]: parameters[0] + shift[0], ...}
	# First create the dictionary for overlap
	values_dict = [dict(zip(right[:] + left[:], parameters.tolist() + (parameters + shift).tolist()))]
		

	# Then the values for the gradient
	for i in range(nparameters):
		values_dict.append(dict(zip(right[:] + left[:], parameters.tolist() + (parameters + shift + ei(i,nparameters)*np.pi/2.0).tolist())))
		values_dict.append(dict(zip(right[:] + left[:], parameters.tolist() + (parameters + shift - ei(i,nparameters)*np.pi/2.0).tolist())))

	# Now evaluate the circuits with the parameters assigned

	results = []

	for values in values_dict:
		sampled_op = sampler.convert(state_wfn,params=values)

		mean  = sampled_op.eval().real
		est_err = 0

		results.append([mean,est_err])

	E = np.zeros(2)
	g = np.zeros((nparameters,2))

	E[0],E[1] = results[0]

	for i in range(nparameters):
		rplus  = results[1+2*i]
		rminus = results[2+2*i]
		# G      = (Ep - Em)/2
		# var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
		g[i,:] = (rplus[0]-rminus[0])/2.0,np.sqrt(rplus[1]**2+rminus[1]**2)/2.0

	return E,g 

def adam_gradient(parameters,shift,count,m,v,g):
	## This function implements adam optimizer
	beta1 = 0.9
	beta2 = 0.999
	eps   = 1e-8
	alpha = [0.001 for i in range(len(parameters))]
	if count == 0:
		count = 1

	new_shift = [0 for i in range(len(parameters))]

	for i in range(len(parameters)):
		m[i] = beta1 * m[i] + (1 - beta1) * g[i]
		v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i],2)

		alpha[i] = alpha[i] * np.sqrt(1 - np.power(beta2,count)) / (1 - np.power(beta1,count))

		new_shift[i] = shift[i] + alpha[i]*(m[i]/(np.sqrt(v[i])+eps))

	return new_shift


def measure_overlap_pVQD(target,ansatz,num_qubits,trotter_steps,param_list):

    ovp_list = []
    
    for params in param_list:
        circ  = ansatz(num_qubits,trotter_steps,params)
        t_wfn = CircuitStateFn(circ)
        t_ovp = np.abs((~target@t_wfn).eval())**2

        ovp_list.append(t_ovp)

    return ovp_list

