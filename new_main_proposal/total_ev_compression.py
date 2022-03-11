## This script compress with YBE the entire time evolution
## Showing the equivalence between 15 and 24 CNOTs at all times

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts
import json
# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

## Import functions from Qiskit
from qiskit                     import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile, Aer
from qiskit.providers.aer       import QasmSimulator
from qiskit.tools.monitor       import job_monitor
from qiskit.circuit             import Parameter, ParameterVector
from qiskit.quantum_info        import Statevector, Pauli
from qiskit.opflow.state_fns    import CircuitStateFn
from qiskit.opflow.expectations import PauliExpectation
from qiskit.utils               import QuantumInstance
from qiskit.opflow              import PauliOp, SummedOp, CircuitSampler, StateFn

# Import state tomography modules

## In the "trotter_function.py" file are contained all the functions useful for a perfect Trotter simulation
from trotter_functions import Heisenberg_Trotter_compressed

from pVQD              import adam_gradient, projector_zero, ei

def Heisenberg_YBE_variational(num_qubits,p):

    circ = QuantumCircuit(num_qubits)
    count = 0
    
    def XYZ_variational(circ,i,j,params):
        circ.cx(i,j)
        circ.rx(params[0],i)
        circ.rx(-np.pi/2,i)
        circ.h(i)
        circ.rz(params[1],j)

        circ.cx(i,j)
        circ.h(i)
        circ.rz(params[2],j)

        circ.cx(i,j)
        circ.rx(np.pi/2,i)
        circ.rx(-np.pi/2,j)

    circ.rx(np.pi,[1,2])
    
    XYZ_variational(circ,1,2,p[count:count+3])
    count += 3
    XYZ_variational(circ,0,1,p[count:count+3])
    count += 3
    XYZ_variational(circ,1,2,p[count:count+3])
    count += 3
    XYZ_variational(circ,0,1,p[count:count+3])
    count += 3
    XYZ_variational(circ,1,2,p[count:count+3])
    count += 3

    return circ

def overlap_and_gradient(right,state_wfn,parameters,expectator,sampler):

    nparameters = len(parameters)
    # build dictionary of parameters to values
    # {left[0]: parameters[0], .. ., right[0]: parameters[0] + shift[0], ...}
    # First create the dictionary for overlap
    values_dict = [dict(zip(right[:], parameters.tolist()))]


    # Then the values for the gradient
    for i in range(nparameters):
        values_dict.append(dict(zip(right[:] , (parameters  + ei(i,nparameters)*np.pi/2.0).tolist())))
        values_dict.append(dict(zip(right[:] , (parameters  - ei(i,nparameters)*np.pi/2.0).tolist())))

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

######################################################################

backend     = Aer.get_backend('statevector_simulator')
instance    = QuantumInstance(backend=backend)
expectation = PauliExpectation()
sampler     = CircuitSampler(instance)


ts             = np.linspace(0, np.pi, 60)
ybe_params_vec = ParameterVector('p',15)
right          = ParameterVector('r', 15)
ybe_parameters = np.zeros(15)
num_parameters = len(ybe_parameters)
t              = Parameter("t")

opt_steps      = 600
opt_ths        = 0.9999
zero_prj       = StateFn(projector_zero(3),is_measurement = True)


## Store the parameters
ybe_params = []
#ybe_params.append(list(ybe_parameters))

## Repeat the procedure for every time step

for (t_step,sim_t) in enumerate(ts):

    print("Step: "+str(t_step))
    ## Create the optimization circuit
    l_ansatz   = QuantumCircuit(3)
    l_ansatz.x([1,2])
    l_ansatz   = l_ansatz.compose(Heisenberg_Trotter_compressed(num_qubits=3,trotter_steps=4,p=t,target_time=sim_t))
    #print(l_ansatz)
    r_ansatz   = Heisenberg_YBE_variational(3,ybe_params_vec)
    r_circ     = r_ansatz.assign_parameters({ybe_params_vec: right})

    total_circ = r_circ+l_ansatz.inverse()
    state_wfn  = expectation.convert(zero_prj @ StateFn(total_circ))


    # Initialise step-quantities
    count          = 0
    overlap        = [0.01,0]
    max_ovp        = 0.01
    new_parameters = ybe_parameters

    # Initialise quantities for the Adam optimiser
    m = np.zeros(num_parameters)
    v = np.zeros(num_parameters)




    while overlap[0] < opt_ths and count < opt_steps:
        print("Optimizing step:",count+1)
        count = count +1 

        ## Measure energy and gradient

        E,g = overlap_and_gradient(right,state_wfn,new_parameters,expectation,sampler)

        print('Overlap',E[0])
        overlap = E


        meas_grad = np.asarray(g[:,0])
        new_parameters = np.asarray(adam_gradient(new_parameters,new_parameters,count,m,v,meas_grad))


        if E[0] > max_ovp:
            max_ovp        = E[0]
            ybe_parameters = new_parameters
    

    ybe_params.append(list(ybe_parameters))
    # Update parameters

    print('\n---------------------------------- \n')
    print("New overlap: " ,max_ovp)




## Finally, save the data


log_data = {}
    
log_data['times']       = list(ts)
log_data['params']      = list(ybe_params)
        
filename = "compression_trial.dat"
json.dump(log_data, open( filename,'w+'))





