import numpy as np
import matplotlib.pyplot as plt
import json
plt.rcParams.update({'font.size': 12})  # enlarge matplotlib fonts

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing standard Qiskit modules
from qiskit                   import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile
from qiskit.providers.aer     import QasmSimulator
from qiskit.tools.monitor     import job_monitor
from qiskit.circuit           import Parameter
from qiskit.quantum_info      import Statevector
from qiskit.opflow.state_fns  import CircuitStateFn

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info                  import state_fidelity



## Import ansatze

from ansatze import *

###################### EXACT ########################

## Function to simulate the system exactly

def H_heis3():
    # Interactions (I is the identity matrix; X, Y, and Z are Pauli matricies; ^ is a tensor product)
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    
    # Sum interactions
    H = XXs + YYs + ZZs
    
    # Return Hamiltonian
    return H

# Returns the matrix representation of U_heis3(t) for a given time t assuming an XXX Heisenberg Hamiltonian for 3 spins-1/2 particles in a line
def U_heis3(t):
    # Compute XXX Hamiltonian for 3 spins in a line
    H = H_heis3()
    
    # Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
    return (t * H).exp_i()


ts = np.linspace(0, np.pi, 100)

# Define initial state |110>
initial_state = One^One^Zero

probs_110_exact = [np.abs((~initial_state @ U_heis3(float(t)) @ initial_state).eval())**2 for t in ts]

############### pVQD ##################################

def measure_overlap_target(target,ansatz,param_list):

    ovp_list = []
    
    for params in param_list:
        circ  = ansatz(3,4,params)
        t_wfn = CircuitStateFn(circ)
        t_ovp = np.abs((~target@t_wfn).eval())**2

        ovp_list.append(t_ovp)

    return ovp_list





#######################################################

# Compute and plot

data   = json.load(open('data/trial_results.dat'))

pvqd_params = data["params"]
pvqd_times  = data["times"]
pvqd_ovps   = measure_overlap_target(One^One^Zero,Heisenberg_YBE_variational,pvqd_params)

print(pvqd_params[-1])


plt.plot(ts, probs_110_exact,linestyle="dashed",color="black",label="Exact")
plt.plot(pvqd_times,pvqd_ovps,label="pVQD",marker="o",color="C0")


plt.xlabel(r'$t$')
plt.ylabel(r'Probability of state $|110\rangle$')
plt.title(r'Evolution of state $|110\rangle$')
plt.legend()
plt.grid()
plt.show()

