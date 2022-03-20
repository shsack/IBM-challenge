# ### This scripts plot the overlap of YBE compression with |110> and compares it with Trotter

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

from trotter_functions import Heisenberg_Trotter_compressed

# # Import ansatze

# ##################### EXACT ########################

# # Function to simulate the system exactly

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

# ############## pVQD ##################################

def measure_overlap_target(target,ansatz,param_list):

    ovp_list = []
    
    for params in param_list:
        circ  = ansatz(3,params)
        t_wfn = CircuitStateFn(circ)
        t_ovp = np.abs((~target@t_wfn).eval())**2

        ovp_list.append(t_ovp)

    return ovp_list


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



# ######################################################

# Compute pVQD overlap

data_15        = json.load(open('compression_trial.dat'))

pvqd_params_15 = data_15["params"]
pvqd_times_15  = data_15["times"]
pvqd_ovps_15   = measure_overlap_target(One^One^Zero,Heisenberg_YBE_variational,pvqd_params_15)

# Compute Trotter overlap

probs_110_trott = []
t = Parameter("t")
for sim_t in ts:
    trott_qr = QuantumRegister(3)
    trott_qc = QuantumCircuit(trott_qr)
    trott_qc.x([1,2])
    
    # Append the Trotterization
    trott_step = Heisenberg_Trotter_compressed(num_qubits=3,trotter_steps=4,p=t,target_time=sim_t).to_instruction()
    trott_qc.append(trott_step, [trott_qr[0], trott_qr[1], trott_qr[2]])
    
    
    trott_wfn = CircuitStateFn(trott_qc)
    trott_ovp = np.abs((~initial_state@trott_wfn).eval())**2
    probs_110_trott.append(trott_ovp)


plt.plot(ts, probs_110_exact,linestyle="dashed",color="black",label="Exact")
plt.plot(ts,probs_110_trott,label="Trotter, 24 CNOTS",marker="o",color="C0")
plt.plot(pvqd_times_15,pvqd_ovps_15,label="YBE, 15 CNOTs",marker="^",color="C1")


plt.xlabel(r'$t$')
plt.ylabel(r'Probability of state $|110\rangle$')
plt.title(r'Evolution of state $|110\rangle$')
plt.legend()
plt.grid()
plt.show()

