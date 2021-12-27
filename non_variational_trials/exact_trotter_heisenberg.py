import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})  # enlarge matplotlib fonts

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing standard Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, IBMQ, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.opflow.state_fns             import CircuitStateFn

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity


from trotter_functions import *


# In this script we are going to analyse the Trotter decomposition at different order and at different times


###################### EXACT ########################
""
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




##################### TROTTER #######################
""
## Here we create the circuit for the Trotterisation


t = Parameter('t')
num_qubits = 3
trotter_steps = 15  


## For the simulation

target_wfn = One^One^Zero

probs_110_trott = []
for target_time in ts:

    ## Now create the circuit at different times
    Trot_qc = Heisenberg_Trotter_1st_ord_compressed(num_qubits,trotter_steps,t,target_time).to_instruction()
    #Trot_qc = Heisenberg_Trotter_1st_ord_YBE_4steps(num_qubits,trotter_steps,t,target_time).to_instruction()

    ## Create the circuit

    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    qc.x([1,2])
    qc.append(Trot_qc, [qr[0], qr[1], qr[2]])

    #print(qc.decompose().decompose().decompose())
    #exit()
    
    wfn = CircuitStateFn(qc)

    ovp = np.abs((~target_wfn@wfn).eval())**2

    probs_110_trott.append(ovp)



### now plot the result
plt.plot(ts, probs_110_exact,linestyle="dashed",color="black",label="Exact")
plt.plot(ts,probs_110_trott,label="Trotter n= "+str(trotter_steps))


plt.xlabel(r'$t$')
plt.ylabel(r'Probability of state $|110\rangle$')
plt.title(r'Evolution of state $|110\rangle$')
plt.legend()
plt.grid()
plt.show()

""

