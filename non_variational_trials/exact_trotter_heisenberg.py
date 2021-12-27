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
num_qubits    = 3
trotter_steps = 4
ybe_steps     = 4


## For the simulation

target_wfn = One^One^Zero

probs_110_trott = []
probs_110_ybe   = []

for target_time in ts:

    ## Now create the circuit at different times
    Trot_qc  = Heisenberg_Trotter_1st_ord_compressed(num_qubits,trotter_steps,t,target_time).to_instruction()
    ybe_qc   = Heisenberg_Trotter_1st_ord_YBE_4steps(num_qubits,ybe_steps,target_time).to_instruction()


    ## Create the Trotter circuit

    t_qr = QuantumRegister(num_qubits)
    t_qc = QuantumCircuit(t_qr)

    t_qc.x([1,2])
    t_qc.append(Trot_qc, [t_qr[0], t_qr[1], t_qr[2]])

    ## Create the YBE compressed circuit

    y_qr = QuantumRegister(num_qubits)
    y_qc = QuantumCircuit(y_qr)
    y_qc.x([1,2])
    y_qc.append(ybe_qc, [y_qr[0], y_qr[1], y_qr[2]])

    if target_time == 0:
        print(t_qc.decompose().decompose())
        print(y_qc.decompose().decompose())


    ## Create the wfns    
    t_wfn = CircuitStateFn(t_qc)
    y_wfn = CircuitStateFn(y_qc)

    ## Calculate overlaps
    t_ovp = np.abs((~target_wfn@t_wfn).eval())**2
    y_ovp = np.abs((~target_wfn@y_wfn).eval())**2

    probs_110_trott.append(t_ovp)
    probs_110_ybe.append(y_ovp)



### now plot the result
plt.plot(ts, probs_110_exact,linestyle="dashed",color="black",label="Exact")
plt.plot(ts,probs_110_trott,label="Trotter n= "+str(trotter_steps),marker="o",color="C0")
plt.plot(ts,probs_110_ybe,label="YBE compression",marker="^",color="C1")


plt.xlabel(r'$t$')
plt.ylabel(r'Probability of state $|110\rangle$')
plt.title(r'Evolution of state $|110\rangle$')
plt.legend()
plt.grid()
plt.show()

""

