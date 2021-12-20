import numpy as np
import matplotlib.pyplot as plt

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

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity


def R_xx(t):
	# This function returns the circuit for R_xx(t)

	XX_qr = QuantumRegister(2)
	XX_qc = QuantumCircuit(XX_qr, name='XX')
	
	XX_qc.ry(np.pi/2,[0,1])
	XX_qc.cnot(0,1)
	XX_qc.rz(2 * t, 1)
	XX_qc.cnot(0,1)
	XX_qc.ry(-np.pi/2,[0,1])

	return XX_qc

def R_yy(t):

	YY_qr = QuantumRegister(2)
	YY_qc = QuantumCircuit(YY_qr, name='YY')
	
	YY_qc.rx(np.pi/2,[0,1])
	YY_qc.cnot(0,1)
	YY_qc.rz(2 * t, 1)
	YY_qc.cnot(0,1)
	YY_qc.rx(-np.pi/2,[0,1])

	return YY_qc



def R_zz(t):

	ZZ_qr = QuantumRegister(2)
	ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

	ZZ_qc.cnot(0,1)
	ZZ_qc.rz(2 * t, 1)
	ZZ_qc.cnot(0,1)

	return ZZ_qc

def R_xyz(t):
	XYZ_qr = QuantumRegister(2)
	XYZ_qc = QuantumCircuit(XYZ_qr, name='XYZ')
	
	XYZ_qc.cnot(0,1)
	XYZ_qc.rx(2*t-np.pi/2, 0)
	XYZ_qc.rz(2 * t, 1)
	XYZ_qc.h(0)
	XYZ_qc.cnot(0,1)
	XYZ_qc.h(0)
	XYZ_qc.rz(-2 * t, 1)
	XYZ_qc.cnot(0,1)
	XYZ_qc.rx(np.pi/2,0)
	XYZ_qc.rx(-np.pi/2,1)

	return XYZ_qc

def Heisenberg_Trotter_1st_ord(num_qubits,trotter_steps,p,target_time):

	dt = target_time/trotter_steps

	XX = R_xx(p).to_instruction()
	YY = R_yy(p).to_instruction()
	ZZ = R_zz(p).to_instruction()
	
	
	# Combine subcircuits into a single multiqubit gate representing a single trotter step

	
	Trot_qr = QuantumRegister(num_qubits)
	Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
	
	for i in range(0, num_qubits - 1):
	    Trot_qc.append(ZZ, [Trot_qr[i], Trot_qr[i+1]])
	    Trot_qc.append(YY, [Trot_qr[i], Trot_qr[i+1]])
	    Trot_qc.append(XX, [Trot_qr[i], Trot_qr[i+1]])
	
	# Now repeat the circuit #trotter_reps

	Trot_gate = Trot_qc.to_instruction()


	# Initialize quantum circuit for 3 qubits
	qr = QuantumRegister(num_qubits)
	qc = QuantumCircuit(qr)


	# Simulate time evolution under H_heis3 Hamiltonian
	for _ in range(trotter_steps):
  	  qc.append(Trot_gate, [qr[0], qr[1], qr[2]])
  	  qc.barrier()

	# Evaluate simulation at target_time meaning each trotter step evolves pi/trotter_steps in time
	qc = qc.bind_parameters({p: dt})

	return qc


def Heisenberg_Trotter_1st_ord_compressed(num_qubits,trotter_steps,p,target_time):

	dt = target_time/trotter_steps

	XYZ = R_xyz(p).to_instruction()
	
	
	# Combine subcircuits into a single multiqubit gate representing a single trotter step

	
	Trot_qr = QuantumRegister(num_qubits)
	Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
	
	for i in range(0, num_qubits - 1):
	    Trot_qc.append(XYZ, [Trot_qr[i], Trot_qr[i+1]])
	
	# Now repeat the circuit #trotter_reps

	Trot_gate = Trot_qc.to_instruction()


	# Initialize quantum circuit for 3 qubits
	qr = QuantumRegister(num_qubits)
	qc = QuantumCircuit(qr)


	# Simulate time evolution under H_heis3 Hamiltonian
	for _ in range(trotter_steps):
  	  qc.append(Trot_gate, [qr[0], qr[1], qr[2]])
  	  qc.barrier()

	# Evaluate simulation at target_time meaning each trotter step evolves pi/trotter_steps in time
	qc = qc.bind_parameters({p: dt})

	return qc

def Heisenberg_Trotter_1st_ord_YBE_4steps(num_qubits,trotter_steps,p,target_time):

	dt = target_time/trotter_steps
	
	
	# Combine subcircuits into a single multiqubit gate representing a single trotter step

	
	Trot_qr = QuantumRegister(num_qubits)
	Trot_qc = QuantumCircuit(Trot_qr, name='Trot')
	
	Trot_qc.append(R_xyz(2*p).to_instruction(), [Trot_qr[0], Trot_qr[1]])
	Trot_qc.append(R_xyz(p).to_instruction(), [Trot_qr[1], Trot_qr[2]])
	Trot_qc.append(R_xyz(3*p).to_instruction(), [Trot_qr[0], Trot_qr[1]])
	Trot_qc.append(R_xyz(p).to_instruction(), [Trot_qr[1], Trot_qr[2]])
	Trot_qc.append(R_xyz(p).to_instruction(), [Trot_qr[0], Trot_qr[1]])
	# Now repeat the circuit #trotter_reps

	Trot_gate = Trot_qc.to_instruction()


	# Initialize quantum circuit for 3 qubits
	qr = QuantumRegister(num_qubits)
	qc = QuantumCircuit(qr)


	# Simulate time evolution under H_heis3 Hamiltonian
	
	qc.append(Trot_gate, [qr[0], qr[1], qr[2]])
	qc.barrier()

	# Evaluate simulation at target_time meaning each trotter step evolves pi/trotter_steps in time
	qc = qc.bind_parameters({p: dt})

	return qc


