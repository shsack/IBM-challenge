import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing basic Qiskit modules to create circuits
from qiskit                               import QuantumCircuit, QuantumRegister
from qiskit.providers.aer                 import QasmSimulator
from qiskit.circuit                       import Parameter


# +
## We start creating the functions for the circuit corresponding to useful 2-qubit gates
# -

def R_xx(t):
	# This function returns the circuit for R_xx(t), as indicated in eq. 34 of arXiv:1907.03505v2

	XX_qr = QuantumRegister(2)
	XX_qc = QuantumCircuit(XX_qr, name='XX')
	
	XX_qc.ry(np.pi/2,[0,1])
	XX_qc.cnot(0,1)
	XX_qc.rz(2 * t, 1)
	XX_qc.cnot(0,1)
	XX_qc.ry(-np.pi/2,[0,1])

	return XX_qc

def R_yy(t):
    # This function returns the circuit for R_yy(t), as indicated in eq. 33 of arXiv:1907.03505v2

	YY_qr = QuantumRegister(2)
	YY_qc = QuantumCircuit(YY_qr, name='YY')
	
	YY_qc.rx(np.pi/2,[0,1])
	YY_qc.cnot(0,1)
	YY_qc.rz(2 * t, 1)
	YY_qc.cnot(0,1)
	YY_qc.rx(-np.pi/2,[0,1])

	return YY_qc


def R_zz(t):
    # This function returns the circuit for R_zz(t), as indicated in eq. 32 of arXiv:1907.03505v2

	ZZ_qr = QuantumRegister(2)
	ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

	ZZ_qc.cnot(0,1)
	ZZ_qc.rz(2 * t, 1)
	ZZ_qc.cnot(0,1)

	return ZZ_qc

def R_xyz(t):
    # This function returns optimal gate decomposition of e^(-it(XX+YY+ZZ)), 
    # as indicated in Fig. 4b of arXiv:1907.03505v2
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

def R_xyz_var(p):
    
    XYZ_qr = QuantumRegister(2)
    XYZ_qc = QuantumCircuit(XYZ_qr, name='XYZ-var')
    
    XYZ_qc.cnot(0,1)
    XYZ_qc.rx(p[0], 0)
    XYZ_qc.rx(-np.pi/2, 0)
    XYZ_qc.rz(p[1], 1)
    XYZ_qc.h(0)
    XYZ_qc.cnot(0,1)
    XYZ_qc.h(0)
    XYZ_qc.rz(p[2], 1)
    XYZ_qc.cnot(0,1)
    XYZ_qc.rx(np.pi/2,0)
    XYZ_qc.rx(-np.pi/2,1)

    return XYZ_qc


# +
## Now we create the Trotter circuit for the XXX Heisenberg model
# -

def Heisenberg_Trotter(num_qubits,trotter_steps,p,target_time):

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


def Heisenberg_Trotter_compressed(num_qubits,trotter_steps,p,target_time):

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


def Heisenberg_Trotter_variational(num_qubits,trotter_steps,p):

	qr = QuantumRegister(num_qubits)
	qc = QuantumCircuit(qr)
	count = 0

	qc.rx(np.pi,[1,2])

	for d in range(trotter_steps):
		for i in range(num_qubits-1):
			qc.append(R_xyz_var(p[count:count+3]).to_instruction(),[qr[i],qr[i+1]])
			count += 3

	return qc


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
