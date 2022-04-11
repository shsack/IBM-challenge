# # These file will contain all the ansatze used for variational quantum simulation
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


# =========================================

def hweff_ansatz(n_spins,depth,p):
	
	circuit = QuantumCircuit(n_spins)
	count = 0

	for j in range(depth):

		if(j%2 == 0):
			# Rx - Rzz block
			for i in range(n_spins):
				circuit.rx(p[count],i)
				count = count +1

			circuit.barrier()

			for i in range(n_spins-1):
				circuit.rzz(p[count],i,i+1)
				count = count +1

			circuit.barrier()

		if(j%2 == 1):
			for i in range(n_spins):
				circuit.ry(p[count],i)
				count = count +1

			circuit.barrier()

			for i in range(n_spins-1):
				circuit.rzz(p[count],i,i+1)
				count = count +1

			circuit.barrier()

	# Final block to close the ansatz
	if (depth%2 == 1):
		for i in range(n_spins):
				circuit.ry(p[count],i)
				count = count +1
	if (depth%2 == 0):
		for i in range(n_spins):
				circuit.rx(p[count],i)
				count = count +1

	return circuit

# ==========================================

def challenge_ansatz(n_spins,depth,p):

	circ = QuantumCircuit(n_spins)
	count = 0

	circ.x(2)
	circ.x(1)

	for d in range(depth):
		for i in range(n_spins-1):
			circ.cx(i,i+1)
			circ.rx(p[count],i)
			circ.rx(-np.pi/2,i)
			circ.h(i)
			circ.rz(p[count+1],i+1)

			circ.cx(i,i+1)
			circ.h(i)
			circ.rz(p[count+2],i+1)

			circ.cx(i,i+1)
			circ.rx(np.pi/2,i)
			circ.rx(-np.pi/2,i+1)

			count += 3

	return circ


def Heisenberg_YBE_variational(n_spins,depth,p):

    '''
    Circuit implementing the YBE compression of the time evolutiom operator for the XXX Heisenberg model.
    The YBE compression for Trotterized evolution was proposed in ArXiv:2112.01690.
    The circuit is for 3 qubits and 4 compressed trotter step, for a total of 15 CNOTs and 15 parameters
    
    Args:
        
        - p: an array of 15 real parameters
        
    Returns:
        A QuantumCircuit implementing the Trotterization of the time evolutiom operator for the XXX Heisenberg
        model
    '''

    # Initialise the 3 qubit circuit
    circ = QuantumCircuit(3)
    count = 0

    circ.x([1,2])
    
    # Circuit implementing the optimal gate decomposition of e^(-it(XX+YY+ZZ))
    # as indicated in Fig. 4b of arXiv:1907.03505v2,
    # now every gate depending on the rotation angle 't' is a parameterized gate.

    # This means 3 parameters per e^(-it(XX+YY+ZZ)) decomposition

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

    # Apply the gate 5 times on qubits (1,2) and (0,1), alternatively.

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















