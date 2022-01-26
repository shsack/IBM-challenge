## These file will contain all the ansatze used for variational quantum simulation

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


#=========================================

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

#==========================================

def challenge_ansatz(n_spins,depth,p):

	circ = QuantumCircuit(n_spins)
	count = 0

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















