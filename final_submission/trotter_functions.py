import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing basic Qiskit modules to create circuits
from qiskit                               import QuantumCircuit, QuantumRegister
from qiskit.providers.aer                 import QasmSimulator
from qiskit.circuit                       import Parameter

 
## Functions for 2 qubits rotations


def R_xx(t):
    
    '''
    Circuit for R_xx(t), as indicated in eq. 34 of arXiv:1907.03505v2
    
    Args:
        - t: parameter of the rotation
        
    Returns:
        A QuantumCircuit implementing the R_xx(t) rotation with name XX
    '''

    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * t, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    return XX_qc

def R_yy(t):
    
    '''
    Circuit for R_yy(t), as indicated in eq. 33 of arXiv:1907.03505v2
    
    Args:
        - t: parameter of the rotation
        
    Returns:
        A QuantumCircuit implementing the R_xx(t) rotation with name YY
    '''

    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')
    
    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * t, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    return YY_qc


def R_zz(t):
    
    '''
    Circuit for R_zz(t), as indicated in eq. 32 of arXiv:1907.03505v2
    
    Args:
        - t: parameter of the rotation
        
    Returns:
        A QuantumCircuit implementing the R_xx(t) rotation with name ZZ
    '''
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * t, 1)
    ZZ_qc.cnot(0,1)

    return ZZ_qc

def R_xyz(t):
    
    '''
    Circuit implementing the optimal gate decomposition of e^(-it(XX+YY+ZZ))
    as indicated in Fig. 4b of arXiv:1907.03505v2
    
    Args:
        - t: parameter of the rotation
        
    Returns:
        A QuantumCircuit implementing the R_{xx+yy+zz}(t) 3 CNOTs rotation with name XYZ
    '''
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
    
    '''
    Circuit implementing the optimal gate decomposition of e^(-it(XX+YY+ZZ))
    as indicated in Fig. 4b of arXiv:1907.03505v2,
    now every gate depending on the rotation angle 't' is a parameterized gate
    
    Args:
        - p: array of 3 parameters
        
    Returns:
        A QuantumCircuit implementing the parameterized R_{xx+yy+zz}(p0,p1,p2) rotation with name XYZ-var
    '''
    
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

def Heisenberg_Trotter(num_qubits,trotter_steps,t,target_time):
    
    '''
    Circuit implementing Trotterization of the time evolutiom operator for the XXX Heisenberg
    model on num_qubits using the circuit indicated in Fig. 4a of arXiv:1907.03505v2.
    Every  Trotter step requires 6*(num_qubits-1) CNOTs
    
    Args:
        - num_qubits: int, number of qubits of the system
        - trotter_steps: the number of trotter steps n to implement
        - t: Qiskit Parameter object, will be binded to dt
        - target_time: the simulation time we are targeting
        
    Returns:
        A QuantumCircuit implementing the Trotterization of the time evolutiom operator for the XXX Heisenberg
        model
    '''
    
    # Given a target time and a number of Trotter steps, every step will evolve the 
    # circuit for a time step dt = target_time/trotter_steps
    dt = target_time/trotter_steps

    XX = R_xx(t).to_instruction()
    YY = R_yy(t).to_instruction()
    ZZ = R_zz(t).to_instruction()
    
    
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

    # Evaluate simulation at target_time meaning each trotter step evolves target_time/trotter_steps in time
    qc = qc.bind_parameters({t: dt})

    return qc


def Heisenberg_Trotter_compressed(num_qubits,trotter_steps,t,target_time):
    
    '''
    Circuit implementing Trotterization of the time evolutiom operator for the XXX Heisenberg
    model on num_qubits using the circuit indicated in Fig. 4b of arXiv:1907.03505v2.
    Every  Trotter step requires 3*(num_qubits-1) CNOTs
    
    Args:
        - num_qubits: int, number of qubits of the system
        - trotter_steps: the number of trotter steps n to implement
        - t: Qiskit Parameter object, will be binded to dt
        - target_time: the simulation time we are targeting
        
    Returns:
        A QuantumCircuit implementing the Trotterization of the time evolutiom operator for the XXX Heisenberg
        model
    '''

    # Given a target time and a number of Trotter steps, every step will evolve the 
    # circuit for a time step dt = target_time/trotter_steps
    dt = target_time/trotter_steps

    XYZ = R_xyz(t).to_instruction()
    
    
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

    # Evaluate simulation at target_time meaning each trotter step evolves target_time/trotter_steps in time
    qc = qc.bind_parameters({t: dt})

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


def Heisenberg_YBE_variational(p):

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
