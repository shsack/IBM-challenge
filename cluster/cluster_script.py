## Import libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts
import pickle

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
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info                  import state_fidelity


## Mitiq libraries
from mitiq import zne
from qiskit.result import Result
from qiskit.result.models import ExperimentResult
from qiskit.result.models import ExperimentResultData
from qiskit.result.models import QobjExperimentHeader



## Now the main part of the script

# Create the ansatz

def Heisenberg_YBE_variational(num_qubits,p):

    circ  = QuantumCircuit(num_qubits)
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


# And take the parameters from pVQD

pvqd_opt_params = [0.6382017062070897,
0.5999999987484098,
0.6382017062066773,
3.0088034895496003,
-3.0869200336945677,
0.4709531470409451,
2.163149581322057,
3.480816125849344,
-2.0741264452466974,
1.2330206913091548,
3.1275100711382064,
1.593744340473751,
6.107319841483039,
3.0177717815840808,
-3.24901805128811]


## Create the circuit
# Define the final circuit that is used to compute the fidelity 
fqr = QuantumRegister(7)
fqc = QuantumCircuit(fqr)
#fqc.rx(np.pi, [3, 5]) # Cannot use X gate due to a bug in mitq, rx(pi) does the same thing
fqc.id([0, 1, 2, 4, 6]) # Need to put identities since mitq cannot handle unused qubits
fqc.append(Heisenberg_YBE_variational(3,pvqd_opt_params), [fqr[1], fqr[3], fqr[5]])


## Info for IBM
#IBMQ.save_account('MY_API_TOKEN')
IBMQ.enable_account('MY_API_TOKEN')
#IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-community', group='ibmquantumawards', project='open-science-22')
jakarta = provider.get_backend('ibmq_jakarta')
# Simulated backend based on ibmq_jakarta's device noise profile
sim_noisy_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))

shots = 8192
#backend = sim_noisy_jakarta
backend = jakarta

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs):
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

def zne_results(tomo_circs, backend, optimization_level, zne_order, shots,job_list):

    # This function runs the tomography circuits and unrolls the gates to increase the noise level
    # The counts that are obtained for the differnt noise levels are then extrapolated to the zero-noise level

    zne_result_list = []
    scale_factors = [1.0, 2.0, 3.0]
    # Loop over the tomography circuits
    for circ in tomo_circs:

        print("\n\n############### Running the "+str(circ.name)+" circuit   ############### ")
        job_list[str(circ.name)] = []
        # Unfold the tomography circuit by a scale factor and evaluate them 
        noise_scaled_circuits = [zne.scaling.fold_global(circ, s) for s in scale_factors]  
        #result_list = [execute(circ_noise, backend=backend, optimization_level=optimization_level, shots=shots).result() for circ_noise in noise_scaled_circuits]

        result_list = []
        pickle_file = "./hw_data/"+str(circ.name)
        pickle_data = {}
        for circ_noise in noise_scaled_circuits:
            job = execute(circ_noise, backend=backend, optimization_level=optimization_level, shots=shots)
            print(str(circ.name)+' circuit,Job ID', job.job_id())
            job_res = job.result()
            print(job_res)
            result_list.append(job_res)

            ## Create pickle dictionary
            pickle_file = pickle_file+"_"+str(job.job_id())
            pickle_data[str(job.job_id())] = job_res

            # Append to job list
            job_list[str(circ.name)].append(str(job.job_id()))

        # Dump on file
        with open(pickle_file,'wb+') as f:
            pickle.dump(pickle_data, f)

        counts_dict = {}
        ordered_bitstrings = dict(sorted(result_list[0].get_counts().items()))
        # Loop over the results of the scaled circuits and collect the data in the correct form
        for key in ordered_bitstrings.keys():
            counts_list = []
            for result in result_list:
                counts_list.append(result.get_counts()[key])
            # Here we extrapolate the counts to zero noise and round to the closest integer 
            zne_counts_value = int(zne.PolyFactory.extrapolate(scale_factors, counts_list, order=zne_order)) 
            if zne_counts_value < 0:
                zne_counts_value = 0
            counts_dict[key] = zne_counts_value
        zne_result_list.append(counts_dict)
        
    # To work with the StateTomographyFitter we need to put the result into a Qiskit Result() object
    name_list = [circ.name for circ in tomo_circs]
    results_tmp = [[ExperimentResult(shots=shots, success=True, data=ExperimentResultData(counts=result_i), header=QobjExperimentHeader(name=name_i))] for (name_i, result_i) in zip(name_list, zne_result_list)]
    results = [Result(backend_name="zne", backend_version="zne", qobj_id='0', job_id='0', success=True, results=result_i) for result_i in results_tmp]

    return results 


# Create the tomography circuits
st_qcs = state_tomography_circuits(fqc.decompose(), [fqr[1], fqr[3], fqr[5]])

# Repeat fidelity measurement
reps = 1 # Needs to be 8 in the final execution
fids = []
job_list = {}
for count in range(reps):
    print("\n\n\n\n REPETITION "+str(count+1)+"\n\n\n\n")
    
    zne_res = zne_results(st_qcs, backend=backend, optimization_level=0, zne_order=2, shots=shots,job_list=job_list)
    fids.append(state_tomo(zne_res, st_qcs))

## Print the final result
print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids))) 

print("\n\n\n JOB LIST:")
print(job_list)


















