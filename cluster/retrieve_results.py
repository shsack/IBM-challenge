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
#jakarta = provider.get_backend('ibmq_jakarta')
# Simulated backend based on ibmq_jakarta's device noise profile
sim_noisy_jakarta = QasmSimulator.from_backend(provider.get_backend('ibmq_jakarta'))

shots = 8192
backend = sim_noisy_jakarta

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

# Function to retrieve results
def retrieve_zne_results(tomo_circs, zne_order, shots,job_list):

    # This function runs the tomography circuits and unrolls the gates to increase the noise level
    # The counts that are obtained for the differnt noise levels are then extrapolated to the zero-noise level

    zne_result_list = []
    scale_factors = [1.0, 2.0, 3.0]
    # Loop over the tomography circuits
    for circ in tomo_circs:

        ## identify the file from which retrieve the data

        job_key = job_list[str(circ.name)]
        result_list = []

        pickle_file = "./data/"+str(circ.name)

        for job_id in job_key:
        	pickle_file = pickle_file+"_"+job_id

        
        pickle_data = {}
        result_file = pickle.load(open(pickle_file,'rb'))
        print("\nLOADED FILE: "+pickle_file)

        for job_id in job_key:
        	result_list.append(result_file[job_id])


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

fids = []

#### Here we put the job list
job_list = {"('X', 'X', 'X')": ['038505f4-6643-497f-b16d-04b129d073e7', 'ff50ce2a-15ec-431d-bd2a-30f7ec1beddd', '2c8d45dd-637c-4ca8-8829-b34315105ec8'], "('X', 'X', 'Y')": ['f8f9d7ff-2f9f-497f-824e-973267d67763', '68009a81-f72e-4a36-af6c-1f178d3df4c1', '11674c75-211a-4367-b332-c682ef9a7aae'], "('X', 'X', 'Z')": ['60f0907b-8a5d-47a3-855c-4d4af8016b05', '37573946-f944-4b17-9b70-900d1ff05529', '3ffcd054-c8fd-4509-913b-bb583be05d70'], "('X', 'Y', 'X')": ['763d0c76-60fc-4b79-bff9-4c385f032f80', '3c1c7d26-e304-4b7a-aaae-66202cbbdbbe', '31302236-60ba-4699-b6c9-76b58e49b3ed'], "('X', 'Y', 'Y')": ['83101cbf-d1b5-4596-a63b-7ae06e77e8ce', '38d1d711-ccb3-4fd9-b669-f7df033c2f40', '117c8681-ad11-4ba8-8019-7d124cf2c520'], "('X', 'Y', 'Z')": ['3cc05487-ca1d-4872-af1b-150c5a352bca', 'f86cdab0-61f9-497c-a94f-e4736e8e6307', '609e511e-b48c-4ff6-8130-d69d54be2541'], "('X', 'Z', 'X')": ['be42eec1-4833-4bd9-a525-c1e7cef3266f', '05be7b89-c60e-4c2b-a42f-62aa1227b82f', '59919853-e54d-4714-83f1-d77947890045'], "('X', 'Z', 'Y')": ['bc969e5d-1ef3-4b3e-ae4b-a23d432a423b', 'f7c9b5f1-642a-4f41-b112-a9159e228689', '334ad5c0-6c92-42db-9029-b3f955ed1bb9'], "('X', 'Z', 'Z')": ['775a49c7-4471-49f6-b82b-d979097426bf', '754863ef-01c5-4bd4-9a82-a264fcabc1d4', '6031f890-d975-44d9-9201-2673047604f4'], "('Y', 'X', 'X')": ['c3574629-1591-4522-b37f-7c56e94e4d02', 'e20d130b-fb2e-4e7a-8bfb-03a3367145aa', '74b8c459-6eda-42f9-87e2-cea7a96196a9'], "('Y', 'X', 'Y')": ['01e1fd3d-7793-4d66-a299-306077e468c5', '87424cf8-a8cf-4feb-9e10-f97d244dc042', '993aacc4-cca1-49e7-b71e-345aeb238415'], "('Y', 'X', 'Z')": ['bfa04918-4444-4703-bdb0-c19b00e1ea84', 'fdd66e23-030d-4f2c-990a-f5c6ee79deb6', '15177b27-852c-4d77-85f8-45bdfd894435'], "('Y', 'Y', 'X')": ['6c69da4e-01a9-4b05-9b87-cb2bd37714ba', '990948f5-3436-41e9-a26f-31ab7569d66b', 'c21e8438-8554-4303-88dc-797f7da83b0e'], "('Y', 'Y', 'Y')": ['6f68849f-d303-4249-a42a-f6cfd2a598dd', '0f48aeae-fa53-4f3a-9e9c-80f38613356c', '2b53c612-dfc5-49d4-bfbd-dc789ece8f87'], "('Y', 'Y', 'Z')": ['e98d22b2-c639-4b9f-a1ad-82ffe6cf6670', '65c39fc8-4728-40f4-8e4e-edc377a0709c', 'd89d7a2d-6712-46a0-990b-5f0556c182de'], "('Y', 'Z', 'X')": ['17bb66fd-6144-4e0b-80e5-24419505c739', '2cf6d543-e768-4cfe-b6c6-c6d765d705c8', '958bc33b-6af2-4d24-a859-4490c9305eeb'], "('Y', 'Z', 'Y')": ['3dcbbf6a-038d-428b-9a13-89188b228762', '89eee4bc-27c4-4756-a3a7-b512803612d5', 'a0e30f7e-4a75-4a74-a163-fd83ddc99129'], "('Y', 'Z', 'Z')": ['69f21db3-8344-43f3-87e6-f37dac3f4f2b', '087ca4f9-5130-4a8c-b68e-83d1d9ebe213', '18569695-9715-43a3-9f37-823bd357c94f'], "('Z', 'X', 'X')": ['7a432c78-2b6b-478e-baf6-a6916b7e1e04', '157b5547-118f-4c93-aeeb-0dfd4b0a704e', '1733768f-b6ae-4923-ac03-19ecdcfbdefd'], "('Z', 'X', 'Y')": ['8383e189-5ada-456d-811a-b028c1233676', '7bb8cdcb-2a4d-4156-b99e-c8b3de8165b3', '5057dc42-707a-412b-bbe2-50da456b70fb'], "('Z', 'X', 'Z')": ['2f677af2-2d6f-4696-b306-903c8e1e8711', 'f9a341e0-519c-4ed6-a874-0d37f9556143', '9b5bfe17-3bf6-49fc-826c-3e2efc960983'], "('Z', 'Y', 'X')": ['f19ca594-37d0-47cc-874c-a516ea5cba5e', '3a6f3d66-7f6f-4107-bddc-37bad43a43d5', '076adb7b-0981-494b-acee-71d38e4c49c4'], "('Z', 'Y', 'Y')": ['6d4b4461-13a9-43f8-8793-38b67e8f3ef9', '0402cbe7-e708-4bfb-9af4-7a424d26f7bf', '54ed6f47-52ab-42f0-be5b-02859af93982'], "('Z', 'Y', 'Z')": ['7a9cfbca-5a36-4029-b640-9b7b452bc897', '9e0d0e88-8ef3-4396-ac17-b778edb839fc', '3a7e60ed-7039-4d48-b121-f899943ef10b'], "('Z', 'Z', 'X')": ['bd07e072-a4cc-44e0-8186-2491dca8cc1b', 'b24cf655-a29f-4c81-8c49-a1ac43a5f0a0', '8fcb0e65-9e0d-47f8-961f-936320873633'], "('Z', 'Z', 'Y')": ['67deeac5-00e0-436d-b4e7-358a87626e95', '727e555a-3902-458d-ace7-2afd8cb1d4f0', '1e50aa4b-7c4b-410a-b043-0e7355be99d2'], "('Z', 'Z', 'Z')": ['0801b5c9-05ac-49fa-b902-2db40e39cb13', '85f061e6-77a9-4092-ac20-8867631078f4', '12154764-0c1b-43b0-b5c8-9cfca361082c']}

    
zne_res = retrieve_zne_results(st_qcs, zne_order=2, shots=shots,job_list=job_list)
fids.append(state_tomo(zne_res, st_qcs))

## Print the final result
print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids))) 





