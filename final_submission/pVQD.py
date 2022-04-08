import numpy as np 
from qiskit.opflow                import Z, I


# Useful functions

def projector_zero(n_qubits):

    '''
    This function create the global projector |00...0><00...0|
    Args:
        - n_qubits: the number of qubits of the system
    Returns:
        The projector |00...0><00...0| as a Qiskit Opflow operator

    '''
    from qiskit.opflow            import Z, I

    prj_list = [0.5*(I+Z) for i in range(n_qubits)]
    prj = prj_list[0]

    for a in range(1,len(prj_list)):
        prj = prj^prj_list[a]

    return prj


def ei(i,n):

    '''
    This function returns the i-th basis vector on a vector space of dimension n.
    This is useful to implement the parameter-shift rule.

    Args:
        - i: the 
        - n: the dimension of the vector

    Returns:
        An array of dimension n with all 0 except for the i-th component, set to 1

    '''
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]


def adam_gradient(params,count,m,v,g):

    '''
    This function implements ADAM optimizer from scratch.
    Given an array of parameters and the gradient it returns a new array of parameters,
    following arXiv: 1412.6980 .
    Args:
        - params: array of parameters to optimize
        - count : the optimization step, counting from 1
        - m     : the moving averages of the gradient, updated after every step
        - v     : the moving averages of the squared gradient, updated after each step
        - g     : the gradient 
        
    Returns:
        The updated array of parameter 

    '''

    # Set the ADAM hyperparameters 
    # β1,β2 control he exponential decay rates of the moving averages
    β1 = 0.9
    β2 = 0.999
    # Regularization constant
    ε  = 1e-8
    # Parameter wise adaptive learning rate
    α  = [0.001 for i in range(len(params))]
    if count == 0:
        count = 1

    new_params = [0 for i in range(len(params))]

    for i in range(len(params)):
        m[i] = β1 * m[i] + (1 - β1) * g[i]
        v[i] = β2 * v[i] + (1 - β2) * np.power(g[i],2)

        α[i] = α[i] * np.sqrt(1 - np.power(β2,count)) / (1 - np.power(β1,count))

        new_params[i] = params[i] + α[i]*(m[i]/(np.sqrt(v[i])+ε))

    return new_params


