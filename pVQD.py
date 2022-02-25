import numpy as np


# 
#  TODO: this should contain the stripped down version of the functions that
# are needed for the pVQD. Stripped down means no class, no checks, etc. Such 
# that it is easy to understand and clean.



def projector_zero_local(n_qubits):
	# This function creates the local version of the cost function 
	# proposed by Cerezo et al: https://www.nature.com/articles/s41467-021-21728-w
	from qiskit.opflow          import Z, I
	
	tot_prj = 0
	
	for k in range(n_qubits):
		prj_list = [I for i in range(n_qubits)]
		prj_list[k] = 0.5*(I+Z)
		prj = prj_list[0]

		for a in range(1,len(prj_list)):
			prj = prj^prj_list[a]

		#print(prj)

		tot_prj += prj

	tot_prj = (1/n_qubits)*tot_prj
	
	return tot_prj


def ei(i,n):
	vi = np.zeros(n)
	vi[i] = 1.0
	return vi[:]




def adam_gradient(self,count,m,v,g):
    ## This function implements adam optimizer
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8
    alpha = [0.001 for i in range(len(self.parameters))]
    if count == 0:
        count = 1

    new_shift = [0 for i in range(len(self.parameters))]

    for i in range(len(self.parameters)):
        m[i] = beta1 * m[i] + (1 - beta1) * g[i]
        v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i],2)

        alpha[i] = alpha[i] * np.sqrt(1 - np.power(beta2,count)) / (1 - np.power(beta1,count))

        new_shift[i] = self.shift[i] + alpha[i]*(m[i]/(np.sqrt(v[i])+eps))

    return new_shift


## Probably use SPSA to have fewer circuit evaluations and the whole thing is a little faster

def compute_overlap_and_gradient_spsa(self,state_wfn,parameters,shift,expectator,sampler,count):

    nparameters = len(parameters)
    # build dictionary of parameters to values
    # {left[0]: parameters[0], .. ., right[0]: parameters[0] + shift[0], ...}

    # Define hyperparameters
    c  = 0.1
    a  = 0.16
    A  = 1
    alpha  = 0.602
    gamma  = 0.101

    a_k = a/np.power(A+count,alpha)
    c_k = c/np.power(count,gamma)

    # Determine the random shift

    delta = np.random.binomial(1,0.5,size=nparameters)
    delta = np.where(delta==0, -1, delta) 
    delta = c_k*delta

    # First create the dictionary for overlap
    values_dict = [dict(zip(self.right[:] + self.left[:], parameters.tolist() + (parameters + shift).tolist()))]
    

    # Then the values for the gradient
    
    values_dict.append(dict(zip(self.right[:] + self.left[:], parameters.tolist() + (parameters + shift + delta).tolist())))
    values_dict.append(dict(zip(self.right[:] + self.left[:], parameters.tolist() + (parameters + shift - delta).tolist())))

    # Now evaluate the circuits with the parameters assigned

    results = []

    for values in values_dict:
        sampled_op = sampler.convert(state_wfn,params=values)

        mean  = sampled_op.eval()[0]
        mean  = np.power(np.absolute(mean),2)
        est_err = 0


        if (not self.instance.is_statevector):
            variance = expectator.compute_variance(sampled_op)[0].real
            est_err  = np.sqrt(variance/self.shots)

        results.append([mean,est_err])

    E = np.zeros(2)
    g = np.zeros((nparameters,2))

    E[0],E[1] = results[0]

    # and the gradient

    rplus  = results[1]
    rminus = results[2]

    for i in range(nparameters):
        # G      = (Ep - Em)/2Δ_i
        # var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
        g[i,:] = a_k*(rplus[0]-rminus[0])/(2.0*delta[i]),np.sqrt(rplus[1]**2+rminus[1]**2)/(2.0*delta[i])

    self.overlap  = E
    self.gradient = g

    return E,g 