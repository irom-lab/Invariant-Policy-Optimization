"""
Self-contained code for finding LQR gains using gradient descent. 
This is primarily for debugging/double-checking things.  This does NOT train
using multiple domains. 
"""


import torch
from  torch.autograd  import  grad
import IPython as ipy
from scipy.linalg import solve_discrete_are
import numpy as np


class LQR():
	"""
	Class for Linear Quadratic Regulator problems.
	"""

	def __init__(self, nx, nu):

		self.nx = nx # Number of states
		self.nu = nu # Number of control inputs

		# Define linear system
		self.A = torch.tensor(self.sample_ortho_A(nx, 0.99)).float() 
		self.B = torch.randn(self.nx, self.nu)
		self.Wc = torch.eye(self.nx)

		# Define LQR costs
		Q = torch.randn(self.nx, self.nx)
		self.Q = torch.matmul(Q, Q.t()) # make it pos. def.
		R = torch.randn(self.nu, self.nu)
		self.R = torch.matmul(R, R.t()) # make it pos. def.

		# Define initial state distribution
		self.mu0 = torch.zeros(nx)
		self.Sigma0 = torch.eye(nx)

		# K matrix
		self.K = torch.nn.Parameter(torch.zeros(self.nu, self.nx))


		# Optimizer
		self.optimizer = torch.optim.Adam([self.K], lr=0.001)

	def S_riccatti(self):
		"""
		Solve for cost-to-go matrix using (discrete-time) Riccatti equation.
		"""
		A = self.A
		B = self.B
		Q = self.Q
		R = self.R

		S = solve_discrete_are(A, B, Q, R)
		S = torch.tensor(S).float()

		return S

	def K_riccatti(self):
		"""
		Solve for LQR gain matrix using (discrete-time) Riccatti equation.
		"""
		A = self.A
		B = self.B
		Q = self.Q
		R = self.R
		B_t = B.t()

		S = self.S_riccatti()
		K = torch.inverse((R+B_t@S@B)) @ (B_t@S@A)

		return K

	def sample_ortho_A(self, dim, scale = 1.0):
	    """
	    Samples orthogonal matrix A. Code from Song et al. 
	    """
	    random_matrix = np.random.rand(dim, dim)
	    q, r = np.linalg.qr(random_matrix)
	    d = np.diagonal(r)
	    ph = d/np.absolute(d)
	    q = np.multiply(q, ph, q)
	    out = q*scale
	    return out

	def kronecker(self, A, B):
		"""
		Kronecker product: see https://discuss.pytorch.org/t/kronecker-product/3919/5
		"""
		return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

	def lin_solve_manual(self, LHS, RHS):
		matTmat = torch.matmul(LHS.t(), LHS)
		res_manual = torch.matmul(torch.matmul(torch.inverse(matTmat), LHS.t()), RHS)
		return res_manual

	def closed_form_P_K(self):
		"""
		Compute P_K (the matrix that defines the quad. prod. for cost)
		See https://arxiv.org/abs/1801.05039 for original derivation.
		Also, see Appendix A.3.1 of https://arxiv.org/pdf/1912.02975.pdf
		"""

		A_minus_BK = self.A - torch.matmul(self.B, self.K)
		A_minus_BK_t = A_minus_BK.t()
		kron_A_minus_BK_t = self.kronecker(A_minus_BK_t, A_minus_BK_t)
		nx_squared = self.nx**2

		# Flatten: (I_flatten - quad) * P_K = Q + K^T * R * K
		
		mat = torch.eye(nx_squared) - kron_A_minus_BK_t
		rhs = torch.reshape(self.Q + torch.matmul(self.K.t(),torch.matmul(self.R, self.K)), [-1,1])

		# P_K = self.lin_solve_manual(mat, rhs)
		P_K, _ = torch.solve(rhs, mat)

		P_K = torch.reshape(P_K, [nx, nx])

		return P_K

	def closed_form_costs(self):
		"""
		Calculate the expected (cumulative) cost: E[x0^T P_K x0]
		"""
	
		P_K = self.closed_form_P_K()
		exp_cost = torch.dot(self.mu0, torch.matmul(P_K, self.mu0)) + torch.trace(torch.matmul(P_K, self.Sigma0))

		return exp_cost

	def cost_to_go_estimate(self, K, x0):
		"""
		Estimate cost to go based on K.
		"""	
		xt = x0
		cost_est = 0
		for t in range(0, 1000):
			ut = -K@xt
			cost_est += xt.t()@self.Q@xt + ut.t()@self.R@ut
			xt = self.A@xt + self.B@ut

		return cost_est	



# Define dimensions
nx = 10 # Number of states
nu = 5 # Number of control inputs

# Define LQR system
lqr = LQR(nx, nu)

# Gradient descent on (cumulative) reward
num_iters = 10000
for iter in range(0, num_iters):
	# Calculate loss
	cost = lqr.closed_form_costs()

	# Optimize K
	lqr.optimizer.zero_grad()
	cost.backward()
	lqr.optimizer.step()

	if iter % 100 == 0:
		# print("K: ", lqr.K)
		print("iter: ", iter)
		print("cost: ", cost.item())

# Print optimized K matrix
print(" ")
print("Optimized K: ", lqr.K)

# Compute actual LQR solution
K_analytic = lqr.K_riccatti()
print("LQR K: ", K_analytic)

# Compute expected cost for our K and LQR
exp_cost_K = lqr.closed_form_costs()
P_lqr = lqr.S_riccatti()
exp_cost_lqr = torch.dot(lqr.mu0, torch.matmul(P_lqr, lqr.mu0)) + torch.trace(torch.matmul(P_lqr, lqr.Sigma0))

print(" ")
print("K exp. cost: ", exp_cost_K.item())
print("LQR oracle exp. cost: ", exp_cost_lqr.item())





