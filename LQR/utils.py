"""
Utility functions.
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

	def __init__(self, A, B, Q, R, Wc, Wd):

		self.nx = A.shape[0] # Number of states
		self.nu = B.shape[1] # Number of control inputs
		self.nyp = Wd.shape[0] # Number of dimensions of "distractor" observations

		# Define linear system
		self.A = A 
		self.B = B

		self.Wc = Wc # Matrix that outputs (transformed) true state
		self.Wd = Wd # Matrix that outputs distractors
		self.W = torch.cat((self.Wc, self.Wd), 0) # matrix that outputs observations

		# Define LQR costs
		self.Q = Q
		self.R = R

		# Define initial state distribution
		self.mu0 = torch.zeros(self.nx)
		self.Sigma0 = torch.eye(self.nx)

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

	def kronecker(self, A, B):
		"""
		Kronecker product: see https://discuss.pytorch.org/t/kronecker-product/3919/5
		"""
		return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

	# def lin_solve_manual(self, LHS, RHS):
	# 	matTmat = torch.matmul(LHS.t(), LHS)
	# 	res_manual = torch.matmul(torch.matmul(torch.inverse(matTmat), LHS.t()), RHS)
	# 	return res_manual

	def closed_form_P_K(self, K):
		"""
		Compute P_K (the matrix that defines the quad. prod. for cost)
		See https://arxiv.org/abs/1801.05039 for original derivation.
		Also, see Appendix A.3.1 of https://arxiv.org/pdf/1912.02975.pdf
		"""

		# K_W = K * W
		K_W = K@self.W

		A_minus_BK = self.A - torch.matmul(self.B, K_W)
		A_minus_BK_t = A_minus_BK.t()
		kron_A_minus_BK_t = self.kronecker(A_minus_BK_t, A_minus_BK_t)
		nx_squared = self.nx**2

		# Flatten: (I_flatten - quad) * P_K = Q + K^T * R * K
		
		mat = torch.eye(nx_squared) - kron_A_minus_BK_t
		rhs = torch.reshape(self.Q + torch.matmul(K_W.t(),torch.matmul(self.R, K_W)), [-1,1])

		# P_K = self.lin_solve_manual(mat, rhs)
		P_K, _ = torch.solve(rhs, mat)

		P_K = torch.reshape(P_K, [self.nx, self.nx])

		return P_K

	def closed_form_costs(self, K):
		"""
		Calculate the expected (cumulative) cost: E[x0^T P_K x0]
		"""
	
		P_K = self.closed_form_P_K(K)
		exp_cost = torch.dot(self.mu0, torch.matmul(P_K, self.mu0)) + torch.trace(torch.matmul(P_K, self.Sigma0))

		return exp_cost

	def cost_to_go_estimate(self, K, x0):
		"""
		Estimate cost to go based on K.
		"""	
		xt = x0
		cost_est = 0
		for t in range(0, 1000):
			obst = self.W@xt
			ut = -K@obst
			cost_est += xt.t()@self.Q@xt + ut.t()@self.R@ut
			xt = self.A@xt + self.B@ut

		return cost_est	



##############################################################		

def sample_ortho_A(dim, scale = 1.0):
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

def sample_ortho_W(new_dim, dim, scale = 1.0):
    """
	Sample semi-orthogonal matrix W. Code from Song et al. 
    """

    random_matrix = np.random.rand(new_dim, new_dim)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    q = np.multiply(q, ph, q)
    actual_W = q[:, 0:dim]
    out = actual_W*scale
    return out