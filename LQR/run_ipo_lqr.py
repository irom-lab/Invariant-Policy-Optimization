"""
LQR with distractor observations: IPO.
"""

import argparse
from utils import *

def main(raw_args=None):

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
	parser.add_argument("--num_domains", type=int, default=2, help="Number of training domains (default: 2)")
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")
	parser.add_argument("--num_distractors", type=int, default=1000, help="dimension of distractors (default: 1000")


	args = parser.parse_args(raw_args)
	seed = args.seed

	# Set seed
	np.random.seed(seed)
	torch.manual_seed(seed)

	##############################################################
	# Define LQR system parameters that are fixed across domains
	##############################################################

	# Define dimensions
	nx = 20 # Number of states
	nu = 20 # Number of control inputs
	nyp = args.num_distractors # Number of dimensions of "distractor" observations

	# Dynamics
	A = torch.tensor(sample_ortho_A(nx, 0.99)).float() 
	B = torch.eye(nu) # torch.randn(nx, nu)

	# Fixed portion of observation matrix
	Wc = torch.tensor(sample_ortho_W(nx, nx)).float() # Matrix that outputs (transformed) true state

	# Costs
	Q = torch.eye(nx)
	R = torch.eye(nu)
	# Q = torch.randn(nx, nx)
	# Q = torch.matmul(Q, Q.t()) # make it pos. def.
	# R = torch.randn(nu, nu)
	# R = torch.matmul(R, R.t()) # make it pos. def.

	##############################################################
	# Define LQR systems corresponding to different domains
	##############################################################

	num_domains = args.num_domains
	lqr_all = []
	for domain in range(num_domains):
		Wd_domain = torch.tensor(sample_ortho_W(nyp, nx)).float() # Matrix that outputs distractors
		lqr_domain = LQR(A, B, Q, R, Wc, Wd_domain)
		lqr_all.append(lqr_domain)


	##############################################################
	# Train using IPO (best-response training)
	##############################################################

	# K matrix and optimizer for different domains 
	K_all = []
	optimizers_all = []
	for domain in range(num_domains):
		K_domain = torch.nn.Parameter(torch.zeros(nu, nx+nyp))
		optimizer_domain = torch.optim.Adam([K_domain], lr=0.0005)
		K_all.append(K_domain)
		optimizers_all.append(optimizer_domain)

	num_iters = 5000
	for iter in range(0, num_iters):
		
		cost_av = 0

		for domain in range(num_domains):

			#################################
			# Optimize K for this domain
			#################################

			# Calculate loss for this domain
			K = sum(K_all)/num_domains
			cost = lqr_all[domain].closed_form_costs(K)

			cost_av += cost/num_domains

			# Optimize
			optimizers_all[domain].zero_grad()
			cost.backward()
			optimizers_all[domain].step()


		if (iter % 100 == 0) and args.verbose:
			print("iter: ", iter)
			print("cost: ", cost_av.item())


	##############################################################
	# Define "test" LQR system (new domain) and run K on this
	##############################################################

	# Define test LQR system
	Wd_test = torch.tensor(sample_ortho_W(nyp, nx)).float() # Matrix that outputs distractors
	lqr_test = LQR(A, B, Q, R, Wc, Wd_test)

	# Compute actual LQR solution for this LQR system
	K_analytic = lqr_test.K_riccatti()
	# print("LQR K: ", K_analytic)

	# Compute expected cost for our K and LQR on test system
	exp_cost_K = lqr_test.closed_form_costs(K)
	exp_cost_K = exp_cost_K.item()
	P_lqr = lqr_test.S_riccatti()
	exp_cost_lqr = torch.dot(lqr_test.mu0, torch.matmul(P_lqr, lqr_test.mu0)) + torch.trace(torch.matmul(P_lqr, lqr_test.Sigma0))
	exp_cost_lqr = exp_cost_lqr.item()

	print(" ")
	print("Costs on test LQR system (IRM)")
	print("K exp. cost: ", exp_cost_K)
	print("LQR oracle exp. cost: ", exp_cost_lqr)

	# # Calculate cost-to-go from some random x0 (via rollout)
	# x0 = torch.randn(nx, 1)
	# K_analytic_obs = torch.cat((K_analytic@Wc.t(), torch.zeros(nu, nyp)), 1)
	# cost1_x0 = lqr_test.cost_to_go_estimate(K, x0)
	# cost2_x0 = lqr_test.cost_to_go_estimate(K_analytic_obs, x0)

	# print(" ")
	# print("Cost to go from x0 (K): ", cost1_x0.item())
	# print("Cost to go from x0 (LQR): ", cost2_x0.item())

	return exp_cost_K, exp_cost_lqr


# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()  




