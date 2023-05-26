import Exam as ex
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import optimize
import time

from types import SimpleNamespace

# Creating definitions for Problem 1 


# Question 1-4

# 1. Defining the optimal labor function
def optimal_labor_supply(w, alpha, kappa, nu, tau):
    """
    Define the optimal labor supply function

    Args: 
        w(float): the real wage 
        alpha(float): parameter 
        kappa(float): free privat consumption component 
        nu(float): disutility of labor scaling factor 
        tau(float): labor income tax-rate
    
    Returns: 
        the optimal labor supply function 
    """
    
    #a. 
    w_tilde = (1 - tau) * w
    
    #b. 
    optimal_l = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * w_tilde**2)) / (2 * w_tilde)
    return optimal_l


# 2. Defining the function for optimal labor supply
def optimal_labor_supply_w_tilde(w_tilde, alpha, kappa, nu):
    """ 
    Defining the optimal labor supply function with w_tilde
    
    Args: 
        w_tilde:  w_tilde = (1 - tau) * w
        alpha(float): parameter 
        kappa(float): free privat consumption component 
        nu(float): disutility of labor scaling factor

    Returns: 
        expression(float): The optimal labor supply function with w_tilde  
    """

    # a. Optimal labor supply function 
    optimal_l = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * w_tilde**2)) / (2 * w_tilde)
    return optimal_l


# 3. Defining the function for government consumption
def government_consumption(tau, w, alpha, kappa, nu):
    """"Defining the goverment consumption function 

    Args: 

        tau(float): labor income tax-rate
        w(float): the real wage 
        alpha(float): parameter 
        kappa(float): free privat consumption component 
        nu(float): disutility of labor scaling factor 
    
    Returns: 

        G(float) : the goverment consumption function 
    """
    # a. Generating w_tilde
    w_tilde = (1 - tau) * w
    
    # b. Generating L_opt
    L_opt = optimal_labor_supply_w_tilde(w_tilde, alpha, kappa, nu)
    
    # c. Generating G
    G = tau * w * L_opt
    return G


# 4. Defining the function for worker utility
def worker_utility(tau, w, alpha, kappa, G, nu):
    """
    Calculate the utility of a worker.

    Parameters:
        tau (float): Tax rate.
        w (float): Wage rate.
        alpha (float): Weight on consumption in utility.
        kappa (float): Fixed cost of working.
        G (float): Public good.
        nu (float): Coefficient of relative risk aversion.

    Returns:
        utility (float): Utility of the worker.
    """

    # a. Gemerate w_tilde
    w_tilde = (1 - tau) * w
    
    # b. Generate optimal labour
    L_opt = optimal_labor_supply_w_tilde(w_tilde, alpha, kappa, nu)
    
    # c. Generate consumption
    C = kappa + (1 - tau) * w * L_opt
    
    # d. Generate utility
    utility = np.log(C**alpha * G**(1 - alpha)) - nu * L_opt**2 / 2
    return utility








# Question 5

# 1 Defining the utility function  
def objective_function_f(w, G, tau, alpha, kappa, sigma, rho, epsilon, nu):
    """Calculating the utility function value.

    Args:
        w (float): The real wage rate.
        G (float): Goverment consumption 
        tau (float): Tax rate.
        alpha (float): Weight on consumption in the objective function.
        kappa (float): The free private consumption component.
        sigma (float): Parameter for the utility function.
        rho (float): Parameter for the utility function.
        epsilon (float): Parameter for the utility function.
        nu (float): The disutility of labor scaling factor

    Returns:
        value (float): Value of the objective function.
    Raises:
        ValueError: If optimization fails.
    """
    # a  calculating the value of the inner objective function for a given labor supply
    def inner_objective(L):
        # i defining the consumption function 
        C = kappa + (1 - tau) * w * L
        # ii returning the utility function 
        return (((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**(sigma / (sigma - 1)))**(1-rho) - 1 ) / (1 - rho) - nu * (L**(1+epsilon)) / (1+epsilon)
    
    # b Using a numerical solver to find the maximum of the inner objective function
    result = optimize.minimize_scalar(lambda L: -inner_objective(L), bounds=(0, 24), method='bounded')
    
    # c if-else statement for debugging
    if result.success:
        return -result.fun  # Return the negativevalue for maximization
    else:
        raise ValueError("Optimization failed.")

# 2 Solving for optimal labor supply L* for a given set of parameters
def find_optimal_labor_supply(w, G, tau, alpha, kappa, sigma, rho, epsilon, nu):
    """Find the optimal labor supply.

    Args:
        w (float): The real wage rate.
        G (float): Goverment consumption 
        tau (float): Tax rate.
        alpha (float): Weight on consumption in the objective function.
        kappa (float): The free private consumption component.
        sigma (float): Parameter for the utility function.
        rho (float): Parameter for the utility function.
        epsilon (float): Parameter for the utility function.
        nu (float): The disutility of labor scaling factor

    Returns:
        L_star (float): Optimal labor supply.
    Raises:
        ValueError: If optimization fails.
    """
    # a Optimizing the labor supply 
    result = optimize.minimize_scalar(lambda L: -objective_function_f(w, G, tau, alpha, kappa, sigma, rho, epsilon, nu), bounds=(0, 24), method='bounded')
    
    # b if-else statement for debugging 
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed.")

# 3 Solving for G that satisfies G = tau * w * L*
def find_G(w, tau, alpha, kappa, sigma, rho, epsilon, nu):
    """Find the value of G that satisfies G = tau * w * L*.

    Args:
        w (float): The real wage rate. 
        tau (float): Tax rate.
        alpha (float): Weight on consumption in the objective function.
        kappa (float): The free private consumption component.
        sigma (float): Parameter for the utility function.
        rho (float): Parameter for the utility function.
        epsilon (float): Parameter for the utility function.
        nu (float): The disutility of labor scaling factor

    Returns:
        G (float): Value of G.
    Raises:
        ValueError: If optimization or root finding fails.
    """

    # a. Initial guess for G
    G = 10  

    # b. defining the goverment consumption function
    def equation(G, L):
        return G - tau * w * L
    
    # c. Defining the utility function for a given labor supply
    def objective_function_l(L, G):
        C = kappa + (1 - tau) * w * L
        return (((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**(sigma / (sigma - 1)))**(1-rho) - 1 ) / (1 - rho) - nu * (L**(1+epsilon)) / (1+epsilon)
        
    # d. Using a numerical solver to find the maximum of the inner objective function
    result = optimize.minimize_scalar(lambda L: -objective_function_l(L, G), bounds=(0, 24), method='bounded')
    
    # e. if-else statement for debugging and returning the results
    if result.success:
        L_star = result.x
        result = optimize.root_scalar(lambda G: equation(G, L_star), method='brentq', bracket=[0, 1000])
        if result.converged:
            return result.root
        else:
            raise ValueError("Root finding failed.")
    else:
        raise ValueError("Optimization failed.")









# Question 6

# 1. Define the objective function
def objective_function(L, w, tau, G, alpha, kappa, sigma, rho, epsilon, nu):
    """Calculate the utility for a given labor supply.

    Args:
        L(float): labor supply
        w (float): The real wage rate. 
        tau (float): Tax rate.
        G (float): Goverment consumption
        alpha (float): Weight on consumption in the objective function.
        kappa (float): The free private consumption component.
        sigma (float): Parameter for the utility function.
        rho (float): Parameter for the utility function.
        epsilon (float): Parameter for the utility function.
        nu (float): The disutility of labor scaling factor

    Returns:
        utility
    """
    
    C = kappa + (1 - tau) * w * L
    return (((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**(sigma / (sigma - 1)))**(1 - rho) - 1) / (1 - rho) - nu * (L**(1 + epsilon)) / (1 + epsilon)


# 2. Define the constraint function
def constraint_function(tau, w, L_opt, G):
    """Calculate the constraint value.

    Args: 
        tau (float): Tax rate.
        w (float): The real wage rate.
        L_opt(float): Optimal labor supply 
        G(float): Goverment consumption

    Returns:
    Constraint value
    """ 
     
    return G - tau * w * L_opt * ((1 - tau) * w)


# 3. Finding the socially optimal tax rate for Set_1
def objective_set1(tau):
    """Calculating the utility for a given tax rate in Set 1.

    Args:
        tau (float): Tax rate.

    Returns:
        utility
    """
    w = 1.0
    rho1 = 1.001
    alpha = 0.5
    kappa = 1.0
    nu = 1 / (2 * 16**2)
    epsilon = 1.0
    sigma1 = 1.001

    # a Find the optimal labor supply for the given tax rate
    L_opt = optimize.minimize_scalar(lambda L: -objective_function(L, w, tau, tau * w * L * ((1 - tau) * w), alpha, kappa, sigma1, rho1, epsilon, nu), bounds=(0, 24), method='bounded').x
    
    # b Calculating the constraint value
    constraint_val = constraint_function(tau, w, L_opt, tau * w * L_opt * ((1 - tau) * w))
    
    # c Calculating the worker utility
    utility = objective_function(L_opt, w, tau, tau * w * L_opt * ((1 - tau) * w), alpha, kappa, sigma1, rho1, epsilon, nu)
    
    # d Penalty term for constraint violation - explanation futher down  
    penalty = 1e8 * np.maximum(0, -constraint_val)
    
    # e Return the negative utility (since we want to maximize)
    return -utility + penalty


# 4. Finding the socially optimal tax rate for Set_2
def objective_set2(tau):
    """Calculating the utility for a given tax rate in Set 2.

    Args:
        tau (float): Tax rate.

    Returns:
        utility
    """
    w = 1.0
    rho2 = 1.5
    alpha = 0.5
    kappa = 1.0
    nu = 1 / (2 * 16**2)
    epsilon = 1.0
    sigma2 = 1.5

    # a. Finding the optimal labor supply for the given tax rate
    L_opt = optimize.minimize_scalar(lambda L: -objective_function(L, w, tau, tau * w * L * ((1 - tau) * w), alpha, kappa, sigma2, rho2, epsilon, nu), bounds=(0, 24), method='bounded').x
    
    # b. Calculate the constraint value
    constraint_val = constraint_function(tau, w, L_opt, tau * w * L_opt * ((1 - tau) * w))
    
    # c. Calculate the worker utility
    utility = objective_function(L_opt, w, tau, tau * w * L_opt * ((1 - tau) * w), alpha, kappa, sigma2, rho2, epsilon, nu)
    
    # d. Penalty term for constraint violation
    penalty = 1e8 * np.maximum(0, -constraint_val)
    
    # e. Return the negative utility (since we want to maximize)
    return -utility + penalty
















# Creating class for problem 2
class problem2:


    # 1. Initializing
    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. Setting parameters
        par.eta = 0.5
        par.w = 1.0
        par.R = (1+0.01)**(1/12) 
        par.jota = 0.01  
        par.rho = 0.9 
        par.t = 120
        par.K = 100000
        par.Delta = 0.05
        par.alpha = 0.5

        # b. Generating values for kappa and l
        par.kappa_values = np.linspace(1.0, 2.0, 100)  
        par.l_values = np.linspace(0.1, 2.0, 100)
        
        # c. Generating epsilon using a normal distribution
        par.std_dev = 0.10  
        par.mean = -0.5 * par.std_dev**2 
        par.epsilon_vals = np.random.normal(par.mean, par.std_dev, size=(par.K, par.t))

        # d. Creating kappa array like the epsilon array
        par.kappa_vals = np.empty_like(par.epsilon_vals)
        par.kappa_vals[:, 0] = 1 

        for t in range(1, par.t):  # start from 1 because kappa[0] is already initialized
            par.kappa_vals[:, t] = np.exp(par.epsilon_vals[:, t]) * (par.kappa_vals[:, t-1] ** par.rho)


        # e. Initializing arrays
        par.l_t1 = np.zeros(par.K)


        sol.profits = []


    # 2. Defining meximizing function for first question
    def profit_max1(self):
        """
        This function finds the optimal kappa and l values and the corresponding profit
        """

        par = self.par
        sol = self.sol


        # a. Creating a mesh grid of kappa and l values        
        sol.kappa_grid, sol.l_grid = np.meshgrid(par.kappa_values, par.l_values)

        # b. Profit values
        sol.profit_grid = sol.kappa_grid * sol.l_grid**(1 - par.eta) - par.w * sol.l_grid

        # c. np.argmax finds the maximum value in the array, np.unravel_index converts the index of the maximum value into a tuple
        max_index = np.unravel_index(np.argmax(sol.profit_grid, axis=None), sol.profit_grid.shape)

        # d. Find the optimal kappa, l, and max profit
        sol.opt_kappa = sol.kappa_grid[max_index]
        sol.opt_l = sol.l_grid[max_index]
        sol.max_profit = sol.profit_grid[max_index]


    # 3. Defining employee policy function
    def employee_func(self, kappa):
        """
        This function takes a value of kappa and returns the corresponding value of l
        
        Args:
            kappa (float): Value of kappa
            
        Returns:
            l (float): Value of l
        """

        par = self.par
        return ((1-par.eta)*kappa/par.w)**(1/par.eta)


    # 4. Defining ex post value function
    def ex_post_val(self, l_t, l_t1,kappa_t):
        """
        This function takes a value of l, l_t-1, and kappa_t and returns the corresponding ex post value of the salon

        Args:
            l_t (float): Value of l at time t
            l_t1 (float): Value of l at time t-1
            kappa_t (float): Value of kappa at time t

        Returns:
            ex_post_val (float): Ex post value of the salon    
        """

        par = self.par
        return par.R**(-par.t)*(kappa_t*l_t**(1-par.eta) - par.w*l_t - par.jota*(l_t != l_t1))  


    # 5. Defining meximizing function for second question    
    def profit_max2(self):
        par = self.par
        sol = self.sol

        par.kappa_vals[:, 0] = 1 


        for t in range(1, par.t):  # start from 1 because kappa[0] is already initialized
            par.kappa_vals[:, t] = np.exp(par.epsilon_vals[:, t]) * (par.kappa_vals[:, t-1] ** par.rho)

        for kappa_t in par.kappa_vals.T:
            l_t = self.employee_func(kappa_t)
            sol.profits.append(self.ex_post_val(l_t, par.l_t1, kappa_t))
            par.l_t1 = l_t
        
        H = np.mean(np.sum(sol.profits, axis=0))
        print(f'The maximum profit, H = {H:.4f}')
    

    # 6. Defining employee policy function with threshold
    def policy_func_delta(self, kappa_t, l_t1, Delta):
        """
        This function takes a value of kappa, previous l value, and a threshold Delta
        and returns the corresponding value of l according to the policy

        Args:
            kappa_t (float): Value of kappa at time t
            l_t1 (float): Value of l at time t-1
            Delta (float): Threshold value for the policy

        Returns:
            l_t (float): Value of l at time t
        """
        par = self.par
        l_star = ((1-par.eta)*kappa_t/par.w)**(1/par.eta)
        return np.where(np.abs(l_t1 - l_star) > Delta, l_star, l_t1)


    # 5. Modifying meximizing function for second question 
    def profit_max3(self):
        """
        This function finds the maximum profit H under our new policy
        """

        par = self.par
        sol = self.sol
        
        # a. Using loop to calculate profits
        for kappa_t in par.kappa_vals.T:
            # i. Using new policy_func_delta to calculate the number of employees
            l_t = self.policy_func_delta(kappa_t, par.l_t1, par.Delta)

            # ii. Calculate profits and append it to profits list
            sol.profits.append(self.ex_post_val(l_t, par.l_t1, kappa_t))

            # iii. Update previous l values to the current l values
            par.l_t1 = l_t

        # b. Finding the mean of the sum of the profits
        H = np.mean(np.sum(sol.profits, axis=0))

        # c. Printing the result
        print('The maximum profit, H: {:.4f}'.format(H)) 


    # 4. Defining funtion for using in optimizer
    def profit_max4(self, Delta):
        """
        This function finds the Delta value which maximizes H
        
        Args:
            Delta (float): Threshold value for the policy
            
        Returns:
            H (float): Maximum profit (negative value as we want to maximize H)
        """
        par = self.par
        sol = self.sol

        # a. Initializing profit list
        sol.profits = []
     
        # b. using loop to calculate profits and appending it to profits list
        for kappa_t in par.kappa_vals.T:
            # i. Using new policy_func_delta to calculate the number of employees
            l_t = self.policy_func_delta(kappa_t, par.l_t1, Delta)

            # ii. Calculate profits and append it to profits list
            sol.profits.append(self.ex_post_val(l_t, par.l_t1, kappa_t))

            # iii. Update previous l values to the current l values
            par.l_t1 = l_t
        
        # c. Finding the mean of the sum of the profits
        H = np.mean(np.sum(sol.profits, axis=0))

        # d. Returning negative H as we want to maximize H
        return -H 
    

    # 5. Defining function for finding optimal Delta
    def opt_delta(self):
        """
        This function finds the optimal Delta value which maximizes H
        """

        # a. Setting initial guess
        initial_guess = 0.05

        # b. Using minimize to find the optimal Delta value
        result = minimize(self.profit_max4, initial_guess, method='Nelder-Mead')

        # c. Setting optimal values
        optimal_delta = result.x[0]
        optimal_H = -self.profit_max4(optimal_delta)  

        # d. Printing results
        print("Optimal Delta is: {:.4f}".format(optimal_delta))
        print('The maximum profit, H: {:.4f}'.format(optimal_H))


    # 6. Defining function for finding optimal Delta with tolerance
    def profit_max5(self):
        par = self.par
        sol = self.sol

        # a. Creating arrays
        l_t1 = np.zeros(par.K)
        
        exp_kappa_vals = np.empty_like(par.kappa_vals)

        # b. Using loop to calculate expected_kappa_vals with rolling mean of the last 5 periods
        for t in range(par.kappa_vals.shape[1]):
            if t == 0:
                exp_kappa_vals[:, t] = par.kappa_vals[:, t]
            else:
                exp_kappa_vals[:, t] = np.mean(par.kappa_vals[:, max(0, t-5):t], axis=1)
        
        # c. Redefine the employee function to use expected_kappa
        def exp_employee_func(kappa, exp_kappa, eta, w):
            return ((1-eta)*(par.alpha*kappa+(1-par.alpha)*exp_kappa)/(w))**(1/eta)

        # d. Create loop to cycle through kappa values and calculate profits
        for t, kappa_t in enumerate(par.kappa_vals.T):
            # i. Using employee_func to calculate the number of employees
            exp_kappa_t = exp_kappa_vals[:, t]
            l_t = exp_employee_func(kappa_t, exp_kappa_t, par.eta, par.w)

            # ii. Checks whether every element in the array is within the tolerance 
            l = np.where(np.abs(l_t1 - l_t) > par.Delta, l_t, l_t1)

            # iii. Calculate profits and append it to profits list
            sol.profits.append(self.ex_post_val(l, l_t1, kappa_t))

            # iv. Update previous l values to the current l values
            l_t1 = l

        # e. Calculate H by first summing the rows and then calculate the mean value
        H = np.mean(np.sum(sol.profits, axis=0))

        # f. Print result
        print(f'The maximum profit, H = {H:.4f}')











# Creating definitions for Problem 3 

# 1. Defining global optimizer
def global_optimize(func, lower_bound, upper_bound, K_warm, K_max, tau):
    
    # a. Set best_x to None
    opt_x = None

    # b. Set best_y to infinity, as we want a minimum value
    opt_y = np.inf

    # c. Initialize list to store initial guesses
    x_initial_guesses = []

    # d. Initialize iterations counter
    iterations = 0

    for k in range(K_max):
        iterations += 1
        # i. drawing random uniformly distributed x_k values within our bounds
        x_k = np.random.uniform(lower_bound, upper_bound, 2)

        # ii. if k < K_warm we skip to step iv
        if k > K_warm:
            # ii.a Calculate chi_k
            chi_k = 0.50 * (2 / (1 + np.exp((k - K_warm) / 100)))

            # ii.b Setting x_k0
            x_k = chi_k * x_k + (1 - chi_k) * opt_x

        # iii. Store x_k values for visualization
        x_initial_guesses.append(x_k)
        
        # iv. Run BFGS optimizer with our x_k values 
        result = minimize(func, x_k, method='BFGS', tol=tau)
        
        # v. Set result for x and y
        if result.success and (k == 0 or result.fun < opt_y):
            opt_x = result.x
            opt_y = result.fun
            
            # vi. Testing if we are within our tolerance
            if opt_y < tau:
                break

    return opt_x, opt_y, x_initial_guesses, iterations

# 2. Defining function that repeats the global optimizer in order to ensure we find the minimum value
def repeated_global_optimize(func, lower_bound, upper_bound, K_warm, K_max, tau, repeat):
    
    # a. Set overall values to None or infinity
    overall_sol_y = np.inf
    overall_sol_x = None
    overall_sol_x_initial_guesses = None

    # b. Repeat the global optimizer
    for _ in range(repeat):
        # i. Run the global optimizer
        opt_x, opt_y, x_initial_guesses, iterations = global_optimize(func, lower_bound, upper_bound, K_warm, K_max, tau)
        
        # ii. If this run found a better solution, update the overall best solution
        if opt_y < overall_sol_y:
            overall_sol_y = opt_y
            overall_sol_x = opt_x
            overall_sol_x_initial_guesses = x_initial_guesses
            overall_iterations = iterations
    
    return overall_sol_x, overall_sol_y, overall_sol_x_initial_guesses, overall_iterations