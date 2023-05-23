from types import SimpleNamespace

import numpy as np
from scipy import optimize
import ipywidgets as widgets
import pandas as pd 
import matplotlib.pyplot as plt

class SolowModel:

    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # 1. Parameters used for numerical and visual solution
        # a. Setting realtistic values for our parameters
        par.alpha_v = 0.33
        par.phi_v = 0.4
        par.delta_v = 0.06
        par.g_v = 0.02
        par.n_v = 0.01
        par.sK_v = 0.27
        par.sH_v = 0.27

        # b. Generating the values for our k_tilde
        par.k_values = np.linspace(0, 2500, 1000)


        # c. Setting initial guesses for our parameters
        par.alpha_guess = 0.33
        par.phi_guess = 0.4
        par.sK_guess = 0.27
        par.sH_guess = 0.27


        # 2. Parameters only for the simulations 
        par.k0 = 1
        par.h0 = 1
        par.a0 = 1
        par.l0 = 1
        par.t0 = 0
        par.T = 250
        

        # 3. Parameters only for the simulation with extension
        par.alpha_vE = 0.25
        par.phi_vE = 0.25
        par.beta_vE = 0.25
        par.epsilon_vE = 0.25
        par.sR_v = 0.1
        par.r0 = 100


        # 4. Arrays for our shock simulation
        sol.k = np.zeros(par.T+1)
        sol.h = np.zeros(par.T+1)
        sol.y = np.zeros(par.T+1)
        sol.l = np.zeros(par.T+1)
        sol.a = np.zeros(par.T+1)
        sol.y = np.zeros(par.T+1)
        

        # Defining the time array
        sol.t = np.arange(par.T)


        # 5. Arrays for our extension simulation
        sol.ext_k = np.zeros(par.T+1) # Physical capital
        sol.ext_h = np.zeros(par.T+1) # Human capital
        sol.ext_y = np.zeros(par.T+1) # Output
        sol.ext_l = np.zeros(par.T+1) # Workers
        sol.ext_a = np.zeros(par.T+1) # Technology
        sol.ext_e = np.zeros(par.T+1) # Amount of oil extracted
        sol.ext_r = np.zeros(par.T+1) # Oil remaining
        sol.ext_yG = [] # Growth in output


    # Defining the plot for our nullclines
    def phasediagram(self, k_null_py, h_null_py, alpha_guess, phi_guess, sK_guess, sH_guess):
        par = self.par
        # Inserting k_values into our nullclines
        k_null_display = k_null_py(par.k_values, alpha_guess, par.delta_v, par.g_v, par.n_v, phi_guess, sK_guess)
        h_null_display = h_null_py(par.k_values, alpha_guess, par.delta_v, par.g_v, par.n_v, phi_guess, sH_guess)

        # Making plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(par.k_values, k_null_display)
        ax.plot(par.k_values, h_null_display)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # setting labels and title
        ax.set_title('Phasediagram')
        ax.set_xlabel('k_tilde')
        ax.set_ylabel('h_tilde')
        ax.grid(True, linewidth=1)

    # par.sK_v, par.sH_v, par.alpha_v, par.phi_v, par.delta_v, par.n_v, par.g_v

    def k_trans_func(self, k_tilde, h_tilde):
        par = self.par
        return (h_tilde**par.phi_v * k_tilde**par.alpha_v *par.sK_v +k_tilde*(1-par.delta_v))/((par.g_v+1) * (par.n_v+1)) -k_tilde

    # Same for human capital
    def h_trans_func(self, k_tilde, h_tilde):
        par = self.par
        return (h_tilde**par.phi_v * k_tilde**par.alpha_v *par.sH_v +h_tilde*(1-par.delta_v))/((par.g_v+1) * (par.n_v+1)) -h_tilde    

    def Num_SS(self):
        par = self.par
        objective = lambda x: [self.k_trans_func(x[0], x[1]), self.h_trans_func(x[0], x[1])]  
    
        # Setting initial guess
        initial_guess =[1,1]
    
        # Creating optimizer
        result = optimize.root(objective, initial_guess, method = "broyden1")
    
        print(f'From our optimizer we get the numerical solution: k* = {result.x[0]:.2f}, h* = {result.x[1]:.2f}')

    def sK_shock(self, shock):
    # Creating a for loop that runs the simulation for T periods

        par = self.par
        sol = self.sol
        

        # Storing initial values in our arrays
        sol.k[0] = par.k0
        sol.h[0] = par.h0
        sol.a[0] = par.a0
        sol.l[0] = par.l0
        sol.yG = []

        for i in range(par.T):
            # if-statement that creates permenant shock to sK from periode 3 and onwards
            if i <= 1:
                par.sK_v = 0.1
            else:
                par.sK_v = shock    

            # The equations that make up the model
            sol.y[i+1] = sol.k[i]**par.alpha_v * sol.h[i]**par.phi_v * (sol.a[i]*sol.l[i])**(1-par.alpha_v-par.phi_v)
            sol.l[i+1] = (1+par.n_v) * sol.l[i]
            sol.a[i+1] = (1+par.g_v) * sol.a[i]
            sol.k[i+1] = par.sK_v * sol.y[i] + (1-par.delta_v) * sol.k[i]
            sol.h[i+1] = par.sH_v * sol.y[i] + (1-par.delta_v) * sol.h[i]

            # Calculating the groth in y
            sol.yG.append(np.log(sol.y[i+1])-np.log(sol.y[i]))

        # Plotting the simulation
        fig = plt.figure(figsize=(13,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(sol.yG[2:], color="blue")
        plt.axhline(sol.yG[249],xmax=1,color="black",linestyle="--")
        ax.set_ylabel('Growth in y')
        ax.set_xlabel('Time')
        ax.set_title('Growth in y, solow model with human capital')
        ax.legend()



    def extension(self):
        par = self.par
        sol = self.sol
        
        # Storing initial values in our arrays
        sol.k[0] = par.k0
        sol.h[0] = par.h0
        sol.a[0] = par.a0
        sol.l[0] = par.l0
        sol.yG = []

        for i in range(par.T):
            sol.y[i+1] = sol.k[i]**par.alpha_v * sol.h[i]**par.phi_v * (sol.a[i]*sol.l[i])**(1-par.alpha_v-par.phi_v)
            sol.l[i+1] = (1+par.n_v) * sol.l[i]
            sol.a[i+1] = (1+par.g_v) * sol.a[i]
            sol.k[i+1] = par.sK_v * sol.y[i] + (1-par.delta_v) * sol.k[i]
            sol.h[i+1] = par.sH_v * sol.y[i] + (1-par.delta_v) * sol.h[i]

            # Calculating the groth in y
            sol.yG.append(np.log(sol.y[i+1])-np.log(sol.y[i]))


        sol.ext_k[0] = par.k0
        sol.ext_h[0] = par.h0
        sol.ext_a[0] = par.a0
        sol.ext_l[0] = par.l0
        sol.ext_r[0] = par.r0

        for i in range(par.T):
            sol.ext_e[i+1] = par.sR_v*sol.ext_r[i] # Amount of oil extracted
            sol.ext_r[i+1] = sol.ext_r[i] - sol.ext_e[i] # Oil remaining for next period
            sol.ext_y[i+1] = sol.ext_k[i]**par.alpha_vE * sol.ext_h[i]**par.phi_vE * (sol.ext_a[i]*sol.ext_l[i])**par.beta_vE *sol.ext_e[i]**par.epsilon_vE # Output next period
            sol.ext_l[i+1] = (1+par.n_v) * sol.ext_l[i] # Workers next periode
            sol.ext_a[i+1] = (1+par.g_v) * sol.ext_a[i] # Technology next period
            sol.ext_k[i+1] = par.sK_v * sol.ext_y[i] + (1-par.delta_v) * sol.ext_k[i] # Physical capital next period
            sol.ext_h[i+1] = par.sH_v * sol.ext_y[i] + (1-par.delta_v) * sol.ext_h[i] # Human capital next period
            sol.ext_yG.append(np.log(sol.ext_y[i+1])-np.log(sol.ext_y[i])) # Growth in output appended to list

        # Plotting the simulation
        fig = plt.figure(figsize=(13,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(sol.ext_yG[1:], color="orange")
        ax.plot(sol.yG[1:], color="blue")
        plt.axhline(sol.yG[249],xmax=1,color="black",linestyle="--")
        plt.axhline(sol.ext_yG[249],xmax=1, color="black", linestyle="--")
        ax.set_ylabel('Growth in y')
        ax.set_xlabel('Time')
        ax.set_title('Growth in y, solow model with human capital and oil')
        ax.legend()

        plt.show()