from types import SimpleNamespace

import numpy as np
from scipy import optimize
import ipywidgets as widgets
import pandas as pd 
import matplotlib.pyplot as plt

class SolowModel:

    # 1. Defining the initiation function
    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. Parameters used for all parts
        # i. Setting realtistic values for our parameters
        par.alpha_v = 0.33 # Share of physical capital in production
        par.phi_v = 0.4 # Share of human capital in production
        par.delta_v = 0.06 # Depreciation rate of physical capital
        par.g_v = 0.02 # Growth rate of technology
        par.n_v = 0.01 # Growth rate of workers
        par.sK_v = 0.27 # Saving rate of physical capital
        par.sH_v = 0.27 # Saving rate of human capital

        # ii. Generating the values for our k_tilde
        par.k_values = np.linspace(0, 2500, 1000)


        # iii. Setting initial guesses for our parameters
        par.alpha_guess = 0.33
        par.phi_guess = 0.4
        par.sK_guess = 0.27
        par.sH_guess = 0.27


        # b. Parameters only for part 6 & 7 
        par.k0 = 1
        par.h0 = 1
        par.a0 = 1
        par.l0 = 1
        par.t0 = 0
        par.T = 250
        

        # c. Parameters only for part 7
        par.alpha_vE = 0.25
        par.phi_vE = 0.25
        par.beta_vE = 0.25
        par.epsilon_vE = 0.25
        par.sR_v = 0.1
        par.r0 = 100


        # d. Arrays for part 6 & 7
        # i. Arrays to hold values for baseline estimation
        sol.k = np.zeros(par.T+1) # Physical capital
        sol.h = np.zeros(par.T+1) # Human capital
        sol.y = np.zeros(par.T+1) # Output
        sol.l = np.zeros(par.T+1) # Workers
        sol.a = np.zeros(par.T+1) # Technology
       
        

        # ii. Defining the time array
        sol.t = np.arange(par.T)


        # e. Arrays for pary 7
        # Arrays to hold values for baseline estimation
        sol.ext_k = np.zeros(par.T+1) # Physical capital
        sol.ext_h = np.zeros(par.T+1) # Human capital
        sol.ext_y = np.zeros(par.T+1) # Output
        sol.ext_l = np.zeros(par.T+1) # Workers
        sol.ext_a = np.zeros(par.T+1) # Technology
        sol.ext_e = np.zeros(par.T+1) # Amount of oil extracted
        sol.ext_r = np.zeros(par.T+1) # Oil remaining
        sol.ext_yG = [] # Growth in output


    # 2. Defining the transition equaiton for physical capital    
    def k_trans_func(self, k_tilde, h_tilde):
        """This function defines the transition function for physical capital.
        
        It takes the following arguments:
        k_tilde: The value of physical capital
        h_tilde: The value of human capital
        
        The function returns the transition function for physical capital.
        """

        par = self.par
        return (h_tilde**par.phi_v * k_tilde**par.alpha_v *par.sK_v +k_tilde*(1-par.delta_v))/((par.g_v+1) * (par.n_v+1)) -k_tilde

    # 2. Defining the transition equaiton for human capital
    def h_trans_func(self, k_tilde, h_tilde):
        """This function defines the transition function for human capital.
        
        It takes the following arguments:
        k_tilde: The value of physical capital
        h_tilde: The value of human capital
        
        The function returns the transition function for human capital.
        """

        par = self.par
        return (h_tilde**par.phi_v * k_tilde**par.alpha_v *par.sH_v +h_tilde*(1-par.delta_v))/((par.g_v+1) * (par.n_v+1)) -h_tilde    

    # 3. Solving SS numerically
    def Num_SS(self):
        """
        This function solves for the steady state values of physical and human capital numerically.
        
        It takes no arguments.
        
        The function returns the steady state values of physical and human capital.
        """

        # a. Defining the objective function
        objective = lambda x: [self.k_trans_func(x[0], x[1]), self.h_trans_func(x[0], x[1])]  
    
        # b. Setting initial guess
        initial_guess =[1,1]
    
        # c. Creating optimizer
        result = optimize.root(objective, initial_guess, method = "broyden1")

        # d. Printing the results
        print(f'Using the Broyden algorithm in out numerical optimizer we get k* = {result.x[0]:.2f}, h* = {result.x[1]:.2f}')


    # 4. Creating phase diagram and plotting nullclines
    def phasediagram(self, k_null_py, h_null_py, alpha_guess, phi_guess, sK_guess, sH_guess):
        """This function plots the nullclines of the model. 
        
        It takes the following arguments:
        k_null_py: The nullcline for physical capital
        h_null_py: The nullcline for human capital
        alpha_guess: Initial guess for alpha
        phi_guess: Initial guess for phi
        sK_guess: Initial guess for sK
        sH_guess: Initial guess for sH
        
        The function returns a plot of the nullclines.
        """

        par = self.par

        # a. Inserting k_values into our nullclines
        k_null_display = k_null_py(par.k_values, alpha_guess, par.delta_v, par.g_v, par.n_v, phi_guess, sK_guess)
        h_null_display = h_null_py(par.k_values, alpha_guess, par.delta_v, par.g_v, par.n_v, phi_guess, sH_guess)

        # b. Making plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(par.k_values, k_null_display)
        ax.plot(par.k_values, h_null_display)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # c. Setting labels and title
        ax.set_title('Phasediagram')
        ax.set_xlabel(r"$\tilde{k_t}$")
        ax.set_ylabel(r"$\tilde{h_t}$")
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        plt.rcParams['text.usetex'] = True

        # d. Setting grid
        ax.grid(True, linewidth=1)

        


    # 5. Creating simulation with interactive shock
    def sK_shock(self, shock):
        """This function simulates the model with an interactive shock to sK.

        It takes the following arguments:
        shock: The shock to sK

        The function returns a plot of the simulation.
        """

        par = self.par
        sol = self.sol

        # a. Storing initial values in our arrays
        sol.k[0] = par.k0
        sol.h[0] = par.h0
        sol.a[0] = par.a0
        sol.l[0] = par.l0
        sol.yG = []

        # b. Simulating and creating shock in period 2 and onwards
        for i in range(par.T):
            # i. Setting sK to 0.1 in period 1
            if i <= 1:
                par.sK_v = 0.1
            # ii. else equal to our chosen shock size
            else:
                par.sK_v = shock    

            # iii. The equations that make up the model
            sol.y[i+1] = sol.k[i]**par.alpha_v * sol.h[i]**par.phi_v * (sol.a[i]*sol.l[i])**(1-par.alpha_v-par.phi_v)
            sol.l[i+1] = (1+par.n_v) * sol.l[i]
            sol.a[i+1] = (1+par.g_v) * sol.a[i]
            sol.k[i+1] = par.sK_v * sol.y[i] + (1-par.delta_v) * sol.k[i]
            sol.h[i+1] = par.sH_v * sol.y[i] + (1-par.delta_v) * sol.h[i]

            # iv. Calculating the growth in y and appending to list
            sol.yG.append(np.log(sol.y[i+1])-np.log(sol.y[i]))

        # c. Creating plot
        fig = plt.figure(figsize=(13,5))
        ax = fig.add_subplot(1,2,1)

        # d. Plotting the simulation
        ax.plot(sol.yG[2:], color="blue", label="Human capital")
        plt.axhline(sol.yG[249],xmax=1,color="black",linestyle="--")
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.00))

        # e. Setting labels and title
        ax.set_ylabel('Growth in y')
        ax.set_xlabel('Time')
        ax.set_title('Growth in y, solow model with human capital')
        ax.set_xlim(-2, 250)
        ax.grid(True)

        plt.show()
        


    # 6. Defining extension model
    def extension(self):
        """
        This function simulates the model with the extension of human capital and oil.
        
        It takes no arguments.

        The function returns a plot of the simulation and of the baseline model.
        """

        par = self.par
        sol = self.sol

        # BASELINE MODEL

        # a. Storing initial values in our arrays
        sol.k[0] = par.k0
        sol.h[0] = par.h0
        sol.a[0] = par.a0
        sol.l[0] = par.l0
        sol.yG = []

        # b. Simulating the model
        for i in range(par.T):
            # i. The equations that make up the model
            sol.y[i+1] = sol.k[i]**par.alpha_v * sol.h[i]**par.phi_v * (sol.a[i]*sol.l[i])**(1-par.alpha_v-par.phi_v)
            sol.l[i+1] = (1+par.n_v) * sol.l[i]
            sol.a[i+1] = (1+par.g_v) * sol.a[i]
            sol.k[i+1] = par.sK_v * sol.y[i] + (1-par.delta_v) * sol.k[i]
            sol.h[i+1] = par.sH_v * sol.y[i] + (1-par.delta_v) * sol.h[i]

            # ii. Calculating the growth in y
            sol.yG.append(np.log(sol.y[i+1])-np.log(sol.y[i]))


        # EXTENSION MODEL
        
        # a. Storing initial values in our arrays
        sol.ext_k[0] = par.k0
        sol.ext_h[0] = par.h0
        sol.ext_a[0] = par.a0
        sol.ext_l[0] = par.l0
        sol.ext_r[0] = par.r0

        # b. Simulating the model
        for i in range(par.T):
            # i. The equations that make up the model
            sol.ext_e[i+1] = par.sR_v*sol.ext_r[i] 
            sol.ext_r[i+1] = sol.ext_r[i] - sol.ext_e[i] 
            sol.ext_y[i+1] = sol.ext_k[i]**par.alpha_vE * sol.ext_h[i]**par.phi_vE * (sol.ext_a[i]*sol.ext_l[i])**par.beta_vE *sol.ext_e[i]**par.epsilon_vE 
            sol.ext_l[i+1] = (1+par.n_v) * sol.ext_l[i] 
            sol.ext_a[i+1] = (1+par.g_v) * sol.ext_a[i] 
            sol.ext_k[i+1] = par.sK_v * sol.ext_y[i] + (1-par.delta_v) * sol.ext_k[i] 
            sol.ext_h[i+1] = par.sH_v * sol.ext_y[i] + (1-par.delta_v) * sol.ext_h[i] 

            # ii. Calculating the growth in y
            sol.ext_yG.append(np.log(sol.ext_y[i+1])-np.log(sol.ext_y[i])) 

        # c. Creating plot
        fig = plt.figure(figsize=(13,5))
        ax = fig.add_subplot(1,2,1)

        # d. Plotting the simulation
        ax.plot(sol.yG[1:], color="blue",label="Baseline: Human capital")
        ax.plot(sol.ext_yG[1:], color="orange",label="Extension: Human capital and oil")
        plt.axhline(sol.yG[249],xmax=1,color="black",linestyle="--")
        plt.axhline(sol.ext_yG[249],xmax=1, color="black", linestyle="--")
        plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.00))

        # e. Setting labels and title
        ax.set_ylabel('Growth in y')
        ax.set_xlabel('Time')
        ax.set_title('Comparison of growth in y, Solow model with and without extension')
        ax.set_xlim(-2, 250)
        ax.grid(True)

        plt.show()