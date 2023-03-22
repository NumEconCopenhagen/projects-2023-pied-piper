#Jeppes kode starter her:

from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha

        elif par.sigma == 0:
            H = np.minimum(HM, HF)

        else:
            H = ((1-par.alpha) * HM ** ((par.sigma-1)/par.sigma) + par.alpha * HF ** ((par.sigma-1)/par.sigma)) ** ((par.sigma)/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continously(self,do_print=False):
        """ solve model continously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # Define the objective for our model
        def objective(x):
            LM,HM,LF,HF = x
            return -self.calc_utility(LM,HM,LF,HF)
        
        

        # Firstly we define the constraints for both male and female 
        def constraint1(x):
            LM,HM,LF,HF = x   
            return 24-(LM+HM)

        def constraint2(x):
            LM,HM,LF,HF = x
            return 24-(LF+HF)

        
        # Create intial guess for our model to use
        initial_guess = [12,12,12,12]
    

        # We combine our two constraints using inequality 
        constraints = [{"type": "ineq", "fun": constraint1}, 
                        {"type": "ineq", "fun": constraint2}]
        
        
        result = optimize.minimize(objective, initial_guess, 
                                   method='SLSQP',constraints=constraints)
        
        sol.LM = result.x[0]
        sol.HM = result.x[1]
        sol.LF = result.x[2]
        sol.HF = result.x[3]

       

        
    
        
        

    
 

     

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        var_WF = [0.8,0.9,1.0,1.1,1.2]

        list_HFHM_q4 = []
        list_WFWM_q4 = []

        for w in var_WF :
            model = HouseholdSpecializationModelClass()
            model.par.wF = w
            model.solve_continously()

            HFHM = model.sol.HF/model.sol.HM #HFHM forhold defineres
            list_HFHM_q4.append(HFHM)


            WFWM = w/model.par.wM # vi finder forholdet mellem wF og wM
            list_WFWM_q4.append(WFWM)

        log_sol_WFWM_q4 = np.log(list_WFWM_q4) # Der tages log til forholdene
        log_sol_HFHM_q4 = np.log(list_HFHM_q4)
            

        


        

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol
        
        x = np.log(par.wF)
        y = np.log(sol.HF/sol.HM)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass