from sympy import *
import numpy as np
from numpy.linalg import multi_dot, inv


"""
This class performs a kinematic fit on set of parameters given a set of
measured parameters and analytical constraints. The derivatives of the
constraint vector are computed analytically. The minimisation is performed
using iteratively the Lagrange multiplier method using definitions and
conventions from A. G. Frodesen, 0. Skjeggestad Chap. 10.8
"""

class KinematicFit:
    def __init__(self, measurements, unmeasurables, constraints):

        """ vector of measurements eta of size N """
        self.measurements = measurements

        """ vector of non measurables parameters chi of size J """
        self.unmeasurables = unmeasurables

        """ vector of constraints f(eta, chi) of size K """
        self.constraints = constraints

        """ compute matrix of df/deta (K,N) and df/dchi (K,J)"""
        self.constraints_derivatives = self.compute_constraints_derivatives(measurements, unmeasurables, constraints)

        """ fit parameters """
        self.iterations = 100  # how many loops
        self.maxchisqdiff = 1.e-06 # criterion for convergence
        self.weight = 1. # for linear constraints use w = 1, smaller for higher


    """ compute matrices of dHi/daj expressions of size (r x n) """
    def compute_constraints_derivatives(self, measurements, unmeasurables, constraints):

        df_deta = []
        df_dchi = []

        # these are expressions
        for i in range(len(constraints)):
            dfi_deta = []
            dfi_dchi = []
            for j in range(len(measurements)):
                constrain_derivative = Derivative(constraints[i], measurements[j]).doit()
                dfi_deta.append(constrain_derivative)
            df_deta.append(dfi_deta)
            for j in range(len(unmeasurables)):
                constrain_derivative = Derivative(constraints[i], unmeasurables[j]).doit()
                dfi_dchi.append(constrain_derivative)
            df_dchi.append(dfi_dchi)

        return df_deta, df_dchi

    """ print expressions of constraints """
    def print_constraints(self):
        K = len(self.constraints)
        print('expressions of constraints:')
        for i in range(K):
            print("  constraint expression : {}".format(self.constraints[i]))

    """ print expressions of constraints """
    def print_constraints_derivatives(self):
        K = len(self.constraints)
        N = len(self.measurements)
        J = len(self.unmeasurables)
        print('expressions of derivatives:')
        for i in range(K):
            print('  df/eta:')
            for j in range(N):
                print("    df{}/d{} = {}".format(i+1,self.measurements[j], self.constraints_derivatives[0][i][j]))
            print('  df/chi:')
            for j in range(J):
                print("    df{}/d{} = {}".format(i+1,self.unmeasurables[j], self.constraints_derivatives[1][i][j]))


    """ create mapping between symbols and value """
    def evaluate_measurements(self, eta):
        sub_list = []
        rows = eta.shape[0]
        for i in range(rows):
            sub_list.append((self.measurements[i],eta[i][0]))
        return sub_list

    """ create mapping between symbols and value """
    def evaluate_unmeasurables(self, chi):
        sub_list = []
        rows = chi.shape[0]
        for i in range(rows):
            sub_list.append((self.unmeasurables[i],chi[i][0]))
        return sub_list


    """ evaluate vector of constraints f(eta, chi)"""
    def evaluate_constraints(self, replacements):
        f = np.zeros((len(self.constraints),1))
        for i in range(f.shape[0]):
            fi = self.constraints[i]
            f[i][0] = fi.subs(replacements)
        return f

    """ evaluate vector of constraints derivative Feta(eta, chi) """
    def evaluate_constraint_derivative_eta(self, replacements):
        Feta = np.zeros((len(self.constraints),len(self.constraints_derivatives[0][0])))
        for i in range(Feta.shape[0]):
            for j in range(Feta.shape[1]):
                dHi_dxj = self.constraints_derivatives[0][i][j]
                Feta[i][j] = dHi_dxj.subs(replacements)
        return Feta

    """ evaluate vector of constraints derivative Fchi(eta, chi) """
    def evaluate_constraint_derivative_chi(self, replacements):
        Fchi = np.zeros((len(self.constraints),len(self.constraints_derivatives[1][0])))
        for i in range(Fchi.shape[0]):
            for j in range(Fchi.shape[1]):
                dHi_dxj = self.constraints_derivatives[1][i][j]
                Fchi[i][j] = dHi_dxj.subs(replacements)
        return Fchi


    """ run kinematic fit """
    def results(self, y, V, eta0, chi0, options):

        # initialize options
        self.iterations   = options[0]
        self.maxchisqdiff = options[1]
        self.weight       = options[2]

        ## start minimisation here
        converged=False
        i=0
        chisqrd=0.
        chisqrd_last=0.

        K = len(self.constraints)
        N = len(self.measurements)
        J = len(self.unmeasurables)

        # initialise measurements and unmeasurables
        eta = eta0
        chi = chi0
        chisqrd 

        # iteratively solve
        while not converged and i < self.iterations:

            eta_val = self.evaluate_measurements(eta)
            chi_val = self.evaluate_unmeasurables(chi)
            eta_chi_val = eta_val + chi_val

            print(eta_val, chi_val, eta_chi_val)


            f = self.evaluate_constraints(eta_chi_val)
            Feta = self.evaluate_constraint_derivative_eta(eta_chi_val)
            Fchi = self.evaluate_constraint_derivative_chi(eta_chi_val)
            FetaT = np.transpose(Feta)
            FchiT = np.transpose(Fchi)

            r = f + Feta.dot(y-eta)
            S = multi_dot([Feta, V, FetaT])
            Sinv = inv(S)
            print(r)
            print(S)
            print(Sinv)
            Ainv = inv(multi_dot([FchiT, Sinv, Fchi]))
            chi_new = chi - multi_dot([Ainv, FchiT, Sinv, r])
            print(chi_new)
            lmbda_new = Sinv.dot(r + Fchi.dot(chi_new - chi))
            eta_new = y - multi_dot([V, FetaT, lmbda_new])


            eta_val_new = self.evaluate_measurements(eta_new)
            chi_val_new = self.evaluate_unmeasurables(chi_new)
            eta_chi_val_new = eta_val_new + chi_val_new

            lmbda_newT = np.transpose(lmbda_new)
            f_new = self.evaluate_constraints(eta_chi_val_new)
            chisq_new = multi_dot([lmbda_newT, S, lmbda_new]) + 2*lmbda_newT.dot(f_new)
            print('chi2', chisq_new, multi_dot([lmbda_newT, S, lmbda_new]), 2*lmbda_newT.dot(f_new))
            print(eta_new, eta)

            if abs(chisqrd - chisqrd_last) > self.maxchisqdiff:


            else:
                converged=True # good enough fit



            '''
            d = self.evaluate_constraints(symb_to_val)
            D = self.evaluate_constraint_derivative(symb_to_val)
            DT = np.transpose(D)
            VD = inv(multi_dot([D, Va, DT]))
            Lambda = VD.dot(d) ## fix me: could be as well V * (d+D*delta_a)
            LambdaT = np.transpose(Lambda)
            delta_a = - self.weight * multi_dot([Va, DT, Lambda])
            delta_a_T = np.transpose(delta_a)
            # fitted values and new covariance matrix
            a = a + delta_a
            Va = Va - Va * DT * VD * D * Va * self.weight
            #chisqrd = np.trace(Va)
            chisqrd = multi_dot([delta_a_T, inv(Va), delta_a]) + 2*LambdaT.dot(d + D.dot(delta_a))

            if i==0: firstchisqrd = chisqrd
            i += 1

            if abs(chisqrd - chisqrd_last) < self.maxchisqdiff:
                 converged=True # good enough fit
            '''

            converged=True
        return (eta, chi, V, chisqrd_last, i)
