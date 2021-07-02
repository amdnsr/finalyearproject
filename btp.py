import copy
import math
import os
import shutil
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import ode, odeint, simps
from tqdm import tqdm

plt.rcParams.update({'figure.max_open_warning': 0})

# https://chemistrygod.com/amagat#:~:text=1%20amagat%20equals%20the%20number,%2C%20P%20%3D%201%20atm).
# https://codereview.stackexchange.com/questions/25672/how-to-handle-returned-value-if-an-exception-happens-in-a-library-code


class DPAL():
    def __init__(self, constants, max_allowable_error, max_number_of_iterations=10):
        self.constants = constants
        self.max_allowable_error = max_allowable_error
        self.max_number_of_iterations = max_number_of_iterations
        self.solve_parameters = {}
        N = self.constants["N"]
        self.Q_list = [0]*(N+1)
        self.T_list = [0]*(N+2)
        self.Omega_list = [0]*(N+1)
        self.r_list = [0]*(N+2)
        self.n_0_list = [0]*(N+1)
        self.n_1_list = [0]*(N+1)
        self.n_2_list = [0]*(N+1)
        self.n_3_list = [0]*(N+1)
        self.F_list = [0]*(N+1)
        self.P_j_list = [0]*(N+1)
        self.P_L_j_list = [0]*(N+1)
        self.folder_name = "./results/" + "w_P" + "_" + str(int(self.constants["w_P"]/1e-6)) + "_" + "T_w"\
            + "_" + str(self.constants["T_w"]) + "_" + "P_P" + "_" + str(self.constants["P_P"]) + "_"\
            + "flow" + "_" + \
            str(self.constants["is_gas_flowing"]) + \
            "_U_" + str(self.constants["U"]) + "/"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        else:
            self.clearFolderContents(self.folder_name)
        # self.lambda_list = []
        # self.P_j_lambda_wo_P_j_peak_list = []

        self.add_calculated_constants()
        self.number_of_iterations = 0
        self.is_gas_flowing = self.constants["is_gas_flowing"]

    def add_calculated_constants(self):
        R_SI = self.constants["R_SI"]
        T_cell = self.constants["T_cell"]
        # Total_Volume = math.pi*R*R*L
        torr_to_bar = 0.00133322
        NA = self.constants["NA"]
        # per_unit_m3
        nHe = self.constants["P_He_Torr"]*torr_to_bar*1e5*NA/(R_SI*T_cell)
        self.constants["nHe"] = nHe
        nEthane = self.constants["P_Ethane_Torr"] * \
            torr_to_bar*1e5*NA/(R_SI*T_cell)
        self.constants["nEthane"] = nEthane
        c = self.constants["c"]
        lambda_D1_nm = self.constants["lambda_D1_nm"]
        nu_L_GHz = c/lambda_D1_nm
        self.constants["nu_L_GHz"] = nu_L_GHz
        self.constants["I_max"] = self.calc_I_max()
        self.calculate_constants_for_Gamma_P_j_numpy()

    def calculate_constants_for_Gamma_P_j_numpy(self):
        # code that needs to be executed only one time

        lambda_D2_nm = self.constants["lambda_D2_nm"]
        delta_lambda_D2_FWHM_nm = self.constants["delta_lambda_D2_FWHM_nm"]
        K_5 = delta_lambda_D2_FWHM_nm/2
        lambda_0_nm = self.constants["lambda_0_nm"]

        starting_lambda_nm = self.constants["starting_lambda_nm"]
        ending_lambda_nm = self.constants["ending_lambda_nm"]
        d_lambda_nm = self.constants["d_lambda_nm"]

        sigma_of_normal_distr = delta_lambda_D2_FWHM_nm / \
            (2*((2*math.log(2))**0.5))
        # inside_exponential = ((lambda_nm-lambda_0_nm)**2)/(2*sigma_of_normal_distr*sigma_of_normal_distr)
        pi = math.pi
        denominator = ((2*pi)**0.5)*sigma_of_normal_distr

        lambda_list = np.arange(
            starting_lambda_nm, ending_lambda_nm, d_lambda_nm)
        self.lambda_list = copy.deepcopy(lambda_list)
        self.constants["lambda_list"] = lambda_list
        # P_j_l_wo_P_j_p stands for P_j_lambda_wo_P_j_peak
        P_j_l_wo_P_j_p = lambda_list - lambda_0_nm
        P_j_l_wo_P_j_p = np.power(P_j_l_wo_P_j_p, 2)
        P_j_l_wo_P_j_p = P_j_l_wo_P_j_p / \
            (2*sigma_of_normal_distr*sigma_of_normal_distr)
        P_j_l_wo_P_j_p = np.exp(-1*P_j_l_wo_P_j_p)
        P_j_l_wo_P_j_p = P_j_l_wo_P_j_p/denominator
        self.P_j_lambda_wo_P_j_peak_list = copy.deepcopy(P_j_l_wo_P_j_p)

        lambda_nm_times_P_j_l_wo_P_j_p = np.multiply(
            lambda_list, P_j_l_wo_P_j_p)
        self.constants["lambda_nm_times_P_j_l_wo_P_j_p"] = lambda_nm_times_P_j_l_wo_P_j_p

        g = lambda_list
        g = g - lambda_D2_nm
        g = g/K_5
        g = np.power(g, 2)
        g = 1 + g
        g = np.power(g, -1)
        self.constants["g"] = g

    def calc_r_j(self, j):
        R = self.constants["R"]
        N = self.constants["N"]
        return R - (j-1)*R/N

    def calc_A_j(self, j):
        r_j = self.calc_r_j(j)
        l = self.constants["l"]
        pi = math.pi
        return 2*pi*r_j*l

    def calc_S_j(self, j):
        r_j = self.calc_r_j(j)
        r_j_p_1 = self.calc_r_j(j+1)
        pi = math.pi
        return pi*(r_j*r_j - r_j_p_1*r_j_p_1)

    def calc_V_j(self, j):
        r_j_p_1 = self.calc_r_j(j+1)
        r_j = self.calc_r_j(j)
        pi = math.pi
        L = self.constants["L"]
        return pi*(r_j**2 - r_j_p_1**2)*L

    def calc_nu_gamma_Ethane(self, T_j):
        kB = self.constants["kB"]
        mCs = self.constants["mCs"]
        mEthane = self.constants["mEthane"]
        val = 3*kB*T_j*(1/mCs + 1/mEthane)
        return val**0.5

    def calc_nu_gamma_He(self, T_j):
        kB = self.constants["kB"]
        mCs = self.constants["mCs"]
        mHe = self.constants["mHe"]
        val = 3*kB*T_j*(1/mCs + 1/mHe)
        return val**0.5

    def calc_gamma_3_2(self, T_j):
        nEthane = self.constants["nEthane"]
        sigma_3_2_Ethane = self.constants["sigma_3_2_Ethane"]
        nHe = self.constants["nHe"]
        sigma_3_2_He = self.constants["sigma_3_2_He"]
        nu_gamma_Ethane = self.calc_nu_gamma_Ethane(T_j)
        nu_gamma_He = self.calc_nu_gamma_He(T_j)
        return nEthane*sigma_3_2_Ethane*nu_gamma_Ethane + nHe*sigma_3_2_He*nu_gamma_He

    def calc_I_max(self):
        P_P = self.constants["P_P"]
        w_P = self.constants["w_P"]
        N = self.constants["N"]
        denominator = 0
        for j in range(1, N+1):
            r_j = self.calc_r_j(j)
            S_j = self.calc_S_j(j)
            inside_exponential = -2*((r_j/w_P)**2)
            denominator += math.exp(inside_exponential)*(S_j)
        ans = P_P/denominator
        return ans

    def calc_n_in_amagat_from_P_in_Torr(self, P_Torr, T_measurement):
        # https://chemistrygod.com/amagat#:~:text=1%20amagat%20equals%20the%20number,%2C%20P%20%3D%201%20atm).
        ans = (P_Torr/760)*(273.15/T_measurement)
        return ans

    def calc_sigma_D2_He_broadened(self, T_j):
        sigma_D2_radiative = self.constants["sigma_D2_radiative"]
        pi = math.pi
        tau_D2_ns = self.constants["tau_D2_ns"]
        P_He_Torr = self.constants["P_He_Torr"]

        # I take the temp to be 110 degree celcius, i.e. 383K, because the cell temeprature,
        # as mentioned in Section 1 Introduction, is 110 degree celcius
        T_cell = self.constants["T_cell"]
        n_He_amagat = self.calc_n_in_amagat_from_P_in_Torr(P_He_Torr, T_cell)
        ans = sigma_D2_radiative / \
            (2*pi*tau_D2_ns*n_He_amagat*19.3*((T_j/294)**0.5))
        return ans

    def calc_sigma_D2_T_j_lambda(self, T_j, lambda_nm):
        sigma_D2_He_broadened = self.calc_sigma_D2_He_broadened(T_j)
        lambda_D2_nm = self.constants["lambda_D2_nm"]
        delta_lambda_D2_FWHM_nm = self.constants["delta_lambda_D2_FWHM_nm"]
        denominator = 1 + ((lambda_nm-lambda_D2_nm) /
                           (delta_lambda_D2_FWHM_nm/2))**2
        ans = sigma_D2_He_broadened/denominator
        return ans

    def calc_sigma_D1_He_broadened(self, T_j):
        sigma_D1_radiative = self.constants["sigma_D1_radiative"]
        pi = math.pi
        tau_D1_ns = self.constants["tau_D1_ns"]
        P_He_Torr = self.constants["P_He_Torr"]

        # I take the temp to be 110 degree celcius, i.e. 383K, because the cell temeprature,
        # as mentioned in Section 1 Introduction, is 110 degree celcius
        T_cell = self.constants["T_cell"]
        n_He_amagat = self.calc_n_in_amagat_from_P_in_Torr(P_He_Torr, T_cell)
        ans = sigma_D1_radiative / \
            (2*pi*tau_D1_ns*n_He_amagat*21.5*((T_j/294)**0.5))
        return ans

    def calc_P_j_peak(self, j):
        I_max = self.constants["I_max"]
        S_j = self.calc_S_j(j)
        r_j = self.calc_r_j(j)
        w_P = self.constants["w_P"]
        inside_exponential = -2*((r_j/w_P)**2)
        P_j_peak = S_j*I_max*math.exp(inside_exponential)
        return P_j_peak

    def calc_P_j_lambda_wo_P_j_peak(self, lambda_nm):
        lambda_0_nm = self.constants["lambda_0_nm"]
        delta_lambda_D2_FWHM_nm = self.constants["delta_lambda_D2_FWHM_nm"]
        sigma_of_normal_distr = delta_lambda_D2_FWHM_nm / \
            (2*((2*math.log(2))**0.5))
        inside_exponential = ((lambda_nm-lambda_0_nm)**2) / \
            (2*sigma_of_normal_distr*sigma_of_normal_distr)
        pi = math.pi
        denominator = ((2*pi)**0.5)*sigma_of_normal_distr
        return (math.exp(-inside_exponential))/denominator

    def calc_P_L_j(self, j):
        P_Thermal = self.solve_parameters["P_Thermal"]
        P_P = self.constants["P_P"]
        P_L = P_P - P_Thermal
        r_j = self.calc_r_j(j)
        r_j_p_1 = self.calc_r_j(j+1)
        R = self.constants["R"]
        val = P_L*(math.exp(-2*r_j_p_1*r_j_p_1/(R*R)) -
                   math.exp(-2*r_j*r_j/(R*R)))
        self.P_L_j_list[j] = val
        return val

    def calc_n_0_j(self, T_j, j):
        T_w = self.constants["T_w"]
        val = self.calc_n_0_1(T_w)
        if j == 1:
            return val
        else:
            return val*(T_w/T_j)

    def calc_n_0_1(self, T_w):
        NA = self.constants["NA"]
        R_SI = self.constants["R_SI"]
        a = 133.322
        b = 8.22127
        c = -4006.048
        d = -0.00060194
        e = -0.196231
        ans = ((a*NA)/(R_SI*T_w)) * \
            (10**(b + c/T_w + d*T_w + e*math.log10(T_w)))
        return ans

    def calc_Omega_j(self, T_j, n_2_j, n_3_j):
        delta_E = self.constants["delta_E"]
        kB = self.constants["kB"]
        gamma_3_2 = self.calc_gamma_3_2(T_j)
        val = n_3_j - 2*n_2_j*math.exp(-delta_E/(kB*T_j))
        return gamma_3_2*val*delta_E

    def calc_Q_j(self, j, T_j, n_2_j, n_3_j):
        V_j = self.calc_V_j(j)
        Omega_j = self.calc_Omega_j(T_j, n_2_j, n_3_j)
        return V_j*Omega_j

    def calc_Phi_j(self, P_Thermal, j):
        val = P_Thermal
        for i in range(1, j):
            val -= self.Q_list[i]
        if not self.is_gas_flowing:
            return val
        for i in range(1, j):
            val -= self.F_list[i]
        return val

    def calc_K_He_at_T_j(self, T_j):
        a = 0.05516
        b = 3.2540*(10**-4)
        c = -2.2723*(10**-8)
        return a + b*T_j + c*(T_j**2)

    def calc_K_Ethane_at_T_j(self, T_j):
        a = -0.01936
        b = 1.2547*(10**-4)
        c = 3.8298*(10**-8)
        return a + b*T_j + c*(T_j**2)

    def calc_K_Overall_at_T_j(self, T_j):
        K_He_at_T_j = self.calc_K_He_at_T_j(T_j)
        K_Ethane_at_T_j = self.calc_K_Ethane_at_T_j(T_j)
        P_He_Torr = self.constants["P_He_Torr"]
        P_Ethane_Torr = self.constants["P_Ethane_Torr"]
        return (P_He_Torr*K_He_at_T_j + P_Ethane_Torr*K_Ethane_at_T_j)/(P_He_Torr + P_Ethane_Torr)

    def calc_Cp_He_at_T_j(self, T_j):
        coefficients = [20.78603, 4.850638*1e-10, -1.582916*1e-10]
        ans = 0
        T_power = 1
        for coefficient in coefficients:
            ans += coefficient*T_power
            T_power *= T_j
        return ans

    def calc_Cp_Ethane_at_T_j(self, T_j):
        # I went to this website to get this equation
        # http://polynomialregression.drque.net/online.php
        # Ethane Molar Heat Capacity file has the data
        # That data has been taken from https://webbook.nist.gov/cgi/cbook.cgi?ID=C74840&Mask=1EFF
        # I just combined the 2nd data, and parts of the first data, to get the final data

        # a = 63.13495041687804203094, b = -0.48062093010468977040, c = 0.00336396965233084690, d = -0.00001072783162400856
        # e = 0.00000002105388296370, f = -0.00000000002477574409, g = 0.00000000000001574575, h = -0.00000000000000000412

        coefficients = [63.13495041687804, -0.48062093010468976, 0.003363969652330847, -0.00001072783162400856,
                        2.10538829637e-8, -2.477574409e-11, 1.574575e-14, -4.12e-18]
        ans = 0
        T_power = 1
        for coefficient in coefficients:
            ans += coefficient*T_power
            T_power *= T_j
        return ans

    def calc_Cp_Overall_at_T_j(self, T_j):
        Cp_He_at_T_j = self.calc_Cp_He_at_T_j(T_j)
        Cp_Ethane_at_T_j = self.calc_Cp_Ethane_at_T_j(T_j)
        P_He_Torr = self.constants["P_He_Torr"]
        P_Ethane_Torr = self.constants["P_Ethane_Torr"]
        return (P_He_Torr*Cp_He_at_T_j + P_Ethane_Torr*Cp_Ethane_at_T_j)/(P_He_Torr + P_Ethane_Torr)

    def calc_F_j(self, T_j, j):
        N = self.constants["N"]
        S_j = self.calc_S_j(j)
        U = self.constants["U"]
        NA = self.constants["NA"]
        T_w = self.constants["T_w"]
        R_SI = self.constants["R_SI"]
        torr_to_bar = 0.00133322

        n_0_j = self.calc_n_0_j(T_j, j)
        integration_part = 0
        dT = 0.1  # K

        n_T_j = n_0_j + (self.constants["P_He_Torr"]*torr_to_bar*1e5*NA/(R_SI*T_w)
                         + self.constants["P_Ethane_Torr"]*torr_to_bar*1e5*NA/(R_SI*T_w))/N

        starting_T = T_w
        ending_T = T_j
        current_T = starting_T

        while current_T <= ending_T:
            Cp = self.calc_Cp_Overall_at_T_j(current_T)
            integration_part += Cp*dT
            current_T += dT
        ans = ((S_j*U*n_T_j)/NA)*integration_part
        return ans

    def calc_Gamma_P_j_by_integration_numpy(self, n_1_j, n_3_j, T_j, j):
        # This is what Saksham suggested i.e. for all the shells whose radii is greater than w_P, Gamma_P ~= 0, and very small values were causing numerical issues
        if self.calc_r_j(j) > self.constants["w_P"]:
            return 0

        R_P = self.constants["R_P"]
        h = self.constants["h"]
        c = self.constants["c"]
        eta_del = self.constants["eta_del"]
        l = self.constants["l"]

        # lambda_nm_times_P_j_l_wo_P_j_p can now be retrieved from self.constants, because this is just a thing,
        # which is dependent on the values of lambda_nm, which in turn depends only on starting_lambda_nm,
        # ending_lambda_nm and d_lambda_nm

        # g can now be retived from self.constants (g does not change for multiple iterations,
        # it remains the same in all of code before this point)
        lambda_list = self.lambda_list
        lambda_nm_times_P_j_l_wo_P_j_p = self.constants["lambda_nm_times_P_j_l_wo_P_j_p"]
        g = copy.deepcopy(self.constants["g"])

        P_j_peak = self.calc_P_j_peak(j)
        self.P_j_list[j] = P_j_peak
        V_j = self.calc_V_j(j)
        K_1 = eta_del*P_j_peak/(V_j*h*c)
        sigma_D2_He_broadened = self.calc_sigma_D2_He_broadened(T_j)

        # g stands for the stuff exp(-((n_1_j - 0.5*n_3_j)*sigma_D2_T_j_lambda*l))
        # where, sigma_D2_T_j_lambda = sigma_D2_He_broadened/(1 + ((lambda_nm - lambda_D2_nm)/K_5)**2)
        g = g*((n_1_j - 0.5*n_3_j)*sigma_D2_He_broadened*l)
        g_temp = np.exp(-g)
        g = np.multiply(1 - g_temp, 1 + R_P*g_temp)

        val = np.multiply(lambda_nm_times_P_j_l_wo_P_j_p, g)
        integration = simps(val, lambda_list)
        ans = K_1*integration
        return ans

    def calc_inside_exponential_Gamma_P_j(self, n_1_j, n_3_j, lambda_nm, lambda_D2_nm, K_5, T_j):
        l = self.constants["l"]
        sigma_D2_He_broadened = self.calc_sigma_D2_He_broadened(T_j)
        sigma_D2_T_j_lambda = sigma_D2_He_broadened / \
            (1 + ((lambda_nm - lambda_D2_nm)/K_5)**2)
        ans = (n_1_j - 0.5*n_3_j)*sigma_D2_T_j_lambda*l
        return ans

    def calc_Gamma_L_j_by_integration(self, n_1_j, n_2_j, T_j, j):
        # I will write the gaussian form of Gamma_L_j as suggested by Saksham, and then, I will solve the 3 differential equations together.
        # i.e., solving for n_1_j, n_2_j, n_3_j, in a simulataneous differential equation.
        TT = self.constants["TT"]
        R_oc = self.constants["R_oc"]
        l = self.constants["l"]
        h = self.constants["h"]
        nu_L_GHz = self.constants["nu_L_GHz"]
        # nu_L_GHz comes out to be 335113.411581
        sigma_D1_He_broadened = self.calc_sigma_D1_He_broadened(T_j)
        inside_exponential = (n_2_j - n_1_j)*sigma_D1_He_broadened*l
        P_L_j = self.calc_P_L_j(j)
        V_j = self.calc_V_j(j)
        val = (P_L_j*R_oc)/(V_j*h*nu_L_GHz*1e9*(1-R_oc))
        val = val*(math.exp(inside_exponential) - 1) * \
            (1 + TT*TT*math.exp(inside_exponential))
        # print("V_j =", V_j, "P_L_j =", P_L_j, "sigma_D1_He_broadened = ", sigma_D1_He_broadened, "Gamma_L_j = ", val)
        return val

    def calc_n_1_j_n_2_j_n_3_j(self, T_j, j):
        # https://stackoverflow.com/questions/51808922/how-to-solve-a-system-of-differential-equations-using-scipy-odeint
        t_span = 1e-10*(np.linspace(0, 15, 101))
        n_0_j = self.calc_n_0_j(T_j, j)
        self.n_0_list[j] = n_0_j
        u0 = (n_0_j, 0, 0)
        solution = odeint(func=self.du_dt, y0=u0, t=t_span, args=(T_j, j))

        # plotting
        number_of_iterations = self.number_of_iterations
        file_names = []
        for i in range(1, 4):
            file_names.append(str(number_of_iterations) +
                              "n_{}_vs_t.jpg".format(i))
            file_name = "iter_{}_n_{}_vs_t".format(number_of_iterations, i)
            # activating figure n_{i}_vs_t
            plt.figure(file_name)
            plt.plot(t_span, solution[:, i-1], label='n_{}_{}'.format(i, j))
            plt.title(file_name)
            plt.xlabel("time")
            plt.ylabel("number_density_(per_m_3)")
            plt.ylim(bottom=0)
            plt.ylim(top=2.75e19)
            plt.legend(loc="right", ncol=5,
                       labelspacing=0.05, prop={"size": 6})
            if j == self.constants["N"]:
                plt.savefig(self.folder_name + file_name + ".jpg")
        return solution

    def du_dt(self, u, t, T_j, j):
        n_1_j, n_2_j, n_3_j = u
        tau_D1_ns = self.constants["tau_D1_ns"]
        tau_D2_ns = self.constants["tau_D2_ns"]
        delta_E = self.constants["delta_E"]
        kB = self.constants["kB"]
        gamma_3_2 = self.calc_gamma_3_2(T_j)

        Gamma_P_j = self.calc_Gamma_P_j_by_integration_numpy(
            n_1_j, n_3_j, T_j, j)
        Gamma_L_j = self.calc_Gamma_L_j_by_integration(n_1_j, n_2_j, T_j, j)

        d_n_1_j_d_t = -Gamma_P_j + Gamma_L_j + n_2_j / \
            (tau_D1_ns*1e-9) + n_3_j/(tau_D2_ns*1e-9)
        d_n_2_j_d_t = -Gamma_L_j + gamma_3_2 * \
            (n_3_j - 2*n_2_j*math.exp(-delta_E/(kB*T_j))) - n_2_j/(tau_D1_ns*1e-9)
        d_n_3_j_d_t = Gamma_P_j - gamma_3_2 * \
            (n_3_j - 2*n_2_j*math.exp(-delta_E/(kB*T_j))) - n_3_j/(tau_D2_ns*1e-9)

        return (d_n_1_j_d_t, d_n_2_j_d_t, d_n_3_j_d_t)

    def solve(self, P_Thermal):
        # print("P_Thermal = ", P_Thermal)
        self.solve_parameters["P_Thermal"] = P_Thermal
        self.number_of_iterations += 1
        print("Iteration number", self.number_of_iterations)
        N = self.constants["N"]
        T_w = self.constants["T_w"]
        j = 1
        T_j = T_w
        self.T_list[j] = T_j
        # check the condition because for N-1th shell, we won't get T_N
        # while j != N:
        for j in tqdm(range(1, N)):
            # print("number_of_iterations = {}, j = {}".format(self.number_of_iterations, j))
            # Default values as mentioned in the research paper [12] Beach
            # n_1_j = 1.13e19/N #per m3 pass
            # n_2_j = 1.19e19/N #per m3 pass
            # n_3_j = 3.77e18/N #per m3 pass

            n_1_j, n_2_j, n_3_j = self.calc_n_1_j_n_2_j_n_3_j(T_j, j)[-1]

            self.n_1_list[j] = n_1_j
            self.n_2_list[j] = n_2_j
            self.n_3_list[j] = n_3_j

            r_j = self.calc_r_j(j)
            r_j_p_1 = self.calc_r_j(j+1)
            A_j = self.calc_A_j(j)
            V_j = self.calc_V_j(j)
            self.r_list[j] = r_j
            self.r_list[j+1] = r_j_p_1

            Omega_j = self.calc_Omega_j(T_j, n_2_j, n_3_j)
            self.Omega_list[j] = Omega_j

            Q_j = V_j*Omega_j
            self.Q_list[j] = Q_j

            if self.is_gas_flowing:
                F_j = self.calc_F_j(T_j, j)
                self.F_list[j] = F_j

            K_Overall_at_T_j = self.calc_K_Overall_at_T_j(T_j)
            Phi_j = self.calc_Phi_j(P_Thermal, j)

            C_j_1 = Omega_j*r_j*r_j / \
                (2*K_Overall_at_T_j) - Phi_j*r_j/(K_Overall_at_T_j*A_j)
            C_j_0 = T_j - C_j_1*math.log(r_j) + \
                Omega_j*r_j*r_j/(4*K_Overall_at_T_j)

            T_j_p_1 = C_j_1*math.log(r_j_p_1) - Omega_j * \
                r_j_p_1*r_j_p_1/(4*K_Overall_at_T_j) + C_j_0
            self.T_list[j+1] = T_j_p_1
            # j = j + 1
            T_j = T_j_p_1

        r_N = self.constants["R"]/N
        T_N = self.T_list[N]

        n_1_N, n_2_N, n_3_N = self.calc_n_1_j_n_2_j_n_3_j(T_j, j)[-1]
        self.n_1_list[N] = n_1_N
        self.n_2_list[N] = n_2_N
        self.n_3_list[N] = n_3_N

        V_N = self.calc_V_j(N)
        Omega_N = self.calc_Omega_j(T_N, n_2_N, n_3_N)
        self.Omega_list[N] = Omega_N
        Q_N = V_N*Omega_N
        self.Q_list[N] = Q_N

        if self.is_gas_flowing:
            F_N = self.calc_F_j(T_N, N)
            self.F_list[N] = F_N

        new_P_Thermal = sum(self.Q_list) - sum(self.F_list)
        K_Overall_at_T_N = self.calc_K_Overall_at_T_j(T_N)
        C_N_0 = T_N + Omega_N*r_N*r_N/(4*K_Overall_at_T_N)
        T_N_p_1 = C_N_0
        self.T_list[N+1] = T_N_p_1

        # plotting Temperature vs radial position for every iteration
        plot_parameters = {"file_name": "Temp_vs_radial_position_iteration_{}".format(self.number_of_iterations),
                           "x": self.r_list[1:], "y": self.T_list[1:], "xlabel_value": "radial_position_(m)",
                           "ylabel_value": "Temperature (K)", "legend_value": None,
                           "save_folder": self.folder_name, "extension": ".jpg"}
        # self.plot_graphs(parameters=plot_parameters)

        # plotting number density vs shell number for every iteration
        file_name = "number_density_vs_shell_number_iteration_{}".format(
            self.number_of_iterations)
        # self.plot_number_density(file_name=file_name)

        # plotting Heat Generated vs j (shell number) for every iteration
        plot_parameters = {"file_name": "heat_generated{}".format(self.number_of_iterations),
                           "x": list(range(len(self.Q_list)))[::-1][1:], "y": self.Q_list[1:], "xlabel_value": "shell_number",
                           "ylabel_value": "Heat Generated", "legend_value": None,
                           "save_folder": self.folder_name, "extension": ".jpg"}
        # self.plot_graphs(parameters=plot_parameters)

        # plotting P_j_peak vs j (shell number) for every iteration
        plot_parameters = {"file_name": "P_j_peak_vs_j_iteration_{}".format(self.number_of_iterations),
                           "x": list(range(len(self.P_j_list)))[:][1:], "y": self.P_j_list[1:], "xlabel_value": "shell_number",
                           "ylabel_value": "P_j_peak", "legend_value": None,
                           "save_folder": self.folder_name, "extension": ".jpg"}
        # self.plot_graphs(parameters=plot_parameters)

        # plotting P_j_lambda_wo_P_j_peak_iteration vs j (shell number) for every iteration
        plot_parameters = {"file_name": "P_j_lambda_wo_P_j_peak_iteration_{}".format(self.number_of_iterations),
                           "x": self.lambda_list, "y": self.P_j_lambda_wo_P_j_peak_list, "xlabel_value": "lambda_nm",
                           "ylabel_value": "P_j_lambda_wo_P_j_peak", "legend_value": None,
                           "save_folder": self.folder_name, "extension": ".jpg"}
        # self.plot_graphs(parameters=plot_parameters)

        # self.steady_state_validation()
        if abs(new_P_Thermal - P_Thermal)/P_Thermal > self.max_allowable_error and self.number_of_iterations < self.max_number_of_iterations:
            return self.solve(new_P_Thermal)
        else:
            if self.number_of_iterations >= self.max_number_of_iterations:
                print("The values did not converge after {} iterations, returning the current state of the model".format(
                    self.max_number_of_iterations))

            # Final Number Density Plot
            file_name = "Final Number Density vs shell number"
            self.plot_number_density(file_name=file_name)

            # Final Temperature Plot
            plot_parameters = {"file_name": "Final Temperature (K) vs Radial Position (mm)",
                               "x": self.r_list[1:], "y": self.T_list[1:], "xlabel_value": "Radial Position (m)",
                               "ylabel_value": "Temperature (K)", "legend_value": None,
                               "save_folder": self.folder_name, "extension": ".jpg"}
            self.plot_graphs(parameters=plot_parameters)

            Final_P_Thermal = new_P_Thermal
            Final_P_Lasing = self.constants["P_P"]-new_P_Thermal
            save_txt = "P_P,{}\nPThermal,{}\nPLasing,{}\n".format(
                self.constants["P_P"], Final_P_Thermal, Final_P_Lasing)
            outfile = open(self.folder_name + "Results.csv", "w")
            outfile.write(save_txt)
            outfile.close()
            # print("Final P_Thermal = {}".format(new_P_Thermal))
            # print("Final P_Lasing = {}".format(self.constants["P_P"]-new_P_Thermal))

            d = {
                "Radial Position": self.r_list[1:][::-1], "Temperature": self.T_list[1:][::-1]}
            df = pd.DataFrame(d)
            data_file_name = 'Temp_vs_radial_position.csv'
            data_file_location = self.folder_name + data_file_name
            df.to_csv(data_file_location, index=False)

            self.Plot3D(data_file_location)
            return (self.T_list[:], self.Q_list[:])

    def steady_state_validation(self):
        calculated_n_0_array = np.array(
            self.n_1_list) + np.array(self.n_2_list) + np.array(self.n_3_list)
        actual_n_0_array = np.array(self.n_0_list)
        error_in_n_0 = calculated_n_0_array - actual_n_0_array
        # print("calculated_n_0_array = ", calculated_n_0_array)
        # print("actual_n_0_array = ", actual_n_0_array)
        # print("error_in_n_0 = ", error_in_n_0)

        l = self.constants["l"]
        TT = self.constants["TT"]
        R_oc = self.constants["R_oc"]
        vector_func = np.vectorize(self.calc_sigma_D1_He_broadened)
        vector_sigma_D1_He_broadened = vector_func(self.T_list[1:-1])
        val = 2*(np.array(self.n_2_list[1:]) - np.array(self.n_1_list[1:]))
        val = np.multiply(val, vector_sigma_D1_He_broadened)
        val = l*val
        val = np.exp(val)
        val = TT*TT*R_oc*val

    def plot_number_density(self, file_name):
        plt.figure(file_name)
        x = 1e6*np.array(self.r_list[2:])
        plt.plot(x, self.n_0_list[1:], label="n_0")
        plt.plot(x, self.n_1_list[1:], label="n_1")
        plt.plot(x, self.n_2_list[1:], label="n_2")
        plt.plot(x, self.n_3_list[1:], label="n_3")
        plt.xscale('log')
        plt.xticks([10, 100, 1000, 10000], rotation=0)
        plt.title(file_name)
        plt.xlabel("Radial_position_(um)")
        plt.ylabel("Number_density_(per_m_3)")
        plt.legend(loc="best")
        plt.savefig(self.folder_name + file_name + ".jpg")

    def plot_graphs(self, parameters={}):
        file_name = parameters["file_name"]
        x = parameters["x"]
        y = parameters["y"]
        xlabel_value = parameters["xlabel_value"]
        ylabel_value = parameters["ylabel_value"]
        plt.figure(file_name)
        plt.plot(x, y)
        plt.title(file_name)
        plt.xlabel(xlabel_value)
        plt.ylabel(ylabel_value)

        if parameters.get("legend_value"):
            plt.legend(loc=parameters["legend_value"])
            plt.legend()

        if parameters.get("xscale_value"):
            xscale_value = parameters["xscale_value"]
            plt.xscale(xscale_value)

        save_folder = parameters["save_folder"]
        image_extension = ".jpg"

        if parameters.get("image_extension"):
            image_extension = parameters["image_extension"]

        final_file_address = save_folder + file_name + image_extension
        plt.savefig(final_file_address)

    def Plot3D(self, data_file):
        df = pd.read_csv(data_file)
        r_list = df["Radial Position"]
        T_list = df["Temperature"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create the mesh in polar coordinates and compute corresponding Z.
        N = self.constants["N"]
        # Converting radii from um to mm
        r = 1000*r_list
        p = np.linspace(0, 2*np.pi, N+1)
        R, P = np.meshgrid(r, p)
        # Z = ((R**2 - 1)**2)
        T, P = np.meshgrid(T_list, p)
        Z = T
        # Express the mesh in the cartesian system.
        X, Y = R*np.cos(P), R*np.sin(P)

        # Plot the surface.
        cmap_name = 'inferno'

        # cmap_names = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
        # 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        # 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        # 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        cmap = plt.get_cmap(cmap_name)
        ax.plot_surface(X, Y, Z, cmap=cmap)

        ax.set_xlabel("Radial Position (mm)")
        ax.set_ylabel("Radial Position (mm)")
        ax.set_zlabel("Temperature (K)")

        final_file_address = self.folder_name + "Final 3D Plot.jpg"
        plt.savefig(final_file_address)

    def clearFolderContents(self, directory):
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        except OSError:
            print("Error in deleting the contents of the directory: " + directory)
