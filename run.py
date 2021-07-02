from btp import DPAL
import sys
constants = {"P_He_Torr": 478.8, "P_Ethane_Torr": 100, "eta_del": 0.90, "L": 0.025, "l": 0.025, "R": 0.0075, "T_w": 383,
             "sigma_D1_radiative": 1.53*(10**-13), "sigma_D2_radiative": 2.31*(10**-13), "R_oc": 0.30, "TT": 0.99, "PumpFWHM_in_GHz": 30,
             "Pump_center_wavelength_nm": 852.3, "D1_transition_wavelength_nm": 894.6, "tau_D1_ns": 34.9, "tau_D2_ns": 30.5,
             # In the Beach paper, Table 3, for D2, collisionally broadened FWHM is 12.9 GHz, which is 23239725.4264 nm
             "delta_lambda_D2_FWHM_nm": 0.03125,
             # I was using 42.94 as the delta_lambda_FWHM_nm value, I can't remember why, also Saksham had once suggested to use 0.06 nm as the value
             "N": 50, "kB": 1.38*(10**-23), "mCs": 132.90545*(1.66054*(10**-27)),
             "mHe": 4.002602*(1.66054*(10**-27)), "mEthane": 30.07*(1.66054*(10**-27)), "delta_E": 1.98644582*(10**-25)*55400,
             "NA": 6.023*(10**23), "R_SI": 8.314, "lambda_D1_nm": 894.6, "lambda_D2_nm": 852.3, "h": 6.634*1e-34, "nu_L_GHz": 19, "sigma_3_2_Ethane": 5.2e-19,
             "sigma_3_2_He": 2.25e-23, "T_cell": 383, "c": 299792458, "P_P": 10, "w_P": 500*1e-6, "lambda_0_nm": 852.3, "R_P": 0,
             "starting_lambda_nm": 851.5, "ending_lambda_nm": 852.7, "d_lambda_nm": 0.1, "is_gas_flowing": False, "U": 0}

no_of_arguments = len(sys.argv)
if no_of_arguments == 1:
    run_on_default_values = True

    valid_input = False
    while not valid_input:
        ans = input("Do you want to run on default values? Enter y or n:")
        if ans == "y" or ans == "Y":
            run_on_default_values = True
            P_Thermal = 5
            valid_input = True
        elif ans == "n" or ans == "N":
            run_on_default_values = False
            P_Thermal = 5
            valid_input = True

    if not run_on_default_values:
        w_P = int(input("Enter the value of w_P, in (um):"))
        T_w = int(input("Enter the value of T_w, in (K):"))
        P_P = int(input("Enter the value of P_P, in (W):"))
        P_Thermal = float(
            input("Enter the initial guess for P_Thermal, in (W):"))
        is_gas_flowing = input("Is the gas flowing? Enter y or n:")
        if is_gas_flowing == "y":
            is_gas_flowing = True
            U = int(input("Enter the value of flow speed U, in m/s:"))
        else:
            is_gas_flowing = False
            U = 0
        constants["w_P"] = w_P*1e-6
        constants["T_w"] = T_w
        constants["P_P"] = P_P
        constants["is_gas_flowing"] = is_gas_flowing
        constants["U"] = U

elif no_of_arguments == 2:
    infile = open(sys.argv[1])
    options = list(infile)
    options = list(map(str.strip, options))

    run_on_default_values = True

    valid_input = False
    ans = options[0]
    if ans == "y" or ans == "Y":
        run_on_default_values = True
        P_Thermal = 5
        valid_input = True
    elif ans == "n" or ans == "N":
        run_on_default_values = False
        valid_input = True

    if not run_on_default_values:
        w_P = int(options[1])
        T_w = int(options[2])
        P_P = int(options[3])
        P_Thermal = float(options[4])
        is_gas_flowing = options[5]
        if is_gas_flowing == "y":
            is_gas_flowing = True
            U = int(options[6])
        else:
            is_gas_flowing = False
            U = 0
        constants["w_P"] = w_P*1e-6
        constants["T_w"] = T_w
        constants["P_P"] = P_P
        constants["is_gas_flowing"] = is_gas_flowing
        constants["U"] = U
dpal = DPAL(constants=constants, max_allowable_error=0.01,
            max_number_of_iterations=10)

T_list, Q_list = dpal.solve(P_Thermal=P_Thermal)
print(f"Result Folder is {dpal.folder_name}")
