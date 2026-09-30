import numpy as np
import pandas as pd
import lmfit as lm
import rrs_estimation_3C_class


def my_3C_v01(data_Ed, data_Lu, data_Ls, output_path, sunzenith):
    """
    Run the 3C model on hyperspectral water data (Ed, Lu, Ls) and return Rrs and Lu/Ed data.

    Parameters:
    - data_Ed, data_Lu, data_Ls: DataFrames of shape (wavelengths x timestamps)
    - output_path: path to save files (not used here, just kept for compatibility)
    - sunzenith: scalar or list/array of sun zenith angles per timestamp

    Returns:
    - Rrs_output, Lu_Ed_obs, Lu_Ed_model: DataFrames of estimated and modeled values
    """

    wl = data_Ed.index.values.astype(int)
    n_samples = data_Ed.shape[1]

    # Ensure sunzenith is iterable
    if np.isscalar(sunzenith):
        sunzenith = [sunzenith] * n_samples
    else:
        sunzenith = list(sunzenith)

    # Initialize output dataframes
    Rrs_output = pd.DataFrame(index=wl, columns=data_Ed.columns)
    Lu_Ed_obs = pd.DataFrame(index=wl, columns=data_Ed.columns)
    Lu_Ed_model = pd.DataFrame(index=wl, columns=data_Ed.columns)

    for jj in range(n_samples):
        Ed = data_Ed.iloc[:, jj].values
        Lu = data_Lu.iloc[:, jj].values
        Ls = data_Ls.iloc[:, jj].values

        print(f"Processing sample {jj + 1}/{n_samples}...")

        params = lm.Parameters()
        params.add_many(
            ('C_chl', 5, True, 0.01, 100, None),
            ('C_mie', 0, False, 0, 100, None),
            ('n_mie', -1, False, -2, 2, None),
            ('C_sm', 1, True, 0.01, 100, None),
            ('C_y', 0.5, True, 0.01, 5, None),
            ('S_y', 0.018, False, 0.01, 0.03, None),
            ('T_w', 20, False, 0, 35, None),
            ('theta_sun', sunzenith[jj], False, 0, 90, None),
            ('theta_view', 40, False, 0, 180, None),
            ('n_w', 1.34, False, 1.33, 1.34, None),
            ('rho_s', 0.0256, False, 0.0, 0.1, None),
            ('am', 1, False, 1, 10, None),
            ('rh', 60, False, 0, 100, None),
            ('pressure', 1013.25, False, 800, 1100, None),
            ('delta', 0.00, False, 0, 1, None),
            ('rho_dd', 0.0, True, 0, 0.1, None),
            ('rho_ds', 0.01, True, 0.0, 0.1, None),
            ('alpha', 1.0, True, 0, 3, None),
            ('beta', 0.05, True, 0.0, 10, None),
        )

        weights = pd.Series(1.0, index=wl)
        weights.loc[:500] = 5.0
        weights.loc[675:750] = 0.1
        weights.loc[760:770] = 0.1

        model = rrs_estimation_3C_class.rrs_model_3C(wl_range=(wl[0], wl[-1]))
        result, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs = model.fit_LuEd(
            wl, Ls, Lu, Ed, params, weights.values
        )

        Rrs_output.iloc[:, jj] = Rrs_obs
        Lu_Ed_obs.iloc[:, jj] = Lu / Ed
        Lu_Ed_model.iloc[:, jj] = Lu_Ed_modelled

    return Rrs_output, Lu_Ed_obs, Lu_Ed_model

#
# import numpy as np
# import pandas as pd
# import lmfit as lm
# import rrs_estimation_3C_class
#
# def my_3C_v01(data_Ed, data_Lu, data_Ls, output_path, sunzenith):
#     """
#     Run the 3C model on hyperspectral water data (Ed, Lu, Ls) and return Rrs and Lu/Ed data.
#
#     Parameters:
#     - data_Ed, data_Lu, data_Ls: DataFrames of shape (wavelengths x timestamps)
#     - output_path: path to save files (not used here, just kept for compatibility)
#     - sunzenith: scalar or list/array of sun zenith angles per timestamp
#
#     Returns:
#     - Rrs_output, Lu_Ed_obs, Lu_Ed_model: DataFrames of estimated and modeled values
#     """
#
#     wl = data_Ed.index.values.astype(int)
#     n_samples = data_Ed.shape[1]
#
#     # Ensure sunzenith is iterable
#     if np.isscalar(sunzenith):
#         sunzenith = [sunzenith] * n_samples
#     else:
#         sunzenith = list(sunzenith)
#
#     # Initialize output dataframes
#     Rrs_output = pd.DataFrame(index=wl, columns=data_Ed.columns)
#     Lu_Ed_obs = pd.DataFrame(index=wl, columns=data_Ed.columns)
#     Lu_Ed_model = pd.DataFrame(index=wl, columns=data_Ed.columns)
#
#     threshold = 1e-6  # threshold for very small values
#
#     for jj in range(n_samples):
#         Ed = data_Ed.iloc[:, jj].values
#         Lu = data_Lu.iloc[:, jj].values
#         Ls = data_Ls.iloc[:, jj].values
#         sample_col = data_Ed.columns[jj]
#
#         print(f"Processing sample {jj + 1}/{n_samples} ({sample_col})...")
#
#         # ----- PRE-CHECK FOR NaNs OR VERY SMALL VALUES -----
#         if np.any(np.isnan(Ed)) or np.any(np.isnan(Lu)) or np.any(np.isnan(Ls)):
#             print(f"  Skipping sample {sample_col} due to NaNs in input data")
#             continue
#         if np.any(Ed < threshold) or np.any(Lu < threshold) or np.any(Ls < threshold):
#             print(f"  Skipping sample {sample_col} due to extremely low values")
#             continue
#
#         # ----- SET UP PARAMETERS -----
#         params = lm.Parameters()
#         params.add_many(
#             ('C_chl', 5, True, 0.01, 100, None),
#             ('C_mie', 0, False, 0, 100, None),
#             ('n_mie', -1, False, -2, 2, None),
#             ('C_sm', 1, True, 0.01, 100, None),
#             ('C_y', 0.5, True, 0.01, 5, None),
#             ('S_y', 0.018, False, 0.01, 0.03, None),
#             ('T_w', 20, False, 0, 35, None),
#             ('theta_sun', sunzenith[jj], False, 0, 90, None),
#             ('theta_view', 40, False, 0, 180, None),
#             ('n_w', 1.34, False, 1.33, 1.34, None),
#             ('rho_s', 0.0256, False, 0.0, 0.1, None),
#             ('am', 1, False, 1, 10, None),
#             ('rh', 60, False, 0, 100, None),
#             ('pressure', 1013.25, False, 800, 1100, None),
#             ('delta', 0.00, False, 0, 1, None),
#             ('rho_dd', 0.0, True, 0, 0.1, None),
#             ('rho_ds', 0.01, True, 0.0, 0.1, None),
#             ('alpha', 1.0, True, 0, 3, None),
#             ('beta', 0.05, True, 0.0, 10, None),
#         )
#
#         weights = pd.Series(1.0, index=wl)
#         weights.loc[:500] = 5.0
#         weights.loc[675:750] = 0.1
#         weights.loc[760:770] = 0.1
#
#         # ----- TRY FITTING -----
#         try:
#             model = rrs_estimation_3C_class.rrs_model_3C(wl_range=(wl[0], wl[-1]))
#             result, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs = model.fit_LuEd(
#                 wl, Ls, Lu, Ed, params, weights.values
#             )
#
#             # Check outputs for NaNs
#             if np.any(np.isnan(Rrs_obs)) or np.any(np.isnan(Lu_Ed_modelled)):
#                 print(f"  Skipping sample {sample_col}: NaNs detected in model output")
#                 continue
#
#             Rrs_output.iloc[:, jj] = Rrs_obs
#             Lu_Ed_obs.iloc[:, jj] = Lu / Ed
#             Lu_Ed_model.iloc[:, jj] = Lu_Ed_modelled
#
#         except Exception as e:
#             print(f"  Error processing sample {jj + 1} ({sample_col}): {e}")
#             continue
#
#     return Rrs_output, Lu_Ed_obs, Lu_Ed_model
#
