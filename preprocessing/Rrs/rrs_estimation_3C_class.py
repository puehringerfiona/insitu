import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lmfit as lm
import time
from pathlib import Path


class rrs_model_3C:
    def __init__(self, downsampling=False, wl_range=False, spectra_path=None):
        spectra_path = Path(spectra_path or Path(__file__).resolve().parent / "spectra")
        a_ph = pd.read_csv(spectra_path / 'phyto.A', encoding="ISO-8859-1", skiprows=11, sep='\t', index_col=0).iloc[:, 0]
        a_w = pd.read_csv(spectra_path / 'water.A', encoding="ISO-8859-1", skiprows=10, sep='\t', index_col=0).iloc[:, 0]
        daw_dT = pd.read_csv(spectra_path / 'daWdT.txt', encoding="ISO-8859-1", skiprows=10, sep='\t', index_col=0).iloc[:, 0]
        astar_y = pd.read_csv(spectra_path / 'Y.A', encoding="ISO-8859-1", skiprows=11, delimiter=' ', index_col=0).loc[350:].iloc[:, 0]

        d = pd.DataFrame(index=a_ph.index)
        d.index.name = 'Wavelength, [nm]'
        d['astar_ph'] = a_ph
        d['astar_y'] = astar_y
        d['a_w'] = a_w
        d['daw_dT'] = daw_dT

        if downsampling:
            self.spectra = d.rolling(window=downsampling, center=True).mean()
        else:
            self.spectra = d

        if wl_range:
            self.spectra = self.spectra.loc[wl_range[0]:wl_range[1]].dropna()

        self.wl = np.array(self.spectra.index)

    def forward(self, params, wl, Ls, Lu, Ed):
        # Unpack and convert all parameters to torch tensors
        p = {k: torch.tensor(v, dtype=torch.float32) for k, v in params.valuesdict().items()}

        wl = torch.tensor(wl, dtype=torch.float32)
        a_w = torch.tensor(self.spectra['a_w'].values, dtype=torch.float32)
        daw_dT = torch.tensor(self.spectra['daw_dT'].values, dtype=torch.float32)
        astar_ph = torch.tensor(self.spectra['astar_ph'].values, dtype=torch.float32)
        astar_y = torch.tensor(self.spectra['astar_y'].values, dtype=torch.float32)
        Ls_Ed = torch.tensor(Ls / Ed, dtype=torch.float32)

        # Angles
        theta_sun = p['theta_sun'] * np.pi / 180.
        theta_view = p['theta_view'] * np.pi / 180.
        wl_a = 550
        wl_ref_y = 440
        wl_ref_mie = 500
        wl_ref_water = 500

        # Atmospheric components
        alpha = p['alpha']
        z1 = torch.where(
            alpha < 0,
            torch.tensor(0.82, dtype=torch.float32),
            torch.where(
                alpha > 1.2,
                torch.tensor(0.65, dtype=torch.float32),
                -0.1417 * alpha + 0.82
            )
        )

        theta_sun_mean = z1

        B3 = torch.log(1 - theta_sun_mean)
        B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
        B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
        Fa = 1 - 0.5 * torch.exp((B1 + B2 * torch.cos(theta_sun)) * torch.cos(theta_sun))

        omega_a = (-0.0032 * p['am'] + 0.972) * torch.exp(3.06e-4 * p['rh'])
        tau_a = p['beta'] * (wl / wl_a) ** (-p['alpha'])

        M = 1 / (torch.cos(theta_sun) + 0.50572 * (90 + 6.07995 - p['theta_sun']) ** (-1.6364))
        M_ = M * p['pressure'] / 1013.25

        Tr = torch.exp(- M_ / (115.6406 * (wl / 1000) ** 4 - 1.335 * (wl / 1000) ** 2))
        Tas = torch.exp(- omega_a * tau_a * M)
        Edd = Tr * Tas
        Edsr = 0.5 * (1 - Tr ** 0.95)
        Edsa = Tr ** 1.5 * (1 - Tas) * Fa
        Ed = Edd + Edsr + Edsa

        Eds_Ed = (Edsr + Edsa) / Ed
        Edd_Ed = Edd / Ed

        # Absorption
        a_ph = p['C_chl'] * astar_ph
        a_y = torch.where(
            p['S_y'] == -1,
            p['C_y'] * astar_y,
            p['C_y'] * torch.exp(-p['S_y'] * (wl - wl_ref_y))
        )

        a_w_corr = a_w + (p['T_w'] - 20.) * daw_dT
        a = a_w_corr + a_ph + a_y

        # Backscattering
        bb_sm = p['C_sm'] * 0.0086 + p['C_mie'] * 0.0042 * (wl / wl_ref_mie) ** p['n_mie']
        bb_water = torch.where(
            p['n_w'] == 1.34,
            torch.tensor(0.00144, dtype=torch.float32),
            torch.tensor(0.00111, dtype=torch.float32)
        ) * (wl / wl_ref_water) ** (-4.32)

        bb = bb_water + bb_sm
        omega_b = bb / (bb + a)

        # Angles under water
        theta_sun_ss = torch.arcsin(torch.sin(theta_sun) / p['n_w'])
        theta_view_ss = torch.arcsin(torch.sin(theta_view) / p['n_w'])

        f = 0.1034 * (1 + 3.3586 * omega_b - 6.5358 * omega_b ** 2 + 4.6638 * omega_b ** 3) * (
                    1 + 2.4121 / torch.cos(theta_sun_ss))
        frs = 0.0512 * (1 + 4.6659 * omega_b - 7.8387 * omega_b ** 2 + 5.4571 * omega_b ** 3) * \
              (1 + 0.1098 / torch.cos(theta_sun_ss)) * (1 + 0.4021 / torch.cos(theta_view_ss))

        R0minus = f * omega_b
        Rrs0minus = frs * omega_b

        Rrs_refl = p['rho_s'] * Ls_Ed + p['rho_dd'] * Edd_Ed / np.pi + p['rho_ds'] * Eds_Ed / np.pi + p['delta']
        Rrs = 0.518 * Rrs0minus / (1 - 0.48 * R0minus)

        Lu_Ed = Rrs + Rrs_refl

        return Rrs, Rrs_refl, Lu_Ed

    def fit_LuEd(self, wl, Ls, Lu, Ed, params, weights, verbose=True):
        def min_funct(params):
            Rrs_modelled, Rrs_refl, Lu_Ed_modelled = self.forward(params, wl, Ls, Lu, Ed)
            Rrs_obs = torch.tensor(Lu / Ed) - Rrs_refl
            resid = torch.sum((Lu_Ed_modelled - Lu / Ed) ** 2 * torch.tensor(weights))
            return resid.item(), Rrs_modelled.detach().numpy(), Rrs_refl.detach().numpy(), Lu_Ed_modelled.detach().numpy(), Rrs_obs.detach().numpy()

        start_time = time.time()
        reg = lm.minimize(lambda x: min_funct(x)[0], params=params, method='lbfgsb',
                          options={'disp': verbose, 'gtol': 1e-16, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16,
                                   'maxls': 20, 'maxcor': 20})
        print("--- %s seconds ---" % (time.time() - start_time))

        resid, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs = min_funct(reg.params)
        reg.params.add('resid', resid, False, 0.0, 100, None)
        return reg, Rrs_modelled, Rrs_refl, Lu_Ed_modelled, Rrs_obs
