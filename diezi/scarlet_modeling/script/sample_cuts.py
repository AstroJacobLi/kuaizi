"""
Make cuts based on color, SB, size, etc.
"""
import numpy as np
import itertools
from scipy.stats import binned_statistic, binned_statistic_dd
import matplotlib.pyplot as plt

default_cuts_dict = {'color_bound': [0., 1.2],
                     'color_width': 0.5,
                     'i_mag_limit': 22.5,
                     'min_re': 1.8,  # arcsec
                     'max_re': 12,  # arcsec
                     'min_cen_SB': 22.5,  # g-band
                     'min_eff_SB': 23.0,  # g-band
                     'max_eff_SB': 27.5,  # g-band
                     'max_ell': 0.65,
                     'max_Gini': 0.7,
                     'max_M20': -1.1,
                     'C_bound': [1.8, 3.5],
                     'max_A_outer': 0.8,
                     }

##### Make cuts based on color, SB, size, etc. #####


def make_cuts_vanilla(lsbg_cat, meas_cat, cuts_dict=None, include_morph_cut=True):
    if cuts_dict is not None:
        cuts_dict = cuts_dict
    else:
        cuts_dict = default_cuts_dict

    try:
        junk = (lsbg_cat['bad_votes'] > lsbg_cat['good_votes'])
        candy = (lsbg_cat['good_votes'] > lsbg_cat['bad_votes']) & (
            lsbg_cat['is_candy'] > lsbg_cat['is_galaxy'])
        gal = (~junk) & (~candy)
    except:
        pass

    g_mag = meas_cat['mag'].data[:, 0]
    r_mag = meas_cat['mag'].data[:, 1]
    i_mag = meas_cat['mag'].data[:, 2]

    gi_color = g_mag - i_mag
    gr_color = g_mag - r_mag

    color_bound = cuts_dict['color_bound']
    half_width = cuts_dict['color_width'] / 2

    # Color and magnitude cuts
    mask = (gi_color < color_bound[1]) & (gi_color > color_bound[0]) & (
        (gr_color) > 0.7 * (gi_color) - half_width) & (
        (gr_color) < 0.7 * (gi_color) + half_width)
    mask &= (i_mag < cuts_dict['i_mag_limit'])

    # Size cuts
    min_size = cuts_dict['min_re']
    max_size = cuts_dict['max_re']
    mask &= (meas_cat['rhalf_circularized'] >= min_size /
             0.168) & (meas_cat['rhalf_circularized'] <= max_size / 0.168)

    # SB cuts
    min_cen_SB = cuts_dict['min_cen_SB']
    min_eff_SB = cuts_dict['min_eff_SB']
    max_eff_SB = cuts_dict['max_eff_SB']
    mask &= (meas_cat['SB_0'][:, 0] > min_cen_SB)
    mask &= (meas_cat['SB_eff_avg'][:, 0] > min_eff_SB)
    mask &= (meas_cat['SB_eff_avg'][:, 0] < max_eff_SB)

    if include_morph_cut:
        print('Including morphological cuts')
        # Shape cuts
        mask &= (meas_cat['ell_sym'] < cuts_dict['max_ell'])

        mask &= (meas_cat['M20'] < cuts_dict['max_M20'])
        mask &= (meas_cat['Gini'] < cuts_dict['max_Gini'])
        # mask &= ~((meas_cat['M20'] < -1.6) & (
        #     meas_cat['Gini'] > meas_cat['M20'] * 0.136 + 0.788) & (meas_cat['Gini'] < meas_cat['M20'] * -0.136 + 0.33))
        mask &= (meas_cat['Gini'] < meas_cat['M20'] * -0.136 + 0.37)

        mask &= (meas_cat['C'] <= cuts_dict['C_bound'][1]) & (
            meas_cat['C'] >= cuts_dict['C_bound'][0])
        mask &= (meas_cat['A_outer'] <= cuts_dict['max_A_outer'])

    # mask &= (meas_cat['A'] < 0.7)
    # mask &= (seg_meas_cat['rhalf_circularized'] > 1.8 / 0.168) & (seg_meas_cat['rhalf_circularized'] < 12 / 0.168)
    try:
        print('Candy:', np.sum(mask & candy))
        print('Gal:', np.sum(mask & gal))
        print('Junk:', np.sum(mask & junk))
    except:
        pass

    return mask


def match_two_arrays(large_array, small_array):
    sorted_index = np.searchsorted(
        large_array[np.argsort(large_array)], small_array)
    yindex = np.take(np.argsort(large_array), sorted_index, mode="clip")
    return yindex


def make_cuts_spergel(cuts_cat, max_rhalf=15, min_rhalf=1.6, max_C=3.5):
    g_mag = cuts_cat['mag'].data[:, 0]
    r_mag = cuts_cat['mag'].data[:, 1]
    i_mag = cuts_cat['mag'].data[:, 2]

    gi_color = g_mag - i_mag
    gr_color = g_mag - r_mag

    color_bound = default_cuts_dict['color_bound']
    half_width = default_cuts_dict['color_width'] / 2

    # Color and magnitude cuts
    mask = (gi_color < color_bound[1]) & (gi_color > color_bound[0]) & (
        (gr_color) > 0.7 * (gi_color) - half_width) & (
        (gr_color) < 0.7 * (gi_color) + half_width)

    mask &= ((cuts_cat['rhalf_spergel'].data * 0.168) < max_rhalf)
    mask &= ((cuts_cat['rhalf_spergel'].data * 0.168) > min_rhalf)
    mask &= ((cuts_cat['rhalf_circularized'].data * 0.168) < max_rhalf)
    mask &= ~((cuts_cat['C'].data > max_C) & (
        cuts_cat['rhalf_spergel'] * 0.168 < 4))
    return mask


# def post_process_cat(input_cuts_cat):
#     cuts_cat = input_cuts_cat.copy()
#     import pickle
#     # Calculate SB and R_e errors
#     SB_g_meas = cuts_cat['SB_eff_avg'][:, 0].data
#     R_e = cuts_cat['rhalf_spergel'].data * 0.168

#     with open('./Catalog/completeness/Re_meas_err.pkl', 'rb') as f:
#         (f_med, f_std) = pickle.load(f)
#     # R_e += f_med(SB_g_meas)
#     R_e_std = f_std(SB_g_meas)
#     cuts_cat['rhalf_spergel'] = R_e
#     cuts_cat['rhalf_spergel_err'] = R_e_std

#     with open('./Catalog/completeness/SB_meas_err.pkl', 'rb') as f:
#         (f_med, f_std) = pickle.load(f)
#     SB_g = SB_g_meas  # + f_med(SB_g_meas)
#     SB_g_std = f_std(SB_g_meas)
#     cuts_cat['SB_eff_avg_g'] = SB_g
#     cuts_cat['SB_eff_avg_g_err'] = SB_g_std

#     # Physical sizes
#     # not consider peculiar motion
#     ang_diam_dist = cuts_cat['host_ang_diam_dist'].data
#     R_e_phys = R_e / 206265 * ang_diam_dist * 1000  # in kpc
#     R_e_phys_std = R_e_std / 206265 * ang_diam_dist * 1000  # in kpc
#     cuts_cat['rhalf_phys'] = R_e_phys
#     cuts_cat['rhalf_phys_err'] = R_e_phys_std

#     # Absolute magnitudes
#     cuts_cat['abs_mag'] = cuts_cat['mag'] - 25 - 5 * \
#         np.log10(ang_diam_dist *
#                  (1 + cuts_cat['host_z'].data)**2)[:, np.newaxis]  # griz

#     g_mag = cuts_cat['mag'][:, 0].data
#     g_abs = cuts_cat['abs_mag'][:, 0].data
#     # with open('./Catalog/completeness/mag_g_meas_err.pkl', 'rb') as f:
#     #     (f_med, f_std) = pickle.load(f)
#     # g_abs += f_med(g_mag)
#     # g_abs_std = f_std(g_mag)

#     # Color
#     gr_color = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 1]).data
#     # with open('./Catalog/completeness/gr_meas_err.pkl', 'rb') as f:
#     #     (f_med, f_std) = pickle.load(f)
#     # gr_color += f_med(gr_color)
#     # gr_color_std = f_std(gr_color)

#     gi_color = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 2]).data
#     # with open('./Catalog/completeness/gi_meas_err.pkl', 'rb') as f:
#     #     (f_med, f_std) = pickle.load(f)
#     # gi_color += f_med(gi_color)
#     # gi_color_std = f_std(gi_color)

#     # average over g-i and g-r results
#     log_ML_g = np.array([1.297 * gi_color - 0.855,
#                          1.774 * gr_color - 0.783]).mean(axis=0)
#     cuts_cat['log_ML_g'] = log_ML_g
#     # log_ML_g_std = np.sqrt((1.297 * gi_color_std)**2 + (1.774 * gr_color_std)**2) / 2

#     log_m_g = -0.4 * (g_abs - 5.03) + log_ML_g
#     cuts_cat['log_m_star'] = log_m_g
#     # log_m_g_std = np.sqrt((0.4 * g_abs_std)**2 + log_ML_g_std**2)

#     with open('./Catalog/completeness/deblend_detection_comp.pkl', 'rb') as f:
#         f_comp = pickle.load(f)

#     comp = np.array([f_comp(*p)[0] for p in zip(R_e, SB_g_meas)])
#     cuts_cat['completeness'] = comp

#     return cuts_cat


# def post_process_cat_new(input_cuts_cat, fake_udg=False):
#     """
#     We correct for bias and add errors to the measured quantities.
#     """
#     import joblib
#     import astropy.units as u
#     import pickle

#     cuts_cat = input_cuts_cat.copy()
#     cuts_cat['rhalf_spergel'] *= 0.168
#     cuts_cat['rhalf_circ'] *= 0.168
#     cuts_cat['rhalf_circularized'] *= 0.168
#     cuts_cat['rhalf_ellip'] *= 0.168

#     # Apply bias correction
#     re_meas = cuts_cat['rhalf_spergel'].data
#     SB_meas = {}
#     SB_err = {}
#     SB_meas['g'] = cuts_cat['SB_eff_avg'].data[:, 0]
#     SB_meas['r'] = cuts_cat['SB_eff_avg'].data[:, 1]
#     SB_meas['i'] = cuts_cat['SB_eff_avg'].data[:, 2]

#     mag_meas = {}
#     mag_err = {}
#     mag_meas['g'] = cuts_cat['mag'][:, 0].data
#     mag_meas['r'] = cuts_cat['mag'][:, 1].data
#     mag_meas['i'] = cuts_cat['mag'][:, 2].data

#     gi_meas = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 2]).data
#     gr_meas = (cuts_cat['mag'][:, 1] - cuts_cat['mag'][:, 2]).data

#     X = np.vstack([re_meas, SB_meas['g']]).T

#     # Re
#     re_bias = joblib.load('./Catalog/mock_sample/spergel/bias_std/re_bias.pkl')
#     re_std = joblib.load('./Catalog/mock_sample/spergel/bias_std/re_std.pkl')
#     bias = re_bias.predict(X)
#     std = re_std.predict(X)
#     std = np.sqrt(std**2 + 0.2**2)
#     re_meas += bias
#     re_err = std

#     # SB
#     for filt in list('gri'):
#         SB_bias = joblib.load(
#             f'./Catalog/mock_sample/spergel/bias_std/SB_eff_{filt}_bias.pkl')
#         SB_std = joblib.load(
#             f'./Catalog/mock_sample/spergel/bias_std/SB_eff_{filt}_std.pkl')
#         bias = SB_bias.predict(X)
#         std = SB_std.predict(X)
#         std = np.sqrt(std**2 + 0.05**2)

#         SB_meas[filt] += bias
#         SB_err[filt] = std

#     # mag
#     for filt in list('gri'):
#         mag_bias = joblib.load(
#             f'./Catalog/mock_sample/spergel/bias_std/SB_eff_{filt}_bias.pkl')
#         mag_std = joblib.load(
#             f'./Catalog/mock_sample/spergel/bias_std/SB_eff_{filt}_std.pkl')
#         bias = mag_bias.predict(X)
#         std = mag_std.predict(X)
#         std = np.sqrt(std**2 + 0.05**2)

#         mag_meas[filt] += bias
#         mag_err[filt] = std

#     # color
#     gi_bias = joblib.load('./Catalog/mock_sample/spergel/bias_std/gi_bias.pkl')
#     gi_std = joblib.load('./Catalog/mock_sample/spergel/bias_std/gi_std.pkl')
#     bias = gi_bias.predict(X)
#     std = gi_std.predict(X)
#     std = np.sqrt(std**2 + 0.05**2)
#     gi_meas += bias
#     gi_err = std

#     gr_bias = joblib.load('./Catalog/mock_sample/spergel/bias_std/gr_bias.pkl')
#     gr_std = joblib.load('./Catalog/mock_sample/spergel/bias_std/gr_std.pkl')
#     bias = gr_bias.predict(X)
#     std = gr_std.predict(X)
#     std = np.sqrt(std**2 + 0.05**2)
#     gr_meas += bias
#     gr_err = std

#     # Write to catalog
#     cuts_cat['rhalf_spergel'] = re_meas * u.arcsec
#     cuts_cat['rhalf_spergel_err'] = re_err * u.arcsec

#     cuts_cat['mag'] = np.vstack(
#         [mag_meas['g'], mag_meas['r'], mag_meas['i']]).T * u.ABmag
#     cuts_cat['mag_err'] = np.vstack(
#         [mag_err['g'], mag_err['r'], mag_err['i']]).T * u.ABmag

#     cuts_cat['SB_eff_avg'] = np.vstack(
#         [SB_meas['g'], SB_meas['r'], SB_meas['i']]).T * u.ABmag
#     cuts_cat['SB_eff_avg_err'] = np.vstack(
#         [SB_err['g'], SB_err['r'], SB_err['i']]).T * u.ABmag

#     cuts_cat['g-i'] = gi_meas * u.ABmag
#     cuts_cat['g-i_err'] = gi_err * u.ABmag

#     cuts_cat['g-r'] = gr_meas * u.ABmag
#     cuts_cat['g-r_err'] = gr_err * u.ABmag

#     cuts_cat['host_ang_diam_dist'] = cuts_cat['host_ang_diam_dist'].data * u.Mpc
#     # Physical sizes
#     # not consider peculiar motion
#     ang_diam_dist = cuts_cat['host_ang_diam_dist'].data
#     R_e_phys = cuts_cat['rhalf_spergel'].data / \
#         206265 * ang_diam_dist * 1000  # in kpc
#     R_e_phys_std = cuts_cat['rhalf_spergel_err'].data / \
#         206265 * ang_diam_dist * 1000  # in kpc
#     cuts_cat['rhalf_phys'] = R_e_phys * u.kpc
#     cuts_cat['rhalf_phys_err'] = R_e_phys_std * u.kpc

#     # Absolute magnitudes
#     cuts_cat['abs_mag'] = cuts_cat['mag'] - 25 - 5 * \
#         np.log10(ang_diam_dist *
#                  (1 + cuts_cat['host_z'].data)**2)[:, np.newaxis]  # griz
#     cuts_cat['abs_mag_err'] = cuts_cat['mag_err'] * u.ABmag  # griz

#     # average over g-i and g-r results
#     log_ML_g = np.array([1.297 * cuts_cat['g-i'].data - 0.855,
#                          1.774 * cuts_cat['g-r'].data - 0.783]).mean(axis=0)
#     cuts_cat['log_ML_g'] = log_ML_g
#     cuts_cat['log_ML_g_err'] = np.abs(cuts_cat['g-i_err'].data * 1.297)

#     cuts_cat['log_m_star'] = -0.4 * \
#         (cuts_cat['abs_mag'][:, 0].data - 5.03) + log_ML_g
#     cuts_cat['log_m_star_err'] = np.sqrt(
#         (cuts_cat['abs_mag_err'].data[:, 0] * 0.4)**2 + cuts_cat['log_ML_g_err']**2)

#     with open('./Catalog/completeness/deblend_detection_comp_S16A.pkl', 'rb') as f:
#         f_comp = pickle.load(f)
#     comp = np.array([f_comp(*p)[0] for p in zip(cuts_cat['rhalf_spergel'].data,
#                                                 cuts_cat['SB_eff_avg'][:, 0].data)])
#     cuts_cat['completeness'] = comp

#     # units
#     cuts_cat['ra'] = cuts_cat['ra'].data * u.deg
#     cuts_cat['dec'] = cuts_cat['dec'].data * u.deg

#     cols1 = ['viz-id', 'ra', 'dec', 'image_flag', 'psf_flag', 'radius',
#              'flux', 'mag', 'mag_err', 'g-i', 'g-i_err', 'g-r', 'g-r_err',
#              'SB_eff_avg', 'SB_eff_avg_err',
#              'rhalf_spergel', 'rhalf_spergel_err', 'rhalf_phys', 'rhalf_phys_err',
#              'abs_mag', 'abs_mag_err', 'log_ML_g', 'log_ML_g_err',
#              'log_m_star', 'log_m_star_err', 'completeness', 'tract', 'patch', 'synth_id',
#              'mag_auto_g', 'mag_auto_r', 'mag_auto_i', 'flux_radius_ave_g', 'flux_radius_ave_i',
#              'A_g', 'A_r', 'A_i',
#              'is_candy', 'is_galaxy', 'is_tidal', 'is_outskirts', 'is_cirrus', 'is_junk',
#              'num_votes', 'good_votes', 'bad_votes']
#     if fake_udg:
#         cols2 = ['host_z', 'host_ang_diam_dist']
#     else:
#         cols2 = [
#             'host_name', 'host_z', 'host_ang_diam_dist', 'host_stellar_mass', 'host_halo_mass',
#             'host_r_vir', 'host_r_vir_ang', 'host_300kpc_ang', 'host_nvotes', 'host_spiral',
#             'host_elliptical', 'host_uncertain', 'host_RA', 'host_DEC']
#     cols3 = [
#         'SB_eff_circ', 'SB_0', 'ell_cen', 'ell_sym',
#         'PA_cen', 'PA_sym', 'rhalf_circularized', 'spergel_nu',
#         'r20', 'r50', 'r80', 'Gini', 'M20', 'F(G,M20)', 'S(G,M20)', 'C', 'A', 'S',
#         'A_outer', 'A_shape', 'sersic_n', 'sersic_rhalf', 'sersic_ell', 'sersic_PA', 'flag_sersic',
#     ]
#     cuts_cat = cuts_cat[cols1 + cols2 + cols3]
#     return cuts_cat


def post_process_cat_new_rbf(input_cuts_cat, fake_udg=False,):
    """
    We correct for bias and add errors to the measured quantities.

    Now the bias and error are estimated using an RBF interpolator,
    instead of using NN. NN is a little bit confusing and not smooth.

    The logic is:
    1. correct re_bias
    2. correct SB_g_bias
    3. correct g-i bias and g-r bias
    4. derive m_g based on 1 and 2. get std from file
    5. derive m_i and m_r based on 4 and 5. get std from file
    6. derive SB_i and SB_r based on 2 and 3. get std from file
    """
    import joblib
    import astropy.units as u
    import pickle

    cuts_cat = input_cuts_cat.copy()
    cuts_cat['rhalf_spergel'] *= 0.168
    cuts_cat['rhalf_circ'] *= 0.168
    cuts_cat['rhalf_circularized'] *= 0.168
    cuts_cat['rhalf_ellip'] *= 0.168

    # Apply bias correction
    re_meas = cuts_cat['rhalf_spergel'].data
    SB_meas = {}
    SB_err = {}
    SB_meas['g'] = cuts_cat['SB_eff_avg'].data[:, 0]
    SB_meas['r'] = cuts_cat['SB_eff_avg'].data[:, 1]
    SB_meas['i'] = cuts_cat['SB_eff_avg'].data[:, 2]

    mag_meas = {}
    mag_err = {}
    mag_meas['g'] = cuts_cat['mag'][:, 0].data
    mag_meas['r'] = cuts_cat['mag'][:, 1].data
    mag_meas['i'] = cuts_cat['mag'][:, 2].data

    gi_meas = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 2]).data
    gr_meas = (cuts_cat['mag'][:, 1] - cuts_cat['mag'][:, 2]).data

    X = np.vstack([re_meas, SB_meas['g']]).T

    # Re
    re_bias = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/re_bias_rbf.pkl')
    re_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/re_std_rbf.pkl')

    bias = re_meas * re_bias(X)
    re_meas += bias
    std = re_std(X)  # * re_meas
    std[std < 0.3] = 0.3
    re_err = std

    # SB_g
    filt = 'g'
    SB_bias = joblib.load(
        f'./Catalog/mock_sample/spergel/bias_std/SB/SB_eff_{filt}_bias_rbf.pkl')
    SB_std = joblib.load(
        f'./Catalog/mock_sample/spergel/bias_std/SB/SB_eff_{filt}_std_rbf.pkl')

    bias = SB_meas[filt] * SB_bias(X)
    SB_meas[filt] += bias
    std = SB_std(X) * SB_meas[filt]
    std[std < 0.05] = 0.05
    SB_err[filt] = std

    # color
    gi_bias = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/color/gi_bias_rbf.pkl')
    gi_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/color/gi_std_rbf.pkl')
    bias = gi_bias(X)
    std = gi_std(X)
    std[std < 0.05] = 0.05
    gi_meas += bias
    gi_err = std

    gr_bias = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/color/gr_bias_rbf.pkl')
    gr_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/color/gr_std_rbf.pkl')
    bias = gr_bias(X)
    std = gr_std(X)
    std[std < 0.05] = 0.05
    gr_meas += bias
    gr_err = std

    # mag
    # 1. g-band
    filt = 'g'
    mag_meas[filt] = SB_meas[filt] - 2.5 * np.log10(2 * np.pi * re_meas**2)
    mag_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/mag/g_std_rbf.pkl')
    std = mag_std(X) * mag_meas[filt]
    std[std < 0.05] = 0.05
    mag_err[filt] = std

    # 2. r-band
    filt = 'r'
    mag_meas[filt] = mag_meas['g'] - (gr_meas)
    mag_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/mag/r_std_rbf.pkl')
    std = mag_std(X) * mag_meas[filt]
    std[std < 0.05] = 0.05
    mag_err[filt] = std

    # 3. i-band
    filt = 'i'
    mag_meas[filt] = mag_meas['g'] - (gi_meas)
    mag_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/mag/i_std_rbf.pkl')
    std = mag_std(X) * mag_meas[filt]
    std[std < 0.05] = 0.05
    mag_err[filt] = std

    # SB
    # 2. r-band
    filt = 'r'
    SB_meas[filt] = SB_meas['g'] - (gr_meas)
    SB_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/SB/r_std_rbf.pkl')
    std = SB_std(X) * SB_meas[filt]
    std[std < 0.05] = 0.05
    SB_err[filt] = std

    # 3. i-band
    filt = 'i'
    SB_meas[filt] = SB_meas['g'] - (gi_meas)
    SB_std = joblib.load(
        './Catalog/mock_sample/spergel/bias_std/SB/i_std_rbf.pkl')
    std = SB_std(X) * SB_meas[filt]
    std[std < 0.05] = 0.05
    SB_err[filt] = std

    #############################################################################
    # Write to catalog
    cuts_cat['rhalf_spergel'] = re_meas * u.arcsec
    cuts_cat['rhalf_spergel_err'] = re_err * u.arcsec

    cuts_cat['mag'] = np.vstack(
        [mag_meas['g'], mag_meas['r'], mag_meas['i']]).T * u.ABmag
    cuts_cat['mag_err'] = np.vstack(
        [mag_err['g'], mag_err['r'], mag_err['i']]).T * u.ABmag

    cuts_cat['SB_eff_avg'] = np.vstack(
        [SB_meas['g'], SB_meas['r'], SB_meas['i']]).T * u.ABmag
    cuts_cat['SB_eff_avg_err'] = np.vstack(
        [SB_err['g'], SB_err['r'], SB_err['i']]).T * u.ABmag

    cuts_cat['g-i'] = gi_meas * u.ABmag
    cuts_cat['g-i_err'] = gi_err * u.ABmag

    cuts_cat['g-r'] = gr_meas * u.ABmag
    cuts_cat['g-r_err'] = gr_err * u.ABmag

    # Apply galactic extinction correction
    cuts_cat['mag'] -= np.vstack([cuts_cat['A_g'],
                                 cuts_cat['A_r'], cuts_cat['A_i']]).T
    cuts_cat['SB_eff_avg'] -= np.vstack([cuts_cat['A_g'],
                                        cuts_cat['A_r'], cuts_cat['A_i']]).T
    cuts_cat['g-r'] -= (cuts_cat['A_g'] - cuts_cat['A_r'])
    cuts_cat['g-i'] -= (cuts_cat['A_g'] - cuts_cat['A_i'])

    ###### Calculate physical quantities ######
    cuts_cat['host_ang_diam_dist'] = cuts_cat['host_ang_diam_dist'].data * u.Mpc
    # Physical sizes
    # not consider peculiar motion
    ang_diam_dist = cuts_cat['host_ang_diam_dist'].data
    R_e_phys = cuts_cat['rhalf_spergel'].data / \
        206265 * ang_diam_dist * 1000  # in kpc
    R_e_phys_std = cuts_cat['rhalf_spergel_err'].data / \
        206265 * ang_diam_dist * 1000  # in kpc
    cuts_cat['rhalf_phys'] = R_e_phys * u.kpc
    cuts_cat['rhalf_phys_err'] = R_e_phys_std * u.kpc

    # Absolute magnitudes
    cuts_cat['abs_mag'] = cuts_cat['mag'] - 25 - 5 * \
        np.log10(ang_diam_dist *
                 (1 + cuts_cat['host_z'].data)**2)[:, np.newaxis]  # griz
    cuts_cat['abs_mag_err'] = cuts_cat['mag_err'] * u.ABmag  # griz

    # average over g-i and g-r results
    log_ML_g = np.array([1.297 * cuts_cat['g-i'].data - 0.855,
                         1.774 * cuts_cat['g-r'].data - 0.783]).mean(axis=0)
    cuts_cat['log_ML_g'] = log_ML_g
    cuts_cat['log_ML_g_err'] = np.abs(cuts_cat['g-i_err'].data * 1.297)

    cuts_cat['log_m_star'] = -0.4 * \
        (cuts_cat['abs_mag'][:, 0].data - 5.03) + \
        log_ML_g  # the stellar mass is based on g-band magnitude
    cuts_cat['log_m_star_err'] = np.sqrt(
        (cuts_cat['abs_mag_err'].data[:, 0] * 0.4)**2 + cuts_cat['log_ML_g_err']**2)

    with open('./Catalog/completeness/deblend_detection_comp_S18A.pkl', 'rb') as f:
        f_comp = pickle.load(f)
    comp = np.array([f_comp(*p)[0] for p in zip(cuts_cat['rhalf_spergel'].data,
                                                cuts_cat['SB_eff_avg'][:, 0].data)])
    cuts_cat['completeness'] = comp

    # units
    cuts_cat['ra'] = cuts_cat['ra'].data * u.deg
    cuts_cat['dec'] = cuts_cat['dec'].data * u.deg

    cols1 = ['viz-id', 'ra', 'dec', 'image_flag', 'psf_flag', 'radius',
             'flux', 'mag', 'mag_err', 'g-i', 'g-i_err', 'g-r', 'g-r_err',
             'SB_eff_avg', 'SB_eff_avg_err',
             'rhalf_spergel', 'rhalf_spergel_err', 'rhalf_phys', 'rhalf_phys_err',
             'abs_mag', 'abs_mag_err', 'log_ML_g', 'log_ML_g_err',
             'log_m_star', 'log_m_star_err', 'completeness', 'tract', 'patch', 'synth_id',
             'mag_auto_g', 'mag_auto_r', 'mag_auto_i', 'flux_radius_ave_g', 'flux_radius_ave_i',
             'A_g', 'A_r', 'A_i',
             'is_candy', 'is_galaxy', 'is_tidal', 'is_outskirts', 'is_cirrus', 'is_junk',
             'num_votes', 'good_votes', 'bad_votes']
    if fake_udg:
        cols2 = ['host_z', 'host_ang_diam_dist']
    else:
        cols2 = [
            'host_name', 'host_z', 'host_ang_diam_dist', 'host_stellar_mass', 'host_halo_mass',
            'host_r_vir', 'host_r_vir_ang', 'host_300kpc_ang', 'host_gi', 'host_nvotes', 'host_spiral',
            'host_elliptical', 'host_uncertain', 'host_RA', 'host_DEC']
    cols3 = [
        'SB_eff_circ', 'SB_0', 'ell_cen', 'ell_sym',
        'PA_cen', 'PA_sym', 'rhalf_circularized', 'spergel_nu',
        'r20', 'r50', 'r80', 'Gini', 'M20', 'F(G,M20)', 'S(G,M20)', 'C', 'A', 'S',
        'A_outer', 'A_shape', 'sersic_n', 'sersic_rhalf', 'sersic_ell', 'sersic_PA', 'flag_sersic',
    ]
    cuts_cat = cuts_cat[cols1 + cols2 + cols3]
    return cuts_cat


##### Moving average #####
def moving_binned_statistic(x, values, x_err=None, statistic='mean', bins=10, range_=None, n_slide=20):
    range_0 = range_
    edges_0 = np.histogram_bin_edges(x, bins=bins, range=range_0)
    delta_x = np.diff(edges_0)[0] / n_slide

    output = np.zeros((n_slide, bins))
    cens = np.zeros((n_slide, bins))  # centers
    for k in np.arange(n_slide):
        i = k - n_slide // 2
        _range = range_0 + i * delta_x
        cens[k] = 0.5 * (edges_0[:-1] + edges_0[1:]) + i * delta_x
        if x_err is None:
            output[k] = binned_statistic(
                x, values, statistic=statistic, bins=bins, range=_range).statistic
        else:
            _x = np.log10(np.abs(10**x + np.random.normal(loc=0, scale=x_err)))
            output[k] = binned_statistic(
                _x, values, statistic=statistic, bins=bins, range=_range).statistic

    return output, cens[len(cens) // 2]


#################### Measurement error ####################

def get_edge_cens(edges):
    return [0.5 * (_edge[:-1] + _edge[1:]) for _edge in edges]


def bin_data(X, y, n_bins=[15, 15], min_num=20, statistic='median',
             _range=[[1, 15], [23, 30]]):
    """
    Bin the data for a given grid.
    For measurement error.
    """
    _extent = [item for sublist in _range for item in sublist]

    ret = binned_statistic_dd(X, y, statistic=statistic, bins=n_bins,
                              range=_range, expand_binnumbers=False)

    ret_cnt = binned_statistic_dd(X, y, statistic='count', bins=n_bins,
                                  range=_range,
                                  binned_statistic_result=ret)

    flag = (ret_cnt.statistic < min_num)
    ret.statistic[flag] = np.nan

    xs = np.meshgrid(*get_edge_cens(ret.bin_edges))
    X = np.vstack([x.ravel() for x in xs]).T
    y = ret.statistic
    y = y.ravel(order='F')  # f works for 2-D
    flag = ~np.isnan(y)
    X = X[flag]
    y = y[flag]

    return X, y


def bin_data_moving_window(X, y, n_slide=[2, 2], n_bins=[15, 15], _range=[[1, 15], [23, 30]],
                           min_num=20, statistic='median'):
    """
    Bin the data in a moving window fashion. 
    For measurement error.
    """
    range_0 = _range
    delta_x = np.diff(np.array(range_0), axis=1).ravel() / \
        np.array(n_bins) / np.array(n_slide)
    output = []
    cens = []
    for k in itertools.product(*[range(_n) for _n in n_slide]):
        i = k[0] - n_slide[0] // 2
        j = k[1] - n_slide[1] // 2
        _range = range_0 + (np.array([i, j]) * delta_x)[:, None]
        cen, out = bin_data(X, y, n_bins=n_bins, _range=_range,
                            statistic=statistic, min_num=min_num)
        cens.append(cen)
        output.append(out)

    return np.vstack(cens), np.concatenate(output)


def quant_error(X, y, n_bins=[15, 15], min_num=20, statistic='median',
                _range=[[1, 15], [23, 30]], min_err=0.2,
                vmin=0, vmax=3,
                method='poly', degree=[1, 1],):
    '''
    Fit the measurement error with polynomial. 
    '''
    _extent = [item for sublist in _range for item in sublist]

    x1, x2, y = bin_data(X, y, n_bins=n_bins, min_num=min_num, statistic=statistic,
                         _range=_range)

    from astropy.modeling import models, fitting
    if method == 'chebyshev':
        p_init = models.Chebyshev2D(x_degree=degree[0], y_degree=degree[1])
    else:
        p_init = models.Polynomial2D(degree=degree[0])

    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x1.ravel(), x2.ravel(), y.ravel())

    x1_test, x2_test = np.meshgrid(np.linspace(*_range[0], 100),
                                   np.linspace(*_range[1], 100))
    C = p(x1_test, x2_test)
    C[C < min_err] = min_err
    plt.pcolormesh(x1_test, x2_test, C, vmin=vmin, vmax=vmax)
    plt.colorbar(label=r'$\sigma(R_e)$ [arcsec]')

    plt.scatter(X[:, 0], X[:, 1], c=Y, s=35, vmin=vmin,
                vmax=vmax, edgecolors='whitesmoke', alpha=0.6)

    plt.xlabel(r'$R_e$ [arcsec]')
    plt.ylabel(r'$\overline{\mu}_{\rm eff} (g)$ [mag arcsec$^{-2}$]')

    return p


def predict_bias_NN(y, X, degree=2, hidden_layer_sizes=(64, 64, 64), random_state=42):
    '''
    Train NN to predict measurement error and bias.
    '''
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if random_state is None:
        random_state = 1
    # Polynomial features
    pipeline = make_pipeline(PolynomialFeatures(degree), StandardScaler())
    X_poly = pipeline.fit_transform(X)

    X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y,
                                                                  test_size=0.3,
                                                                  random_state=random_state)

    # NN regressor
    regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        random_state=random_state,
                        max_iter=500
                        ).fit(X_train_poly, y_train)
    print('Score:', regr.score(X_test_poly, y_test))
    pipeline2 = make_pipeline(pipeline, regr)
    return pipeline2


def quant_measurement(X, y,
                      n_slide=[5, 5],
                      n_bins=[15, 15], min_num=20, statistic='median',
                      method='NN', kernel_kwargs={},
                      _range=[[1, 15], [23, 30]], min_value=None, max_value=None,
                      degree=2, hidden_layer_sizes=(64, 64, 64),
                      display=True, xlim=[1, 15], ylim=[24, 28.5],
                      vmin=0, vmax=3, cbar_label=r'$\sigma(R_e)$ [arcsec]',):
    '''
    The machinery to characterize the measurement error.

    y as a function of X[0] and X[1].
    '''
    xs, y = bin_data_moving_window(
        X,
        y,
        n_slide=n_slide,
        n_bins=n_bins,
        _range=_range,
        statistic=statistic,
        min_num=min_num)

    if max_value is not None:
        xs = xs[y < max_value]
        y = y[y < max_value]

    from astropy.modeling import models, fitting

    if method == 'NN':
        ppl = predict_bias_NN(y, xs, degree=degree,
                              hidden_layer_sizes=hidden_layer_sizes)
        x1_test, x2_test = np.meshgrid(np.linspace(*_range[0], 100),
                                       np.linspace(*_range[1], 100))
        C = ppl.predict(np.vstack([x1_test.ravel(), x2_test.ravel()]).T)

    elif method == 'RBF':
        from scipy.interpolate import RBFInterpolator
        x1_test, x2_test = np.meshgrid(np.linspace(*_range[0], 100),
                                       np.linspace(*_range[1], 100))
        p = RBFInterpolator(xs, y, **kernel_kwargs)
        C = p(np.vstack([x1_test.ravel(), x2_test.ravel()]).T)
    else:
        if method == 'chebyshev':
            p_init = models.Chebyshev2D(x_degree=degree[0], y_degree=degree[1])
        elif method == 'poly':
            p_init = models.Polynomial2D(degree=degree[0])

        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, xs[:, 0].ravel(), xs[:, 1].ravel(), y.ravel())
        x1_test, x2_test = np.meshgrid(np.linspace(*_range[0], 100),
                                       np.linspace(*_range[1], 100))
        C = p(x1_test, x2_test)

    if min_value is not None:
        C[C < min_value] = min_value

    if display:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        plt.sca(ax1)
        plt.scatter(xs[:, 0], xs[:, 1], c=y, vmin=vmin,
                    vmax=vmax, s=25, alpha=0.8)
        plt.colorbar(label=cbar_label)
        plt.xlabel(r'$R_e$ [arcsec]')
        plt.ylabel(r'$\overline{\mu}_{\rm eff} (g)$ [mag arcsec$^{-2}$]')
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.sca(ax2)
        plt.pcolormesh(x1_test, x2_test, C.reshape(
            *x1_test.shape), vmin=vmin, vmax=vmax, alpha=1)
        plt.colorbar(label=cbar_label)
        plt.xlabel(r'$R_e$ [arcsec]')
        plt.ylabel(r'$\overline{\mu}_{\rm eff} (g)$ [mag arcsec$^{-2}$]')

    if method == 'NN':
        return ppl
    else:
        return p
