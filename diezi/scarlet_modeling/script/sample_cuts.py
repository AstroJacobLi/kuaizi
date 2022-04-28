"""
Make cuts based on color, SB, size, etc.
"""
import numpy as np

default_cuts_dict = {'color_bound': [0.1, 1.2],
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


def make_cuts_vanilla(lsbg_cat, meas_cat, cuts_dict=None):
    if cuts_dict is not None:
        cuts_dict = cuts_dict
    else:
        cuts_dict = default_cuts_dict

    junk = (lsbg_cat['bad_votes'] > lsbg_cat['good_votes'])
    candy = (lsbg_cat['good_votes'] > lsbg_cat['bad_votes']) & (
        lsbg_cat['is_candy'] > lsbg_cat['is_galaxy'])
    gal = (~junk) & (~candy)

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

    print('Candy:', np.sum(mask & candy))
    print('Gal:', np.sum(mask & gal))
    print('Junk:', np.sum(mask & junk))

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


def post_process_cat(input_cuts_cat):
    cuts_cat = input_cuts_cat.copy()
    import pickle
    # Calculate SB and R_e errors
    SB_g_meas = cuts_cat['SB_eff_avg'][:, 0].data
    R_e = cuts_cat['rhalf_spergel'].data * 0.168

    with open('./Catalog/completeness/Re_meas_err.pkl', 'rb') as f:
        (f_med, f_std) = pickle.load(f)
    # R_e += f_med(SB_g_meas)
    R_e_std = f_std(SB_g_meas)
    cuts_cat['rhalf_spergel'] = R_e
    cuts_cat['rhalf_spergel_err'] = R_e_std

    with open('./Catalog/completeness/SB_meas_err.pkl', 'rb') as f:
        (f_med, f_std) = pickle.load(f)
    SB_g = SB_g_meas  # + f_med(SB_g_meas)
    SB_g_std = f_std(SB_g_meas)
    cuts_cat['SB_eff_avg_g'] = SB_g
    cuts_cat['SB_eff_avg_g_err'] = SB_g_std

    # Physical sizes
    # not consider peculiar motion
    ang_diam_dist = cuts_cat['host_ang_diam_dist'].data
    R_e_phys = R_e / 206265 * ang_diam_dist * 1000  # in kpc
    R_e_phys_std = R_e_std / 206265 * ang_diam_dist * 1000  # in kpc
    cuts_cat['rhalf_phys'] = R_e_phys
    cuts_cat['rhalf_phys_err'] = R_e_phys_std

    # Absolute magnitudes
    cuts_cat['abs_mag'] = cuts_cat['mag'] - 25 - 5 * \
        np.log10(ang_diam_dist *
                 (1 + cuts_cat['host_z'].data)**2)[:, np.newaxis]  # griz

    g_mag = cuts_cat['mag'][:, 0].data
    g_abs = cuts_cat['abs_mag'][:, 0].data
    # with open('./Catalog/completeness/mag_g_meas_err.pkl', 'rb') as f:
    #     (f_med, f_std) = pickle.load(f)
    # g_abs += f_med(g_mag)
    # g_abs_std = f_std(g_mag)

    # Color
    gr_color = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 1]).data
    # with open('./Catalog/completeness/gr_meas_err.pkl', 'rb') as f:
    #     (f_med, f_std) = pickle.load(f)
    # gr_color += f_med(gr_color)
    # gr_color_std = f_std(gr_color)

    gi_color = (cuts_cat['mag'][:, 0] - cuts_cat['mag'][:, 2]).data
    # with open('./Catalog/completeness/gi_meas_err.pkl', 'rb') as f:
    #     (f_med, f_std) = pickle.load(f)
    # gi_color += f_med(gi_color)
    # gi_color_std = f_std(gi_color)

    # average over g-i and g-r results
    log_ML_g = np.array([1.297 * gi_color - 0.855,
                         1.774 * gr_color - 0.783]).mean(axis=0)
    cuts_cat['log_ML_g'] = log_ML_g
    # log_ML_g_std = np.sqrt((1.297 * gi_color_std)**2 + (1.774 * gr_color_std)**2) / 2

    log_m_g = -0.4 * (g_abs - 5.03) + log_ML_g
    cuts_cat['log_m_star'] = log_m_g
    # log_m_g_std = np.sqrt((0.4 * g_abs_std)**2 + log_ML_g_std**2)

    with open('./Catalog/completeness/deblend_detection_comp.pkl', 'rb') as f:
        f_comp = pickle.load(f)

    comp = np.array([f_comp(*p)[0] for p in zip(R_e, SB_g_meas)])
    cuts_cat['completeness'] = comp

    return cuts_cat
