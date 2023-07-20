# Make figures for the paper
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, Row, vstack, hstack
from sample_cuts import moving_binned_statistic
import pickle


# def plot_size_distribution(udg_cat, fake_udg_cat, udg_area, fake_udg_area, fake_udg_repeats=10 * 20, name='UDG',
#                            fit_line=False, refit=False, save=True,
#                            ax=None, n_bins=10, n_slide=20, range_0=np.array([np.log10(1.5), np.log10(6.1)]),
#                            top_banner=True, vline=False, dots_legend=True, legend_fontsize=14, verbose=False):
#     """
#     Distribution of the physical size of UDG/UPGs.

#     """
#     # R_e distribution of the real udg sample
#     output, cen = moving_binned_statistic(np.log10(udg_cat['rhalf_phys']),
#                                           np.log10(udg_cat['rhalf_phys']),
#                                           x_err=udg_cat['rhalf_phys_err'],
#                                           bins=n_bins, range_=range_0,
#                                           statistic='count', n_slide=n_slide)

#     n_cen = np.nanmean(output, axis=0) / (1) / udg_area / np.diff(cen)[0]
#     n_std1 = np.sqrt(np.nanmean(output, axis=0)) / \
#         (1) / udg_area / np.diff(cen)[0]  # Poisson error for counts
#     n_std2 = np.nanstd(output, axis=0) / (1) / udg_area / \
#         np.diff(cen)[0]  # error of shiting bins
#     n_std = np.sqrt(n_std1**2 + n_std2**2)

#     # R_e distribution of the fake udg sample: remove background
#     output, cen = moving_binned_statistic(np.log10(fake_udg_cat['rhalf_phys']),
#                                           np.log10(fake_udg_cat['rhalf_phys']),
#                                           x_err=fake_udg_cat['rhalf_phys_err'],
#                                           bins=n_bins, range_=range_0,
#                                           statistic='count', n_slide=n_slide)
#     n_cen_bkg = np.nanmean(output, axis=0) / (fake_udg_repeats) / \
#         fake_udg_area / np.diff(cen)[0]
#     n_std_bkg1 = np.sqrt(np.nanmean(output, axis=0)) / \
#         (fake_udg_repeats) / 24 / np.diff(cen)[0]  # Poisson error for counts
#     n_std_bkg2 = np.nanstd(output, axis=0) / (fake_udg_repeats) / \
#         fake_udg_area / np.diff(cen)[0]  # error of shiting bins
#     n_std_bkg = np.sqrt(n_std_bkg1**2 + n_std_bkg2**2)

#     # completeness in each size bin
#     output, cen = moving_binned_statistic(np.log10(udg_cat['rhalf_phys']),
#                                           udg_cat['completeness'],
#                                           x_err=udg_cat['rhalf_phys_err'],
#                                           bins=n_bins, range_=range_0,
#                                           statistic=np.nanmean,
#                                           n_slide=n_slide)
#     comp_avg = np.nanmean(output, axis=0)
#     comp_std = np.nanstd(output, axis=0)

#     ### Fit a linear line using numpyro ###
#     if fit_line and (~refit):
#         from numpyro.diagnostics import hpdi
#         predictions = np.load(
#             './Catalog/nsa_z001_004/{}_size_distribution_fit.npy'.format(name))
#         pred_mean = predictions.mean(axis=0)
#         pred_hpdi = hpdi(predictions, 0.68)

#     if fit_line and refit:
#         try:
#             import jax.numpy as jnp
#             from jax import random
#             import numpyro
#             import numpyro.distributions as dist
#             from numpyro.infer import MCMC, NUTS
#             from numpyro.infer import Predictive
#             from numpyro.diagnostics import hpdi
#         except ImportError:
#             print('Numpyro is not installed. Skipping fitting.')
#             fit_line = False

#         # don't include the last bin, because the statistic is very poor there
#         x0 = np.linspace(*range_0, 10)
#         x = cen[:-1]
#         y = ((n_cen - n_cen_bkg) / comp_avg)[:-1]
#         yerr = (np.sqrt(n_std**2 + n_std_bkg**2) / comp_avg)[:-1]

#         def model(x, y=None, yerr=0.1):
#             a = numpyro.sample('a', dist.Uniform(-20, 0))
#             b = numpyro.sample('b', dist.Uniform(-10, 10))
#             y_ = 10**(b + a * x)
#             # notice that we clamp the outcome of this sampling to the observation y
#             numpyro.sample('obs', dist.Normal(y_, yerr), obs=y)

#         # need to split the key for jax's random implementation
#         rng_key = random.PRNGKey(0)
#         rng_key, rng_key_ = random.split(rng_key)

#         # run HMC with NUTS
#         kernel = NUTS(model, target_accept_prob=0.9)
#         mcmc = MCMC(kernel, num_warmup=1000, num_samples=3000)
#         mcmc.run(rng_key_, x=x, y=y, yerr=yerr)
#         if verbose:
#             mcmc.print_summary()
#         samples = mcmc.get_samples()
#         predictive = Predictive(model, samples)
#         predictions = predictive(rng_key_, x=x0, yerr=0)['obs']
#         predictions = predictions[-1000:]
#         pred_mean = predictions.mean(axis=0)
#         pred_hpdi = hpdi(predictions, 0.68)
#         if save:
#             np.save(
#                 './Catalog/nsa_z001_004/{}_size_distribution_fit.npy'.format(name), predictions)

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     else:
#         plt.sca(ax)

#     sct1 = plt.errorbar(cen, n_cen, yerr=n_std,
#                         capsize=0,
#                         fmt='+', color='k', label='Raw counts')

#     sct2 = plt.errorbar(cen, n_cen_bkg, capsize=0,
#                         yerr=n_std_bkg, fmt='.', color='gray',
#                         markersize=5, label='Background')

#     sct3 = plt.errorbar(cen - 0.006, n_cen - n_cen_bkg, capsize=0,
#                         yerr=n_std, fmt='p', color='teal', alpha=0.4,
#                         markersize=5, label='Background subtracted')

#     sct4 = plt.errorbar(cen + 0.006, (n_cen - n_cen_bkg) / comp_avg,
#                         yerr=np.sqrt(n_std**2 + n_std_bkg**2) / comp_avg,
#                         fmt='s', color='orangered', alpha=0.9,
#                         markersize=7, label='Completeness corrected')
#     if dots_legend:
#         leg = plt.legend(loc=(0., 0.025), fontsize=legend_fontsize)
#         ax.add_artist(leg)

#     x0 = np.linspace(*range_0, 10)
#     line1 = plt.plot(x0, 10**(-2.71 * x0 + 2.49), ls='-.',
#                      color='dimgray', label='vdBurg+17')
#     if fit_line:
#         line2 = plt.plot(x0, pred_mean, color='salmon',
#                          lw=2, label='This work')
#         plt.fill_between(
#             x0, pred_hpdi[0], pred_hpdi[1], alpha=0.3, color='salmon', interpolate=True)
#         plt.legend(handles=[line1[0], line2[0]], fontsize=legend_fontsize)
#     else:
#         plt.legend(handles=[line1[0]], fontsize=legend_fontsize)

#     if vline:
#         plt.axvline(np.log10(1.5), ls=':', color='gray')
#     plt.xlabel(r'$\log\,r_e\ [\rm kpc]$')
#     plt.ylabel(r'$n\ [\rm dex^{-1}]$')
#     plt.yscale('log')

#     if top_banner:
#         ax2 = ax.twiny()
#         ax2.tick_params(direction='in')
#         lin_label = [1, 1.5, 2, 3, 5, 7, 9]
#         lin_pos = [np.log10(i) for i in lin_label]
#         ax2.set_xticks(lin_pos)
#         ax2.set_xlim(ax.get_xlim())
#         ax2.set_xlabel(r'kpc', fontsize=16)
#         ax2.xaxis.set_label_coords(1, 1.035)
#         ax2.tick_params(which='minor', top=False)

#         ax2.set_xticklabels(
#             [r'$\mathrm{' + str(i) + '}$' for i in lin_label], fontsize=16)
#         for tick in ax2.xaxis.get_major_ticks():
#             tick.label.set_fontsize(16)

#     if ax is None:
#         return fig, ax
#     else:
#         return ax.get_figure(), ax


def plot_size_distribution_new(udg_cat, fake_udg_cat, udg_area, fake_udg_area, fake_udg_repeats=10 * 20, name='UDG',
                               fit_line=False, refit=False, save=True, only_result=False, color=None,
                               ax=None, n_bins=10, n_slide=20, range_0=np.array([np.log10(1.5), np.log10(6.1)]),
                               top_banner=True, vline=False, dots_legend=True, legend_fontsize=14, verbose=False,
                               nolinelegend=False, label='Completeness corrected'):
    """
    Distribution of the physical size of UDG/UPGs.

    """
    log_re = np.log10(udg_cat['rhalf_phys'])
    log_re_err = udg_cat['rhalf_phys_err'] / udg_cat['rhalf_phys']

    # R_e distribution of the fake udg sample: remove background
    output, cen = moving_binned_statistic(np.log10(fake_udg_cat['rhalf_phys']),
                                          np.log10(fake_udg_cat['rhalf_phys']),
                                          x_err=fake_udg_cat['rhalf_phys_err'] /
                                          fake_udg_cat['rhalf_phys'],
                                          bins=n_bins, range_=range_0,
                                          statistic='count', n_slide=n_slide)
    n_cen_bkg = np.nanmean(output, axis=0) / (fake_udg_repeats) / \
        fake_udg_area / \
        np.diff(cen)[0]  # num of fake UDG per sqr deg per log_re
    n_std_bkg1 = np.sqrt(np.nanmean(output, axis=0)) / (fake_udg_repeats) / \
        fake_udg_area / np.diff(cen)[0]  # Poisson error for counts
    n_std_bkg2 = np.nanstd(output, axis=0) / (fake_udg_repeats) / \
        fake_udg_area / np.diff(cen)[0]  # error of shiting bins
    n_std_bkg = np.sqrt(n_std_bkg1**2 + n_std_bkg2**2)

    # R_e distribution of the real udg sample
    # First we need to know the total searched area
    unique_name, ind = np.unique(udg_cat['host_name'].data, return_index=True)
    # deg^2
    vir_areas = (np.pi * (udg_cat['host_r_vir_ang'].data[ind]**2))
    print('Total angular area [deg2]:', vir_areas.sum())

    result = {}
    result['n_cen'] = []
    result['n_std'] = []
    result['n_cen_nobkg'] = []

    for i in range(10):
        np.random.seed(i)
        _unique_name = unique_name  # np.random.choice(unique_name, size=200)
        n_cens = []
        n_cens_nobkg = []
        n_stds_hist = []
        for hostname in _unique_name:
            output, cen = moving_binned_statistic(log_re[udg_cat['host_name'] == hostname],
                                                  log_re[udg_cat['host_name']
                                                         == hostname],
                                                  x_err=udg_cat['rhalf_phys_err'][udg_cat['host_name']
                                                                                  == hostname],
                                                  bins=n_bins, range_=range_0,
                                                  statistic='count', n_slide=n_slide)
            area = (
                np.pi * (udg_cat['host_r_vir_ang'].data[[udg_cat['host_name'] == hostname]]**2))
            _n_cen = np.nanmean(output, axis=0) / np.diff(cen)[0]
            _n_cen_nobkg = _n_cen - n_cen_bkg * area[0]
            _n_std = np.nanstd(output, axis=0) / np.diff(cen)[0]
            # n_cens_nobkg.append(_n_cen_nobkg / area[0] * vir_areas.mean())
            # n_cens.append(_n_cen / area[0] * vir_areas.mean())
            # n_stds_hist.append(_n_std / area[0] * vir_areas.mean())
            n_cens_nobkg.append(_n_cen_nobkg)
            n_cens.append(_n_cen)
            n_stds_hist.append(_n_std)

        result['n_cen'].append(np.mean(n_cens, axis=0))
        result['n_cen_nobkg'].append(np.mean(n_cens_nobkg, axis=0))
        result['n_std'].append(
            np.sqrt(np.sum(np.array(n_stds_hist) ** 2, axis=0)) / len(n_stds_hist))

    # completeness in each size bin
    output, cen = moving_binned_statistic(np.log10(udg_cat['rhalf_phys']),
                                          udg_cat['completeness'],
                                          x_err=udg_cat['rhalf_phys_err'],
                                          bins=n_bins, range_=range_0,
                                          statistic=np.nanmean,
                                          n_slide=n_slide)
    comp_avg = np.nanmean(output, axis=0)
    comp_std = np.nanstd(output, axis=0)

    n_cen = np.mean(np.array(result['n_cen_nobkg']), axis=0)
    n_std = np.mean(np.array(result['n_std']), axis=0)
    n_cen_bkg = n_cen_bkg * np.mean(vir_areas)
    n_std_bkg = n_std_bkg * np.mean(vir_areas)
    n_corr = n_cen / comp_avg
    n_corr_std = n_std / comp_avg
    n_corr_std = np.sqrt((n_std / n_cen)**2 +
                         (comp_std / comp_avg)**2) * n_corr

    ### Fit a linear line using numpyro ###
    if fit_line and (~refit):
        from numpyro.diagnostics import hpdi
        predictions = np.load(
            './Catalog/nsa_z001_004/{}_size_distribution_fit.npy'.format(name))
        pred_mean = predictions.mean(axis=0)
        pred_hpdi = hpdi(predictions, 0.68)

    if fit_line and refit:
        try:
            import jax.numpy as jnp
            from jax import random
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
            from numpyro.infer import Predictive
            from numpyro.diagnostics import hpdi, summary
        except ImportError:
            print('Numpyro is not installed. Skipping fitting.')
            fit_line = False

        # don't include the last bin, because the statistic is very poor there
        x0 = np.linspace(*range_0, 10)
        x = cen[:-1]
        y = n_corr[:-1]
        yerr = n_corr_std[:-1]

        def model(x, y=None, yerr=0.1):
            a = numpyro.sample('a', dist.Uniform(-10, 0))
            b = numpyro.sample('b', dist.Uniform(-10, 10))
            y_ = 10**(b + a * x)
            # notice that we clamp the outcome of this sampling to the observation y
            numpyro.sample('obs', dist.Normal(y_, yerr), obs=y)

        # need to split the key for jax's random implementation
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # run HMC with NUTS
        kernel = NUTS(model, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000)
        mcmc.run(rng_key_, x=x, y=y, yerr=yerr)
        if verbose:
            mcmc.print_summary()
        samples = mcmc.get_samples()
        summary_dict = summary(samples, group_by_chain=False)
        predictive = Predictive(model, samples)
        predictions = predictive(rng_key_, x=x0, yerr=0)['obs']
        predictions = predictions[-2000:]
        pred_mean = predictions.mean(axis=0)
        pred_hpdi = hpdi(predictions, 0.95)
        if save:
            np.save(
                './Catalog/nsa_z001_004/{}_size_distribution_fit.npy'.format(name), predictions)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        plt.sca(ax)

    if only_result:
        color = 'orangered' if color is None else color
        sct4 = plt.errorbar(cen + 0.006, n_corr,
                            yerr=n_corr_std,
                            fmt='s', color=color, alpha=0.9,
                            markersize=7, label=label)
    else:
        sct1 = plt.errorbar(cen, n_cen + n_cen_bkg, yerr=n_std,
                            capsize=0,
                            fmt='+', color='k', label='Raw counts')

        sct2 = plt.errorbar(cen, n_cen_bkg, capsize=0,
                            yerr=n_std_bkg, fmt='.', color='gray',
                            markersize=5, label='Background')

        sct3 = plt.errorbar(cen - 0.006, n_cen, capsize=0,
                            yerr=n_std, fmt='p', color='teal', alpha=0.4,
                            markersize=5, label='Background subtracted')

        sct4 = plt.errorbar(cen + 0.006, n_corr,
                            yerr=n_corr_std,
                            fmt='s', color='orangered', alpha=0.9,
                            markersize=7, label='Completeness corrected')
    if dots_legend:
        leg = plt.legend(loc=(0., 0.025), fontsize=legend_fontsize)
        ax.add_artist(leg)

    x0 = np.linspace(*range_0, 10)
    line1 = plt.plot(x0, 10**(-2.71 * x0 + 1.49), ls='-.',
                     color='dimgray', label=r'vdBurg+17: $\mathrm{d} n / \mathrm{d} \log r_e \propto r_e^{-2.71\pm0.33}$')
    if fit_line:
        line2 = plt.plot(x0, pred_mean, color='salmon',
                         lw=2, label=r'This work: $\mathrm{d} n / \mathrm{d}\log r_e \propto r_e^{' + f'{summary_dict["a"]["mean"]:.2f}\pm{summary_dict["a"]["std"]:.2f}' + '}$')
        plt.fill_between(
            x0, pred_hpdi[0], pred_hpdi[1], alpha=0.3, color='salmon', interpolate=True)
        if not nolinelegend:
            plt.legend(handles=[line1[0], line2[0]], fontsize=legend_fontsize)
    else:
        if not nolinelegend:
            plt.legend(handles=[line1[0]], fontsize=legend_fontsize)

    if vline:
        plt.axvline(np.log10(1.5), ls=':', color='gray')
    plt.xlabel(r'$\log\,r_e\ [\rm kpc]$')
    plt.ylabel(r'$\mathrm{d} n / \mathrm{d}\,\log\,r_e$')
    # plt.ylabel(r'$\frac{\mathrm{d} n}{\mathrm{d} \log\,r_e}\ [\rm dex^{-1}]$')
    plt.yscale('log')

    if top_banner:
        ax2 = ax.twiny()
        ax2.tick_params(direction='in')
        lin_label = [1, 1.5, 2, 3, 5, 7, 9]
        lin_pos = [np.log10(i) for i in lin_label]
        ax2.set_xticks(lin_pos)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel(r'kpc', fontsize=16)
        ax2.xaxis.set_label_coords(1, 1.035)
        ax2.tick_params(which='minor', top=False)

        ax2.set_xticklabels(
            [r'$\mathrm{' + str(i) + '}$' for i in lin_label], fontsize=16)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    if ax is None:
        return fig, ax
    else:
        return ax.get_figure(), ax


def plot_radial_number_profile(udg_cat, fake_udg_cat, fake_udg_area, fake_udg_repeats=10 * 20, name='UDG',
                               refit=False, save=True, r_min=0.2, amp_vdb16=0.5,
                               ax=None, n_bins=13, n_slide=10, range_0=np.array([0.05, 1.0]),
                               dots_legend=True, lines_legend=True, legend_fontsize=15, verbose=False):
    from colossus.cosmology import cosmology
    from colossus.halo import profile_nfw, profile_einasto
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    cosmology.setCosmology('planck15')

    # Calculate the projected distance from LSBG to host in the unit of R_Vir
    udg_coord = SkyCoord(udg_cat['ra'], udg_cat['dec'], unit='deg')
    host_coord = SkyCoord(udg_cat['host_RA'], udg_cat['host_DEC'], unit='deg')
    dist = udg_coord.separation(host_coord)
    dist_ratio = dist.to(u.deg).value / udg_cat['host_r_vir_ang'].data

    unique_name, ind = np.unique(udg_cat['host_name'].data, return_index=True)

    # number density per square degree of contaminants
    density_bkg = len(fake_udg_cat) / (fake_udg_repeats) / \
        fake_udg_area  # bkg per sqr deg

    # completeness in each radial distance bin
    output, cen = moving_binned_statistic(dist_ratio,
                                          udg_cat['completeness'],
                                          x_err=None,
                                          bins=n_bins, range_=range_0,
                                          statistic=np.nanmean,
                                          n_slide=n_slide)
    comp_avg = np.nanmean(output, axis=0)
    comp_std = np.nanstd(output, axis=0)

    # Method 1: bootstrap
    result = {}
    result['n_cen'] = []
    result['n_std'] = []
    result['n_cen_nobkg'] = []
    bins = np.histogram_bin_edges(dist_ratio, bins=n_bins, range=range_0)

    if refit:
        for i in range(20):
            np.random.seed(i)
            _unique_name = np.random.choice(unique_name, size=200)
            n_cens = []
            n_stds_hist = []
            for hostname in _unique_name:
                output, cen = moving_binned_statistic(dist_ratio[udg_cat['host_name'] == hostname],
                                                      dist_ratio[udg_cat['host_name']
                                                                 == hostname],
                                                      x_err=None,
                                                      bins=n_bins, range_=range_0,
                                                      statistic='count', n_slide=n_slide)
                # number per annulus area [R_vir^2]
                _n_cen = np.nanmean(output, axis=0) / \
                    (np.diff(np.pi * bins**2))
                # error due to histogram
                _n_std_poisson = np.sqrt(_n_cen)
                _n_std = np.nanstd(output, axis=0) / (np.diff(np.pi * bins**2))
                _n_std = np.sqrt(_n_std**2 + _n_std_poisson**2)

                contam_profile = density_bkg * np.diff(
                    np.pi * (bins * udg_cat[udg_cat['host_name']
                                            == hostname]['host_r_vir_ang'][:, None])**2
                ) / (np.diff(np.pi * bins**2))

                n_cens.append(_n_cen - contam_profile[0])
                n_stds_hist.append(_n_std)

            n_cen = np.mean(n_cens, axis=0)
            n_std = np.sqrt(np.sum(np.array(n_stds_hist) **
                            2, axis=0)) / len(n_stds_hist)
            result['n_cen'].append(n_cen)
            result['n_std'].append(n_std)

        if save:
            with open('./Catalog/nsa_z001_004/{}_radial_num_profile.pkl'.format(name), 'wb') as f:
                pickle.dump(result, f)
    else:
        with open('./Catalog/nsa_z001_004/{}_radial_num_profile.pkl'.format(name), 'rb') as f:
            result = pickle.load(f)

    # correct for the completeness
    n_cen = np.mean(np.array(result['n_cen']), axis=0)
    n_std = np.array(result['n_std']).mean(axis=0)
    # n_std = np.array(result['n_cen']).std(axis=0)
    n_corr = n_cen / comp_avg
    # n_corr_std = n_std / comp_avg
    n_corr_std = np.sqrt((n_std / n_cen)**2 +
                         (comp_std / comp_avg)**2) * n_corr

    # Fit NFW and Einasto profiles
    flag = (cen > r_min)
    p_nfw = profile_nfw.NFWProfile(rhos=10, rs=0.1, z=0.0, mdef='vir')
    res_nfw = p_nfw.fit(cen[flag], n_corr[flag],
                        quantity='Sigma',
                        q_err=n_std[flag]
                        )
    p_einasto = profile_einasto.EinastoProfile(
        M=10, alpha=0.8, c=1.83, z=0.0, mdef='vir')
    res_ein = p_einasto.fit(cen[flag], n_corr[flag],
                            q_err=n_corr_std[flag],
                            quantity='Sigma',
                            method='leastsq')
    nfw_conc = 1 / res_nfw['x'][1]
    ein_conc = 1 / res_ein['x'][1]
    ein_alpha = res_ein['x'][2]

    if verbose:
        print(f'NFW conc = {nfw_conc:.3f} +', 1 /
              res_nfw['x_err'].T[1] - nfw_conc)
        print(f'Einasto alpha = {ein_alpha:.3f} +',
              res_ein['x_err'].T[2] - ein_alpha)
        print(f'Einasto conc = {ein_conc:.3f} +',
              1 / res_ein['x_err'].T[1] - ein_conc)

    # Plot the result
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        plt.sca(ax)

    flag = (cen > r_min)
    ### Show r<r_min dots with light colors
    sct3 = plt.errorbar((cen - 0.006)[~flag], n_cen[~flag],
                        capsize=0,
                        yerr=n_std[~flag], fmt='p', color='teal', alpha=0.1,
                        markersize=5)
    sct4 = plt.errorbar((cen + 0.006)[~flag], n_corr[~flag],
                        yerr=n_corr_std[~flag],
                        fmt='s', color='orangered', alpha=0.4,
                        markersize=7, zorder=10)
    ### Show r>rmin with good colors
    sct3 = plt.errorbar((cen - 0.006)[flag], n_cen[flag],
                        capsize=0,
                        yerr=n_std[flag], fmt='p', color='teal', alpha=0.4,
                        markersize=5, label='Background subtracted')
    sct4 = plt.errorbar((cen + 0.006)[flag], n_corr[flag],
                        yerr=n_corr_std[flag],
                        fmt='s', color='orangered', alpha=0.9,
                        markersize=7, label='Completeness corrected', zorder=10)
    
    if dots_legend and lines_legend:
        leg = plt.legend(loc='upper right', fontsize=legend_fontsize)
        ax.add_artist(leg)
    elif dots_legend:
        plt.legend(loc='upper right', fontsize=legend_fontsize)

    r = np.linspace(0.15, 1, 100)
    sigma = p_einasto.surfaceDensity(r)
    line1 = plt.plot(r, sigma, ls='-', color='r', alpha=0.7,
                     lw=2, label='Einasto projected')

    sigma = p_nfw.surfaceDensity(r)
    line2 = plt.plot(r, sigma, ls='-', color='steelblue', lw=2,
                     label='NFW projected')

    p_ein_vdb = profile_einasto.EinastoProfile(
        rhos=amp_vdb16, alpha=0.92, rs=1 / 1.83, z=0.0, mdef='vir')
    line3 = plt.plot(r, p_ein_vdb.surfaceDensity(r), color='gray',
                     label='vdBurg+16', ls='--')

    if lines_legend:
        plt.legend(handles=[line1[0], line2[0], line3[0]],
                   fontsize=legend_fontsize, loc='upper right')  # (0.5, 0.6)

    # handles, labels = ax.get_legend_handles_labels()
    # order = [0, 1, 2, 3, 4]  # [3, 4, 5, 0, 1, 2]
    # plt.legend([handles[idx] for idx in order], [labels[idx]
    #                                              for idx in order])
    plt.xlabel(r'$R\ [R_{\rm vir}]$')
    plt.ylabel(r'$n\ [(R_{\rm vir})^{-2}]$')
    plt.xlim(0.0, 1.05)

    if ax is None:
        return fig, ax
    else:
        return ax.get_figure(), ax


def re_SB_distribution(udg_cat, ax, xlim=(28.4, 23), ylim=(0.8, 8), show_legend=True,):
    # Distribution of the full sample after junk cuts
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        plt.sca(ax)

    red = (udg_cat['g-i'].data > 0.8)

    markers, caps, bars = ax.errorbar(udg_cat[red]['SB_eff_avg'][:, 0],
                                      udg_cat[red]['rhalf_phys'],
                                      xerr=udg_cat[red]['SB_eff_avg_err'][:, 0],
                                      yerr=udg_cat[red]['rhalf_phys_err'],
                                      color='r', fmt='o', ms=4, alpha=0.3, label='$g-i > 0.8$', rasterized=True)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    markers, caps, bars = ax.errorbar(udg_cat[~red]['SB_eff_avg'][:, 0],
                                      udg_cat[~red]['rhalf_phys'],
                                      xerr=udg_cat[~red]['SB_eff_avg_err'][:, 0],
                                      yerr=udg_cat[~red]['rhalf_phys_err'],
                                      color='steelblue', fmt='o', ms=4, alpha=0.35, label='$g-i < 0.8$', rasterized=True)
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$\overline{\mu}_{\rm eff}(g)\ [\rm mag\ arcsec^{-2}\,]$')
    ax.set_ylabel(r'$r_e\ [\rm kpc]$')
    ax.set_yscale('log')
    plt.tick_params(axis='y', which='minor', right=False)
    ax.set_yticks([1], minor=False)
    ax.set_yticklabels([1], minor=False)
    ax.set_yticks([2, 3, 4, 5, 6, 7], minor=True)
    ax.set_yticklabels([2, 3, 4, 5, 6, 7], minor=True)

    ax_histx = ax.inset_axes((0, 1.02, 1, .2))
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.hist(udg_cat[red]['SB_eff_avg'][:, 0], lw=1.5,
                  histtype='step', density=False, color='r', alpha=0.5, label='$g-i > 0.8$')
    ax_histx.hist(udg_cat[~red]['SB_eff_avg'][:, 0], lw=1.5,
                  histtype='step', density=False, color='steelblue', label='$g-i < 0.8$')
    ax_histx.set_xlim(ax.get_xlim())

    ax_histy = ax.inset_axes((1.02, 0, 0.2, 1))
    ax_histy.tick_params(axis="y", which='both', labelleft=False)
    ax_histy.hist(udg_cat[red]['rhalf_phys'], lw=1.5,
                  histtype='step', density=False, orientation='horizontal', color='r', alpha=0.5)
    ax_histy.hist(udg_cat[~red]['rhalf_phys'], lw=1.5,
                  histtype='step', density=False, orientation='horizontal', color='steelblue')
    ax_histy.set_yscale('log')
    ax_histy.set_ylim(ax.get_ylim())

    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    if show_legend:
        leg = ax_histx.legend(loc=(0.01, 0.17), frameon=False, fontsize=13)

    if ax is None:
        return fig, [ax, ax_histx, ax_histy]
    else:
        return ax.get_figure(), [ax, ax_histx, ax_histy]


def quenched_frac(udg_cat, fake_udg_cat, fake_udg_num, udg_area, fake_udg_area, fake_udg_repeats=10 * 20,
                  name=None, min_completeness=0.1, color_bins=20, quench_is_color=False, quenched_intercept=-0.23,
                  flag=None, mass_range=(6.4, 9), mass_bins=8,
                  ax=None, zorder=1, linecolor='firebrick', linewidth=1, fmt='o-', linealpha=0.85, fillalpha=0.4,
                  plot_udg=True, plot_elves=True, plot_saga=True, plot_MW_M31=False, plot_sim=False, plot_elves_upg=False):

    # Exclude objects with too small completeness
    udg_area *= np.sum(udg_cat['completeness'] >
                       min_completeness) / len(udg_cat)
    fake_udg_area *= np.sum(fake_udg_cat['completeness']
                            > min_completeness) / len(fake_udg_cat)

    if flag is not None:
        flag = flag[udg_cat['completeness'] > min_completeness]
    udg_cat = udg_cat[udg_cat['completeness'] > min_completeness]
    fake_udg_cat = fake_udg_cat[fake_udg_cat['completeness']
                                > min_completeness]

    # Calculate the total fraction of bkg contaminants
    p_contam = np.mean(fake_udg_num) * (1) / \
        fake_udg_area * udg_area / \
        len(udg_cat)  # the fraction of contamination in the UDG sample
    p_contam_std = np.std(fake_udg_num) * (1) / \
        fake_udg_area * udg_area / len(udg_cat)

    print(
        f'% of contaminants in the sample: {p_contam*100:.2f} +- {p_contam_std*100:.2f}')

    # Assign weights based on color
    n1, bins = np.histogram(udg_cat['g-i'],
                            bins=color_bins, range=(-0.1, 1.3), density=True)
    n2, bins = np.histogram(fake_udg_cat['g-i'],
                            bins=color_bins, range=(-0.1, 1.3), density=True)

    n_weights = p_contam * n2 / n1
    n_weights = np.nan_to_num(n_weights, nan=0.0, posinf=0.0, neginf=0.0)
    n_weights[n_weights >= 1] = 1

    # Below are weights for each galaxy in the udg_cat
    # only consider bkg contaminants
    weights1 = 1 - n_weights[np.digitize(udg_cat['g-i'], bins) - 1]
    # weights1 = np.ones(len(udg_cat))
    # also consider completeness. It is helpful to exclude objects with super small completeness.
    weights2 = weights1 / udg_cat['completeness']
    weights2 = np.nan_to_num(weights2, posinf=0.0, neginf=0.0)
    # just to make the weights larger for visualization purpose
    weights2 /= np.percentile(weights2, 80)
    np.savetxt(f'{name}_weights.txt', weights2)

    # Calculate V-band abs mag according to Lupton 2005: https://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html#Lupton2005
    V = udg_cat['mag'][:, 0] - 0.5784 * \
        (udg_cat['mag'][:, 0] - udg_cat['mag'][:, 1]) - 0.0038
    V_abs = V - 25 - 5 * np.log10(udg_cat['host_ang_diam_dist'].data *
                                  (1 + udg_cat['host_z'].data)**2)
    # The criterion for quenching is from Carlsten+22. The original intercept is -0.23. I think 0.28 works better for our sample.
    quenched = (udg_cat['g-i'] > (-0.067 * V_abs + quenched_intercept))
    if quench_is_color:
        quenched = (udg_cat['g-i'] > 0.8)

    # Quenched frac data
    # This is based on H-alpha EW (from Yao)
    saga_m_star = np.array([6.791, 7.394, 7.739, 7.971, 8.145, 8.580, 9.119])
    saga_fq = np.array([0.056, 0.222, 0.222, 0.222, 0.167, 0.056, 0.053]) + \
        np.array([0.53, 0.168, 0.107, 0.041, 0.035, 0.025, 0.012])
    saga_fq_err = np.array([[-0.056, 0.071], [-0.102, 0.102], [-0.102, 0.102], [-0.102, 0.102],
                            [-0.094, 0.094], [-0.056, 0.071], [-0.053, 0.067]])

    # This is based on color
    saga_q = np.array([[6.747422680412371, 0.3474698795180722],
                       [7.247422680412371, 0.35903614457831323],
                       [7.747422680412371, 0.188433734939759],
                       [8.24742268041237, 0.19132530120481916],
                       [8.74742268041237, 0.11759036144578305],
                       [9.24742268041237, 0.010602409638553967]])
    saga_q_err = np.array([[-0.1, 0.12], [-0.084, 0.094], [-0.06, 0.078],
                           [-0.05, 0.07], [-0.05, 0.091], [-0.1, 0.1]])

    # MW + M31
    MW_M31_q = np.array([5.4949053857350805, 0.9487179487179488,
        6.496360989810771, 1.,
        7.491994177583697, 0.832579185520362,
        8.499272197962156, 0.6666666666666667,
        9.500727802037847, 0]).reshape(-1, 2)
    MW_M31_q_upper = np.array([5.500727802037846, 0.9834087481146305,
        6.496360989810771, 1.,
        7.497816593886464, 0.8959276018099548,
        8.49344978165939, 0.8129713423831071,
        9.500727802037847, 0.4570135746606335,]).reshape(-1, 2)
    MW_M31_q_lower = np.array([5.4949053857350805, 0.8506787330316743,
        6.496360989810771, 0.832579185520362,
        7.497816593886464, 0.6003016591251886,
        8.499272197962156, 0.3815987933634992,
        9.500727802037847, 0]).reshape(-1, 2)
    
    # ELVES
    elves_confirmed_q = np.array([5.742268041237113, 0.8506024096385543,
                                  6.252577319587629, 0.8650602409638555,
                                  6.742268041237113, 0.752289156626506,
                                  7.247422680412371, 0.6612048192771085,
                                  7.747422680412371, 0.7334939759036145,
                                  8.24742268041237, 0.392289156626506,
                                  8.74742268041237, 0.36337349397590357,
                                  9.24742268041237, 0.12626506024096373]).reshape(-1, 2)

    elves_confirmed_q_upper = np.array([5.752577319587629, 0.895421686746988,
                                        6.242268041237113, 0.9026506024096386,
                                        6.757731958762887, 0.7956626506024097,
                                        7.247422680412371, 0.7204819277108434,
                                        7.747422680412371, 0.7913253012048194,
                                        8.24742268041237, 0.4949397590361446,
                                        8.74742268041237, 0.5137349397590361,
                                        9.24742268041237, 0.28385542168674693]).reshape(-1, 2)

    elves_confirmed_q_lower = np.array([5.752577319587629, 0.7927710843373494,
                                        6.252577319587629, 0.8144578313253013,
                                        6.742268041237113, 0.6959036144578314,
                                        7.247422680412371, 0.5961445783132531,
                                        7.747422680412371, 0.6684337349397591,
                                        8.24742268041237, 0.29831325301204814,
                                        8.74742268041237, 0.23759036144578305,
                                        9.231958762886597, 0.05831325301204815]).reshape(-1, 2)

    samuel_q = np.array([[5.48, 1],
                         [6.48, 0.9837662337662338],
                         [7.48, 0.5016233766233766],
                         [8.48, 0.168831168831169],
                         [9.48, 0.008116883116883189]])
    samuel_q_lower = np.array([[5.48, 0.9870129870129871],
                               [6.48, 0.9431818181818183],
                               [7.48, 0.4269480519480521],
                               [8.48, 0.11688311688311703],
                               [9.48, 0.004870129870129913]])
    samuel_q_upper = np.array([[5.48, 1.0],
                               [6.48, 0.9870129870129871],
                               [7.48, 0.5779220779220781],
                               [8.48, 0.28733766233766234],
                               [9.48, 0.45779220779220786]])
    
    # This is from Samuel paper. No error bars.
    font_q_old = np.array([5.809045226130653, 1,
                       6.464824120603015, 0.9769357495881383,
                       7.1206030150753765, 0.8929159802306426,
                       7.768844221105527, 0.6177924217462931,
                       8.077889447236181, 0.48434925864909384,
                       8.424623115577889, 0.33443163097199335,
                       9.065326633165828, 0.11037891268533773,
                       9.72110552763819, 0.008237232289950436]).reshape(-1, 2)
    
    # Now this is from Font's paper, not Samuel paper.
    font_q = np.array([5.746223564954683, 1,
                    6.24773413897281, 0.9982269503546098,
                    6.7492447129909365, 0.9556737588652482,
                    7.244712990936556, 0.8386524822695035,
                    7.75226586102719, 0.624113475177305,
                    8.253776435045317, 0.3936170212765958,
                    8.749244712990937, 0.2358156028368794,
                    9.256797583081571, 0.042553191489361764,
                    9.75226586102719, 0.]).reshape(-1, 2)
    font_q_lower = np.array([5.74622356e+00, 9.80496454e-01, 6.24773414e+00, 9.73404255e-01,
       6.74924471e+00, 8.97163121e-01, 7.24471299e+00, 7.48226950e-01,
       7.75226586e+00, 4.96453901e-01, 8.25377644e+00, 2.74822695e-01,
       8.74924471e+00, 1.24113475e-01, 9.25679758e+00, 8.86524823e-03,
       9.75226586e+00, 0.00000000e+00]).reshape(-1, 2)
    font_q_upper = np.array([5.74622356, 1.        , 6.24773414, 1.        , 6.74924471,
       0.98404255, 7.24471299, 0.90248227, 7.75226586, 0.74113475,
       8.25377644, 0.52659574, 8.74924471, 0.40602837, 9.25679758,
       0.19326241, 9.75226586, 0.21276596]).reshape(-1, 2)

    

    greene_upg_q = np.array([[5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                             [1.0, 0.8198198198198198, 0.7752808988764045, 0.6666666666666666, 0.6666666666666666, 0.3333333333333333]]).T
    greene_upg_q_err = np.array([[5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                                 [0.14798573455920383, 0.13684434857425543, 0.21038378982621647, 0.27595045183908556, 0.26989033114002436, 0.2857142857142857]]).T

    greene_nonupg_q = np.array([[5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                                [0.8459715639810425, 0.8821892393320966, 0.7302977232924693, 0.6468129571577848, 0.6774193548387096, 0.42857142857142855]]).T
    greene_nonupg_q_err = np.array([[5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                                    [0.05556766016165791, 0.04391157106134102, 0.0587319236484889, 0.0690956594049264, 0.08395897088010118, 0.1079898494312077]]).T

        
    if flag is None:
        flag = np.ones_like(udg_cat['host_z']).data.astype(bool)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.9, 5.4))
    else:
        plt.sca(ax)
        
    if plot_MW_M31:
        plt.fill_between(MW_M31_q[:, 0],
                         MW_M31_q_upper[:, 1],
                         MW_M31_q_lower[:, 1],
                         color='firebrick',
                         alpha=0.1)
        plt.errorbar(MW_M31_q[:, 0], MW_M31_q[:, 1],
                     yerr=[-(MW_M31_q_lower[:, 1] - MW_M31_q[:, 1]), (MW_M31_q_upper[:, 1] - MW_M31_q[:, 1])],
                     fmt='s', mfc='firebrick', markersize=8,
                     color='firebrick',
                     )
        # plt.plot(MW_M31_q[:, 0], MW_M31_q[:, 1],
        #          color='firebrick', 
        #          ls='-.', lw=1)

    y1, cens = moving_binned_statistic(udg_cat['log_m_star'][flag],
                                       quenched[flag].astype(
                                           float) * weights2[flag],
                                       x_err=udg_cat['log_m_star_err'][flag],
                                       statistic='sum', range_=mass_range, bins=mass_bins, n_slide=20)

    y2, cens = moving_binned_statistic(udg_cat['log_m_star'][flag],
                                       weights2[flag],
                                       x_err=udg_cat['log_m_star_err'][flag],
                                       statistic='sum', range_=mass_range, bins=mass_bins, n_slide=20)

    num, cens = moving_binned_statistic(udg_cat['log_m_star'][flag],
                                        udg_cat['log_m_star'][flag],
                                        x_err=udg_cat['log_m_star_err'][flag],
                                        statistic='count', range_=mass_range, bins=mass_bins, n_slide=20)
    num = np.median(num, axis=0)

    quench_frac = np.median(y1 / y2, axis=0)
    quench_frac_std = np.sqrt(quench_frac * (1 - quench_frac) / num)
    quench_frac_std = np.std(y1 / y2, axis=0)

    quench_frac[num < 10] = np.nan
    quench_frac_std[num < 10] = np.nan
    
    if plot_udg:
        flag = (cens <= 8.6)
        plt.errorbar(cens[flag],
                    quench_frac[flag],
                    yerr=quench_frac_std[flag], fmt=fmt,
                    label=name,
                    color=linecolor, lw=linewidth, alpha=linealpha, zorder=zorder)
        plt.fill_between(cens[flag],
                        (quench_frac - quench_frac_std)[flag],
                        (quench_frac + quench_frac_std)[flag],
                        color=linecolor, alpha=fillalpha, zorder=zorder)

        flag = (cens > 8.2)
        plt.errorbar(cens[flag],
                    quench_frac[flag],
                    yerr=quench_frac_std[flag], fmt=fmt,
                    label=name,
                    color=linecolor, lw=linewidth, alpha=linealpha / 3, zorder=zorder)
        plt.fill_between(cens[flag],
                        (quench_frac - quench_frac_std)[flag],
                        (quench_frac + quench_frac_std)[flag],
                        color=linecolor, alpha=fillalpha / 3, zorder=zorder)

    # plt.errorbar(cens,
    #              quench_frac,
    #              yerr=quench_frac_std, fmt=fmt,
    #              label=name,
    #              color=linecolor, lw=linewidth, alpha=linealpha, zorder=zorder)
    # plt.fill_between(cens,
    #                  quench_frac - quench_frac_std,
    #                  quench_frac + quench_frac_std,
    #                  color=linecolor, alpha=fillalpha, zorder=zorder)
        
    if plot_elves:
        # ELVES Carlsten et al. average quenched fraction
        plt.plot(elves_confirmed_q[:, 0], elves_confirmed_q[:, 1],
                 color='orange',  # label='ELVES',
                 ls='-.', zorder=0, alpha=1, lw=2)
        plt.fill_between(elves_confirmed_q[:, 0],
                         elves_confirmed_q_upper[:, 1],
                         elves_confirmed_q_lower[:, 1],
                         color='orange',
                         alpha=0.3, zorder=2)
    if plot_saga:
        # SAGA f_q based on colors, from Carlsten et al. 2022 paper
        plt.errorbar(saga_q[:, 0], saga_q[:, 1],
                     yerr=[-saga_q_err[:, 0], saga_q_err[:, 1]],
                     fmt='s', mfc='none',
                     color='grey',
                     #  label='SAGA'
                     )
        # plt.errorbar(saga_m_star, saga_fq,
        #              yerr=[-saga_fq_err[:, 0], saga_fq_err[:, 1]],
        #              fmt='s', mfc='r',
        #              color='grey',
        #              #  label='SAGA'
        #              )


    if plot_sim:
        plt.plot(samuel_q[:, 0], samuel_q[:, 1],
                 color='hotpink',  # label='ELVES',
                 ls=':', zorder=0, alpha=1, lw=1.5)
        plt.fill_between(samuel_q[:, 0],
                         samuel_q_lower[:, 1],
                         samuel_q_upper[:, 1],
                         color='hotpink',
                         alpha=0.1, zorder=0)

        # ARTEMIS
        plt.plot(font_q[:, 0], font_q[:, 1],
                 color='#a125ff',  # label='ELVES',
                 ls='-', zorder=0, alpha=1, lw=1.5)
        plt.fill_between(font_q[:, 0],
                         font_q_lower[:, 1],
                         font_q_upper[:, 1],
                         color='#a125ff',
                         alpha=0.1, zorder=0)
        
    if plot_elves_upg:
        # Greene et al. UPG quenched fraction
        plt.plot(greene_upg_q[:, 0], greene_upg_q[:, 1],
                 color='#D2B48C',  # label='ELVES',
                 ls='-.', zorder=0, alpha=1, lw=2)
        plt.fill_between(greene_upg_q[:, 0],
                         np.minimum(
                             1, (greene_upg_q[:, 1] + greene_upg_q_err[:, 1])),
                         np.maximum(
                             0, (greene_upg_q[:, 1] - greene_upg_q_err[:, 1])),
                         color='#FAF0E6',
                         alpha=0.7, zorder=-1)
    plt.xlabel(r'$\log\ M_\star\ [M_\odot]$')
    plt.ylabel(r'Quenched Fraction')

    plt.ylim(0, 1.1)

    # return weights2
    if ax is None:
        return fig, ax
    else:
        return ax.get_figure(), ax


def plot_measurement_paper_onlymorph(lsbg_cat, meas_cat, axes=None,
                                     gal_zorder=2, candy_zorder=3, junk_zorder=0,
                                     gal_size=10, candy_size=20, junk_size=15,
                                     gal_alpha=0.1, candy_alpha=0.2, junk_alpha=0.1,
                                     gal_label=r'$\texttt{galaxy}$', candy_label=r'$\texttt{candy}$', junk_label=r'$\texttt{junk}$',
                                     gal_color='steelblue', candy_color='forestgreen', junk_color='r',
                                     ):
    from matplotlib.patches import Rectangle

    junk = (lsbg_cat['bad_votes'] > lsbg_cat['good_votes'])
    candy = (lsbg_cat['good_votes'] > lsbg_cat['bad_votes']) & (
        lsbg_cat['is_candy'] > lsbg_cat['is_galaxy'])
    gal = (~junk) & (~candy)

    candy = candy | gal

    print('# of Candy:', np.sum(candy))
    print('# of Gal:', np.sum(gal))
    print('# of Junk:', np.sum(junk))

    g_mag = meas_cat['mag'].data[:, 0]
    r_mag = meas_cat['mag'].data[:, 1]
    i_mag = meas_cat['mag'].data[:, 2]

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plt.sca(axes[0])
    plt.scatter(meas_cat['M20'][candy], meas_cat['Gini']
                [candy],
                color=candy_color, label=candy_label +
                f': {np.sum(candy)}',
                s=candy_size, alpha=candy_alpha,
                zorder=candy_zorder, rasterized=True)
    plt.scatter(meas_cat['M20'][junk], meas_cat['Gini']
                [junk],
                color=junk_color, label=junk_label +
                f': {np.sum(junk)}',
                s=junk_size, alpha=junk_alpha,
                zorder=junk_zorder, rasterized=True)
    # plt.scatter(meas_cat['M20'][gal], meas_cat['Gini']
    #             [gal],
    #             color=gal_color, label=gal_label +
    #             f': {np.sum(gal)}',
    #             s=gal_size, alpha=gal_alpha,
    #             zorder=gal_zorder, rasterized=True)
    plt.xlabel(r'$M_{20}$')
    plt.ylabel(r'Gini')
    x = np.linspace(-3, -1.1, 10)
    plt.plot(x, -0.136 * x + 0.37, color='k', ls='--', lw=2)
    plt.vlines(-1.1, ymin=0.3, ymax=-0.136 * -
               1.1 + 0.37, color='k', ls='--', lw=2)
    # plt.axhline(0.7, color='k', ls='--', lw=2)
    plt.xlim(-2.5, -0.4)
    plt.ylim(0.3, 0.9)

    lgd = plt.legend(loc='upper left',
                     bbox_to_anchor=(-0.08, 1.03), handletextpad=0.02, fontsize=20.5)

    for l in lgd.legendHandles:
        l._sizes = [30]
        l._alpha = 1.0
    # x = np.linspace(-3, -1.6, 10)
    # plt.plot(x, 0.136 * x + 0.788, color='orange')

    plt.sca(axes[1])
    plt.scatter(meas_cat['C'][candy], meas_cat['A_outer']
                [candy],
                color=candy_color, label=candy_label +
                f': {np.sum(candy)}',
                s=candy_size, alpha=candy_alpha,
                zorder=candy_zorder, rasterized=True)
    plt.scatter(meas_cat['C'][gal], meas_cat['A_outer']
                [gal],
                color=gal_color, label=gal_label +
                f': {np.sum(gal)}',
                s=gal_size, alpha=gal_alpha,
                zorder=gal_zorder, rasterized=True)
    plt.scatter(meas_cat['C'][junk], meas_cat['A_outer']
                [junk],
                color=junk_color, label=junk_label +
                f': {np.sum(junk)}',
                s=junk_size, alpha=junk_alpha,
                zorder=junk_zorder, rasterized=True)

    rect = Rectangle(
        (1.8, -0.12), (3.5 - 1.8), (0.8 - -0.12),
        lw=2., edgecolor='k', ls='--', facecolor='none', zorder=12)
    plt.gca().add_patch(rect)
    # plt.axvline(1.8, color='k', ls='--', lw=2)
    # plt.axvline(3.5, color='k', ls='--', lw=2)
    # plt.axhline(0.8, color='k', ls='--', lw=2)

    plt.xlim(0., 4.8)
    plt.ylim(-.1, 1.8)
    plt.xlabel(r'$C$')
    plt.ylabel(r'$A$')

    # lgd = plt.legend(loc='upper left',
    #                  bbox_to_anchor=(-0.08, 1.03), handletextpad=0.03)

    # for l in lgd.legendHandles:
    #     l._sizes = [30]
    #     l._alpha = 1.0

    if axes is None:
        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        return fig, axes
    else:
        return axes
