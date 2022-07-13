# Make figures for the paper
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column, Row, vstack, hstack
from sample_cuts import moving_binned_statistic
import pickle


def plot_size_distribution(udg_cat, fake_udg_cat, udg_area, fake_udg_area, fake_udg_repeats=10 * 20, name='UDG',
                           fit_line=False, refit=False, save=True,
                           ax=None, n_bins=10, n_slide=20, range_0=np.array([np.log10(1.5), np.log10(6.1)]),
                           top_banner=True, vline=False, dots_legend=True, legend_fontsize=14, verbose=False):
    """
    Distribution of the physical size of UDG/UPGs.

    """
    # R_e distribution of the real udg sample
    output, cen = moving_binned_statistic(np.log10(udg_cat['rhalf_phys']),
                                          np.log10(udg_cat['rhalf_phys']),
                                          x_err=udg_cat['rhalf_phys_err'],
                                          bins=n_bins, range_=range_0,
                                          statistic='count', n_slide=n_slide)

    n_cen = np.nanmean(output, axis=0) / (1) / udg_area / np.diff(cen)[0]
    n_std1 = np.sqrt(np.nanmean(output, axis=0)) / \
        (1) / udg_area / np.diff(cen)[0]  # Poisson error for counts
    n_std2 = np.nanstd(output, axis=0) / (1) / udg_area / \
        np.diff(cen)[0]  # error of shiting bins
    n_std = np.sqrt(n_std1**2 + n_std2**2)

    # R_e distribution of the fake udg sample: remove background
    output, cen = moving_binned_statistic(np.log10(fake_udg_cat['rhalf_phys']),
                                          np.log10(fake_udg_cat['rhalf_phys']),
                                          x_err=fake_udg_cat['rhalf_phys_err'],
                                          bins=n_bins, range_=range_0,
                                          statistic='count', n_slide=n_slide)
    n_cen_bkg = np.nanmean(output, axis=0) / (fake_udg_repeats) / \
        fake_udg_area / np.diff(cen)[0]
    n_std_bkg1 = np.sqrt(np.nanmean(output, axis=0)) / \
        (fake_udg_repeats) / 24 / np.diff(cen)[0]  # Poisson error for counts
    n_std_bkg2 = np.nanstd(output, axis=0) / (fake_udg_repeats) / \
        fake_udg_area / np.diff(cen)[0]  # error of shiting bins
    n_std_bkg = np.sqrt(n_std_bkg1**2 + n_std_bkg2**2)

    # completeness in each size bin
    output, cen = moving_binned_statistic(np.log10(udg_cat['rhalf_phys']),
                                          udg_cat['completeness'],
                                          x_err=udg_cat['rhalf_phys_err'],
                                          bins=n_bins, range_=range_0,
                                          statistic=np.nanmean,
                                          n_slide=n_slide)
    comp_avg = np.nanmean(output, axis=0)
    comp_std = np.nanstd(output, axis=0)

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
            from numpyro.diagnostics import hpdi
        except ImportError:
            print('Numpyro is not installed. Skipping fitting.')
            fit_line = False

        # don't include the last bin, because the statistic is very poor there
        x0 = np.linspace(*range_0, 10)
        x = cen[:-1]
        y = ((n_cen - n_cen_bkg) / comp_avg)[:-1]
        yerr = (np.sqrt(n_std**2 + n_std_bkg**2) / comp_avg)[:-1]

        def model(x, y=None, yerr=0.1):
            a = numpyro.sample('a', dist.Uniform(-20, 0))
            b = numpyro.sample('b', dist.Uniform(-10, 10))
            y_ = 10**(b + a * x)
            # notice that we clamp the outcome of this sampling to the observation y
            numpyro.sample('obs', dist.Normal(y_, yerr), obs=y)

        # need to split the key for jax's random implementation
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # run HMC with NUTS
        kernel = NUTS(model, target_accept_prob=0.9)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=3000)
        mcmc.run(rng_key_, x=x, y=y, yerr=yerr)
        if verbose:
            mcmc.print_summary()
        samples = mcmc.get_samples()
        predictive = Predictive(model, samples)
        predictions = predictive(rng_key_, x=x0, yerr=0)['obs']
        predictions = predictions[-1000:]
        pred_mean = predictions.mean(axis=0)
        pred_hpdi = hpdi(predictions, 0.68)
        if save:
            np.save(
                './Catalog/nsa_z001_004/{}_size_distribution_fit.npy'.format(name), predictions)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        plt.sca(ax)

    sct1 = plt.errorbar(cen, n_cen, yerr=n_std,
                        capsize=0,
                        fmt='+', color='k', label='Raw counts')

    sct2 = plt.errorbar(cen, n_cen_bkg, capsize=0,
                        yerr=n_std_bkg, fmt='.', color='gray',
                        markersize=5, label='Background')

    sct3 = plt.errorbar(cen - 0.006, n_cen - n_cen_bkg, capsize=0,
                        yerr=n_std, fmt='p', color='teal', alpha=0.4,
                        markersize=5, label='Background subtracted')

    sct4 = plt.errorbar(cen + 0.006, (n_cen - n_cen_bkg) / comp_avg,
                        yerr=np.sqrt(n_std**2 + n_std_bkg**2) / comp_avg,
                        fmt='s', color='orangered', alpha=0.9,
                        markersize=7, label='Completeness corrected')
    if dots_legend:
        leg = plt.legend(loc=(0., 0.025), fontsize=legend_fontsize)
        ax.add_artist(leg)

    x0 = np.linspace(*range_0, 10)
    line1 = plt.plot(x0, 10**(-2.71 * x0 + 2.49), ls='-.',
                     color='dimgray', label='vdBurg+17')
    if fit_line:
        line2 = plt.plot(x0, pred_mean, color='salmon',
                         lw=2, label='This work')
        plt.fill_between(
            x0, pred_hpdi[0], pred_hpdi[1], alpha=0.3, color='salmon', interpolate=True)
        plt.legend(handles=[line1[0], line2[0]], fontsize=legend_fontsize)
    else:
        plt.legend(handles=[line1[0]], fontsize=legend_fontsize)

    if vline:
        plt.axvline(np.log10(1.5), ls=':', color='gray')
    plt.xlabel(r'$\log\,r_e\ [\rm kpc]$')
    plt.ylabel(r'$n\ [\rm dex^{-1}]$')
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
                               refit=False, save=True, r_min=0.2,
                               ax=None, n_bins=13, n_slide=10, range_0=np.array([0.05, 1.0]),
                               dots_legend=True, lines_legend=True, legend_fontsize=14, verbose=False):
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

    # completeness in each size bin
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
                _n_std = np.nanstd(output, axis=0) / (np.diff(np.pi * bins**2))

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
    n_std = np.array(result['n_cen']).std(axis=0)
    n_corr = n_cen / comp_avg
    n_corr_std = n_std / comp_avg

    # Fit NFW and Einasto profiles
    flag = (cen > r_min)
    p_nfw = profile_nfw.NFWProfile(rhos=10, rs=0.1, z=0.0, mdef='vir')
    res_nfw = p_nfw.fit(cen[flag], n_corr[flag],
                        quantity='Sigma',
                        q_err=n_std[flag - 1]
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

    sct3 = plt.errorbar(cen - 0.006, n_cen,
                        capsize=0,
                        yerr=n_std, fmt='p', color='teal', alpha=0.4,
                        markersize=5, label='Background subtracted')
    sct4 = plt.errorbar(cen + 0.006, n_corr,
                        yerr=n_corr_std,
                        fmt='s', color='orangered', alpha=0.9,
                        markersize=7, label='Completeness corrected', zorder=10)
    if dots_legend:
        leg = plt.legend(loc='upper right', fontsize=legend_fontsize)
        ax.add_artist(leg)

    r = np.linspace(0.15, 1, 100)
    sigma = p_einasto.surfaceDensity(r)
    line1 = plt.plot(r, sigma, ls='-', color='r', alpha=0.7,
                     lw=2, label='Einasto projected')

    sigma = p_nfw.surfaceDensity(r)
    line2 = plt.plot(r, sigma, ls='-', color='steelblue', lw=2,
                     label='NFW projected')

    p_ein_vdb = profile_einasto.EinastoProfile(
        rhos=0.5, alpha=0.92, rs=1 / 1.83, z=0.0, mdef='vir')
    line3 = plt.plot(r, p_ein_vdb.surfaceDensity(r), color='gray',
                     label='vdBurg+17', ls='--')

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
                                      color='r', fmt='o', ms=5, alpha=0.3, label='$g-i > 0.8$')
    [bar.set_alpha(0.1) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    markers, caps, bars = ax.errorbar(udg_cat[~red]['SB_eff_avg'][:, 0],
                                      udg_cat[~red]['rhalf_phys'],
                                      xerr=udg_cat[~red]['SB_eff_avg_err'][:, 0],
                                      yerr=udg_cat[~red]['rhalf_phys_err'],
                                      color='steelblue', fmt='o', ms=5, alpha=0.35, label='$g-i < 0.8$')
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
                  histtype='step', density=True, color='r', alpha=0.5, label='$g-i > 0.8$')
    ax_histx.hist(udg_cat[~red]['SB_eff_avg'][:, 0], lw=1.5,
                  histtype='step', density=True, color='steelblue', label='$g-i < 0.8$')
    ax_histx.set_xlim(ax.get_xlim())

    ax_histy = ax.inset_axes((1.02, 0, 0.2, 1))
    ax_histy.tick_params(axis="y", which='both', labelleft=False)
    ax_histy.hist(udg_cat[red]['rhalf_phys'], lw=1.5,
                  histtype='step', density=True, orientation='horizontal', color='r', alpha=0.5)
    ax_histy.hist(udg_cat[~red]['rhalf_phys'], lw=1.5,
                  histtype='step', density=True, orientation='horizontal', color='steelblue')
    ax_histy.set_yscale('log')
    ax_histy.set_ylim(ax.get_ylim())

    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    if show_legend:
        leg = ax_histx.legend(loc=(0.02, 0.17), frameon=False, fontsize=13)

    if ax is None:
        return fig, [ax, ax_histx, ax_histy]
    else:
        return ax.get_figure(), [ax, ax_histx, ax_histy]


def quenched_frac(udg_cat, fake_udg_cat, fake_udg_num, udg_area, fake_udg_area, fake_udg_repeats=10 * 20,
                  name=None, min_completeness=0.1, color_bins=20, quench_is_color=False, quenched_intercept=-0.23,
                  flag=None, mass_range=(6.4, 9), mass_bins=8,
                  ax=None, zorder=1, linecolor='firebrick', linewidth=1, fmt='o-', linealpha=0.85, fillalpha=0.4, plot_ref=True):

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
    # also consider completeness. It is helpful to exclude objects with super small completeness.
    weights2 = weights1 / udg_cat['completeness']
    weights2 = np.nan_to_num(weights2, posinf=0.0, neginf=0.0)
    # just to make the weights larger for visualization purpose
    weights2 /= np.percentile(weights2, 80)

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
    saga_q = np.array([[6.747422680412371, 0.3474698795180722],
                       [7.247422680412371, 0.35903614457831323],
                       [7.747422680412371, 0.188433734939759],
                       [8.24742268041237, 0.19132530120481916],
                       [8.74742268041237, 0.11759036144578305],
                       [9.24742268041237, 0.010602409638553967]])
    saga_q_err = np.array([[-0.1, 0.12], [-0.084, 0.094], [-0.06, 0.078],
                           [-0.05, 0.07], [-0.05, 0.091], [-0.1, 0.1]])

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

    if flag is None:
        flag = np.ones_like(udg_cat['host_z']).data.astype(bool)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.9, 5.4))
    else:
        plt.sca(ax)

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

    plt.errorbar(cens,
                 quench_frac,
                 yerr=quench_frac_std, fmt=fmt,
                 label=name,
                 color=linecolor, lw=linewidth, alpha=linealpha, zorder=zorder)
    plt.fill_between(cens,
                     quench_frac - quench_frac_std,
                     quench_frac + quench_frac_std,
                     color=linecolor, alpha=fillalpha, zorder=zorder)

    if plot_ref:
        plt.plot(elves_confirmed_q[:, 0], elves_confirmed_q[:, 1],
                 color='orange',  # label='ELVES',
                 ls='-.', zorder=0, alpha=1, lw=1.5)
        plt.fill_between(elves_confirmed_q[:, 0],
                         elves_confirmed_q_upper[:, 1],
                         elves_confirmed_q_lower[:, 1],
                         color='orange',
                         alpha=0.2)
        plt.errorbar(saga_q[:, 0], saga_q[:, 1],
                     yerr=[-saga_q_err[:, 0], saga_q_err[:, 1]], fmt='s',
                     color='grey',
                     #  label='SAGA'
                     )

    plt.xlabel(r'$\log\ M_\star\ [M_\odot]$')
    plt.ylabel(r'Quenched Fraction')

    plt.ylim(0, 1.06)

    if ax is None:
        return fig, ax
    else:
        return ax.get_figure(), ax
