def init_extended_source(
    sky_coord,
    frame,
    observations,
    coadd=None,
    bg_cutoff=None,
    thresh=1,
    symmetric=True,
    monotonic="flat",
    min_grad=0.1,
):
    """Initialize the source that is symmetric and monotonic
    See `ExtendedSource` for a description of the parameters
    """
    try:
        iter(observations)
    except TypeError:
        observations = [observations]
    # determine initial SED from peak position
    # SED in the frame for source detection
    seds = []
    for obs in observations:
        if type(obs) is scarlet.LowResObservation:
            norm = "sum"
        else:
            norm = "max"
        _sed = scarlet.get_psf_sed(sky_coord, obs, frame, normalization=norm)
        seds.append(_sed)

    sed = np.concatenate(seds).flatten()
    if np.all(sed <= 0):
        # If the flux in all channels is  <=0,
        msg = f"Zero or negative SED {sed} at y={sky_coord[0]}, x={sky_coord[1]}"
        logger.warning(msg)
    if coadd is None:
        # which observation to use for detection and morphology
        try:
            bg_rms = np.array([[1 / np.sqrt(w[w > 0].mean()) for w in obs_.weights] for obs_ in observations])
        except:
            raise AttributeError(
                "Observation.weights missing! Please set inverse variance weights"
            )
        coadd, bg_cutoff = build_sed_coadd(seds, bg_rms, observations)
    else:
        if bg_cutoff is None:
            raise AttributeError(
                "background cutoff missing! Please set argument bg_cutoff"
            )
    # Apply the necessary constraints
    center = frame.get_pixel(sky_coord)
    if symmetric:
        morph = scarlet.operator.prox_uncentered_symmetry(
            coadd.copy(), 0, center=center, algorithm="sdss" # *1 is to artificially pass a variable that is not coadd
        )
    else:
        morph = coadd
    if monotonic:
        if monotonic is True:
            monotonic = "angle"
        # use finite thresh to remove flat bridges
        prox_monotonic = scarlet.operator.prox_weighted_monotonic(
            morph.shape, neighbor_weight=monotonic, center=center, min_gradient=min_grad
        )
        morph = prox_monotonic(morph, 0).reshape(morph.shape)
    origin = (np.array(frame.shape)/2).astype(int)
    origin[0]=0
    bbox = scarlet.Box(frame.shape,(0,0,0))

    #morph, bbox = trim_morphology(sky_coord, frame, morph, bg_cutoff, thresh)
    return sed, morph, bbox