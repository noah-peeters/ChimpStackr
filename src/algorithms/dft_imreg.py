import math
import cv2
import numpy as np
import pyfftw.interfaces.numpy_fft as fft
import scipy.ndimage as ndi
import scipy.ndimage.interpolation as ndii

# -- UTILITIES --#
# Examine the average value of the image at most radius pixels from the edge
def get_borderval(img, radius=None):
    if radius is None:
        mindim = min(img.shape)
        radius = max(1, mindim // 20)

    mask = np.zeros_like(img, dtype=np.bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[:radius, :] = True
    mask[-radius:, :] = True
    mean = np.median(img[mask])
    return mean


def wrap_angle(angles, ceil=2 * np.pi):
    """
    Args:
        angles (float or ndarray, unit depends on kwarg ``ceil``)
        ceil (float): Turnaround value
    """
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def get_apofield(shape, aporad):
    """
    Returns an array between 0 and 1 that goes to zero close to the edges.
    """
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, "Apodization radius %d too big for shape dim. %d" % (
            aporad,
            dim,
        )
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    apofield = np.outer(vecs[0], vecs[1])
    return apofield


def _apodize(what, aporad=None, ratio=None):
    """
    Given an image, it apodizes it (so it becomes quasi-seamless).
    When ``ratio`` is None, color near the edges will converge
    to the same colour, whereas when ratio is a float number, a blurred
    original image will serve as background.

    Args:
        what: The original image
        aporad (int): Radius [px], width of the band near the edges
            that will get modified
        ratio (float or None): When None, the apodization background will
            be a flat color.
            When a float number, the background will be the image itself
            convolved with Gaussian kernel of sigma (aporad / ratio).

    Returns:
        The apodized image
    """
    if aporad is None:
        mindim = min(what.shape)
        aporad = int(mindim * 0.12)
    apofield = get_apofield(what.shape, aporad)
    res = what * apofield
    if ratio is not None:
        ratio = float(ratio)
        bg = ndi.gaussian_filter(what, aporad / ratio, mode="wrap")
    else:
        bg = get_borderval(what, aporad // 2)
    res += bg * (1 - apofield)
    return res


def _logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[0])[:, np.newaxis]
    xx = np.linspace(-np.pi / 2.0, np.pi / 2.0, shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy**2 + xx**2)
    filt = 1.0 - np.cos(rads) ** 2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    return filt


def _get_pcorr_shape(shape):
    return (int(max(shape) * 1.0),) * 2


# TODO: Check if used multiple times
def _get_angles(shape):
    """
    In the log-polar spectrum, the (first) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective angle.
    """
    ret = np.zeros(shape, dtype=np.float64)
    ret -= np.linspace(0, np.pi, shape[0], endpoint=False)[:, np.newaxis]
    return ret


# TODO: Check if used multiple times
def _get_lograd(shape, log_base):
    """
    In the log-polar spectrum, the (second) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective scale.

    Returns:
        2D np.ndarray of shape ``shape``, -1 coord contains scales
            from 0 to log_base ** (shape[1] - 1)
    """
    ret = np.zeros(shape, dtype=np.float64)
    ret += np.power(log_base, np.arange(shape[1], dtype=float))[np.newaxis, :]
    return ret


def _get_constraint_mask(shape, log_base, constraints=None):
    """
    Prepare mask to apply to constraints to a cross-power spectrum.
    """
    if constraints is None:
        constraints = {}

    mask = np.ones(shape, float)

    # Here, we create masks that modulate picking the best correspondence.
    # Generally, we look at the log-polar array and identify mapping of
    # coordinates to values of quantities.
    if "scale" in constraints:
        scale, sigma = constraints["scale"]
        scales = fft.ifftshift(_get_lograd(shape, log_base))
        # vvv This issome kind of transformation of result of _get_lograd
        # vvv (log radius in pixels) to the linear scale.
        scales *= log_base ** (-shape[1] / 2.0)
        # This makes the scales array low near where scales is near 'scale'
        scales -= 1.0 / scale
        if sigma == 0:
            # there isn't: ascales = np.abs(scales - scale)
            # because scales are already low for values near 'scale'
            ascales = np.abs(scales)
            scale_min = ascales.min()
            mask[ascales > scale_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-(scales**2) / sigma**2)

    if "angle" in constraints:
        angle, sigma = constraints["angle"]
        angles = _get_angles(shape)
        # We flip the sign on purpose
        # TODO: ^^^ Why???
        angles += np.deg2rad(angle)
        # TODO: Check out the wrapping. It may be tricky since pi+1 != 1
        wrap_angle(angles, np.pi)
        angles = np.rad2deg(angles)
        if sigma == 0:
            aangles = np.abs(angles)
            angle_min = aangles.min()
            mask[aangles > angle_min] = 0
        elif sigma is None:
            pass
        else:
            mask *= np.exp(-(angles**2) / sigma**2)

    mask = fft.fftshift(mask)
    return mask


def _argmax2D(array):
    """
    Simple 2D argmax function with simple sharpness indication
    """
    amax = np.argmax(array)
    ret = list(np.unravel_index(amax, array.shape))

    return np.array(ret)


def _argmax_ext(array, exponent):
    """
    Calculate coordinates of the COM (center of mass) of the provided array.

    Args:
        array (ndarray): The array to be examined.
        exponent (float or 'inf'): The exponent we power the array with. If the
            value 'inf' is given, the coordinage of the array maximum is taken.

    Returns:
        np.ndarray: The COM coordinate tuple, float values are allowed!
    """

    # When using an integer exponent for _argmax_ext, it is good to have the
    # neutral rotation/scale in the center rather near the edges

    ret = None
    if exponent == "inf":
        ret = _argmax2D(array)
    else:
        col = np.arange(array.shape[0])[:, np.newaxis]
        row = np.arange(array.shape[1])[np.newaxis, :]

        arr2 = array**exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return np.zeros(2)
        arrprody = np.sum(arr2 * col) / arrsum
        arrprodx = np.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return np.array(ret)


def _get_subarr(array, center, rad):
    """
    Args:
        array (ndarray): The array to search
        center (2-tuple): The point in the array to search around
        rad (int): Search radius, no radius (i.e. get the single point)
            implies rad == 0
    """
    dim = 1 + 2 * rad
    subarr = np.zeros((dim,) * 2)
    corner = np.array(center) - rad
    for ii in range(dim):
        yidx = corner[0] + ii
        yidx %= array.shape[0]
        for jj in range(dim):
            xidx = corner[1] + jj
            xidx %= array.shape[1]
            subarr[ii, jj] = array[yidx, xidx]
    return subarr


def _interpolate(array, rough, rad=2):
    """
    Returns index that is in the array after being rounded.

    The result index tuple is in each of its components between zero and the
    array's shape.
    """
    rough = np.round(rough).astype(int)
    surroundings = _get_subarr(array, rough, rad)
    com = _argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def _get_success(array, coord, radius=2):
    """
    Given a coord, examine the array around it and return a number signifying
    how good is the "match".

    Args:
        radius: Get the success as a sum of neighbor of coord of this radius
        coord: Coordinates of the maximum. Float numbers are allowed
            (and converted to int inside)

    Returns:
        Success as float between 0 and 1 (can get slightly higher than 1).
        The meaning of the number is loose, but the higher the better.
    """
    coord = np.round(coord).astype(int)
    coord = tuple(coord)

    subarr = _get_subarr(array, coord, 2)

    theval = subarr.sum()
    theval2 = array[coord]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    success = np.sqrt(theval * theval2)
    return success


def argmax_angscale(array, log_base, exponent, constraints=None):
    """
    Given a power spectrum, we choose the best fit.

    The power spectrum is treated with constraint masks and then
    passed to :func:`_argmax_ext`.
    """
    mask = _get_constraint_mask(array.shape, log_base, constraints)
    array_orig = array.copy()

    array *= mask
    ret = _argmax_ext(array, exponent)
    ret_final = _interpolate(array, ret)

    success = _get_success(array_orig, tuple(ret_final), 0)
    return ret_final, success


def _get_log_base(shape, new_r):
    """
    Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Args:
        shape: Shape of the original image.
        new_r (float): The r-size of the log-polar transform array dimension.

    Returns:
        float: Base of the log-polar transform.
        The following holds:
        :math:`log\_base = \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
        or the equivalent :math:`log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.
    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    EXCESS_CONST = 1.1
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    log_base = np.exp(np.log(old_r) / new_r)
    return log_base


def _logpolar(image, shape, log_base, bgval=None):
    """
    Return log-polar transformed image
    Takes into account anisotropicity of the freq spectrum
    of rectangular images

    Args:
        image: The image to be transformed
        shape: Shape of the transformed image
        log_base: Parameter of the transformation, get it via
            :func:`_get_log_base`
        bgval: The backround value. If None, use minimum of the image.

    Returns:
        The transformed image
    """
    if bgval is None:
        bgval = np.percentile(image, 1)
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = _get_angles(shape)
    radius_x = _get_lograd(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    ndii.map_coordinates(
        image, [y, x], output=output, order=3, mode="constant", cval=bgval
    )
    return output


def _phase_correlation(im0, im1, callback=None, *args):
    """
    Computes phase correlation between im0 and im1

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X). Translation vector of (0, 0)
            means that the two images match.
    """
    if callback is None:
        callback = _argmax2D

    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def _get_ang_scale(ims, exponent="inf", constraints=None):
    """
    Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`
        constraints (dict, optional)

    Returns:
        tuple: Scale, angle. Describes the relationship of
        the subject image to the first one.
    """
    shape = ims[0].shape

    ims_apod = [_apodize(im) for im in ims]
    dfts = [fft.fftshift(fft.fft2(im)) for im in ims_apod]
    # TODO: No need to re-calculate every time. Images are the same shape (or atleast should be)
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    # TODO: No need to re-calculate every time
    pcorr_shape = _get_pcorr_shape(shape)
    # TODO: No need to re-calculate every time
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(np.abs(dft), pcorr_shape, log_base) for dft in dfts]

    (arg_ang, arg_rad), _ = _phase_correlation(
        stuffs[0],
        stuffs[1],
        argmax_angscale,
        log_base,
        exponent,
        constraints,
    )

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = wrap_angle(angle, 360)
    scale = log_base**arg_rad

    angle = -angle
    scale = 1.0 / scale

    if not 0.5 < scale < 2:
        raise ValueError(
            "Images are not compatible. Scale change %g too big to be true." % scale
        )

    return scale, angle


def _get_emslices(shape1, shape2):
    """
    Common code used by :func:`embed_to` and :func:`undo_embed`
    """
    slices_from = []
    slices_to = []
    for dim1, dim2 in zip(shape1, shape2):
        diff = dim2 - dim1
        # In fact: if diff == 0:
        slice_from = slice(None)
        slice_to = slice(None)

        # dim2 is bigger => we will skip some of their pixels
        if diff > 0:
            # diff // 2 + rem == diff
            rem = diff - (diff // 2)
            slice_from = slice(diff // 2, dim2 - rem)
        if diff < 0:
            diff *= -1
            rem = diff - (diff // 2)
            slice_to = slice(diff // 2, dim1 - rem)
        slices_from.append(slice_from)
        slices_to.append(slice_to)
    return slices_from, slices_to


def embed_to(where, what):
    """
    Given a source and destination arrays, put the source into
    the destination so it is centered and perform all necessary operations
    (cropping or aligning)

    Args:
        where: The destination array (also modified inplace)
        what: The source array

    Returns:
        The destination array
    """
    slices_from, slices_to = _get_emslices(where.shape, what.shape)

    where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
    return where


# TODO: Completely replace with cv2.warpAffine
def transform_img(
    img, scale=1.0, angle=0.0, tvec=(0, 0), mode="constant", bgval=None, order=1
):
    """
    Return translation vector to register images.

    Args:
        img (2D or 3D numpy array): What will be transformed.
            If a 3D array is passed, it is treated in a manner in which RGB
            images are supposed to be handled - i.e. assume that coordinates
            are (Y, X, channels).
            Complex images are handled in a way that treats separately
            the real and imaginary parts.
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        mode (string): The transformation mode (refer to e.g.
            :func:`scipy.ndimage.shift` and its kwarg ``mode``).
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc. Linear works surprisingly well.

    Returns:
        np.ndarray: The transformed img, may have another
        i.e. (bigger) shape than the source.
    """
    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        print(img.shape[2])
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], scale, angle, tvec, mode, bgval, order)
        return ret
    elif np.iscomplexobj(img):
        decomposed = np.empty(img.shape + (2,), float)
        decomposed[:, :, 0] = img.real
        decomposed[:, :, 1] = img.imag
        # The bgval makes little sense now, as we decompose the image
        res = transform_img(decomposed, scale, angle, tvec, mode, None, order)
        ret = res[:, :, 0] + 1j * res[:, :, 1]
        return ret

    if bgval is None:
        bgval = get_borderval(img)

    bigshape = np.round(np.array(img.shape) * 1.2).astype(int)
    bg = np.zeros(bigshape, img.dtype) + bgval

    dest0 = embed_to(bg, img.copy())
    # TODO: We have problems with complex numbers
    # that are not supported by zoom(), rotate() or shift()
    if scale != 1.0:
        dest0 = ndii.zoom(dest0, scale, order=order, mode=mode, cval=bgval)
    if angle != 0.0:
        dest0 = ndii.rotate(dest0, angle, order=order, mode=mode, cval=bgval)

    if tvec[0] != 0 or tvec[1] != 0:
        dest0 = ndii.shift(dest0, tvec, order=order, mode=mode, cval=bgval)

    bg = np.zeros_like(img) + bgval
    dest = embed_to(bg, dest0)
    return dest


def _get_odds(angle, target, stdev):
    """
    Determine whether we are more likely to choose the angle, or angle + 180°

    Args:
        angle (float, degrees): The base angle.
        target (float, degrees): The angle we think is the right one.
            Typically, we take this from constraints.
        stdev (float, degrees): The relevance of the target value.
            Also typically taken from constraints.

    Return:
        float: The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
    """
    ret = 1
    if stdev is not None:
        diffs = [
            abs(wrap_angle(ang, 360)) for ang in (target - angle, target - angle + 180)
        ]
        odds0, odds1 = 0, 0
        if stdev > 0:
            odds0, odds1 = [np.exp(-(diff**2) / stdev**2) for diff in diffs]
        if odds0 == 0 and odds1 > 0:
            # -1 is treated as infinity in _phase_correlation
            ret = -1
        elif stdev == 0 or (odds0 == 0 and odds1 == 0):
            ret = -1
            if diffs[0] < diffs[1]:
                ret = 0
        else:
            ret = odds1 / odds0
    return ret


def argmax_translation(array, filter_pcorr, constraints=None):
    if constraints is None:
        constraints = dict(tx=(0, None), ty=(0, None))

    # We want to keep the original and here is obvious that
    # it won't get changed inadvertently
    array_orig = array.copy()
    if filter_pcorr > 0:
        array = ndi.minimum_filter(array, filter_pcorr)

    ashape = np.array(array.shape, int)
    mask = np.ones(ashape, float)
    # first goes Y, then X
    for dim, key in enumerate(("ty", "tx")):
        if constraints.get(key, (0, None))[1] is None:
            continue
        pos, sigma = constraints[key]
        alen = ashape[dim]
        dom = np.linspace(-alen // 2, -alen // 2 + alen, alen, False)
        if sigma == 0:
            # generate a binary array closest to the position
            idx = np.argmin(np.abs(dom - pos))
            vals = np.zeros(dom.size)
            vals[idx] = 1.0
        else:
            vals = np.exp(-((dom - pos) ** 2) / sigma**2)
        if dim == 0:
            mask *= vals[:, np.newaxis]
        else:
            mask *= vals[np.newaxis, :]

    array *= mask

    # WE ARE FFTSHIFTED already.
    # ban translations that are too big
    aporad = (ashape // 6).min()
    mask2 = get_apofield(ashape, aporad)
    array *= mask2
    # Find what we look for
    tvec = _argmax_ext(array, "inf")
    tvec = _interpolate(array_orig, tvec)

    # If we use constraints or min filter,
    # array_orig[tvec] may not be the maximum
    success = _get_success(array_orig, tuple(tvec), 2)

    return tvec, success


def translation(im0, im1, filter_pcorr=0, odds=1, constraints=None):
    """
    Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of the minimum spectrum filter
            for translation detection, use the filter when detection fails.
            Values > 3 are likely not useful.
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
            The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns:
        dict: Contains following keys: ``angle``, ``tvec`` (Y, X),
            and ``success``.
    """
    angle = 0
    # We estimate translation for the original image...
    tvec, succ = _phase_correlation(
        im0, im1, argmax_translation, filter_pcorr, constraints
    )
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    ret = np.rot90(im1, 2)  # Rotate the input array over 180°
    tvec2, succ2 = _phase_correlation(
        im0, im1, argmax_translation, filter_pcorr, constraints
    )

    pick_rotated = False
    if succ2 * odds > succ or odds == -1:
        pick_rotated = True

    if pick_rotated:
        tvec = tvec2
        succ = succ2
        angle += 180

    ret = dict(tvec=tvec, success=succ, angle=angle)
    return ret


def _get_precision(shape, scale=1):
    """
    Given the parameters of the log-polar transform, get width of the interval
    where the correct values are.

    Args:
        shape (tuple): Shape of images
        scale (float): The scale difference (precision varies)
    """
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # * 0.25 <= we got subpixel precision now and 0.5 / 2 == 0.25
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.25
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.25
    return Dangle, Dscale


# Compute the similarity (rotation, scale, translation) offset between two images
def compute_similarity(
    im0,
    im1,
    numiter,
    order,
    constraints,
    filter_pcorr,
    exponent,
):
    """
    Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    """
    bgval = get_borderval(im1, 5)
    # TODO: Most time spent here
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif im0.ndim != 2:
        raise ValueError("Images must be 2-dimensional.")

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    constraints_default = dict(angle=[0, None], scale=[1, None])
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default.update(constraints)
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    # TODO: Most time is spent here
    # Register image for scale/rotation
    for ii in range(numiter):
        newscale, newangle = _get_ang_scale([im0, im2], exponent, constraints_dynamic)
        scale *= newscale
        angle += newangle

        constraints_dynamic["scale"][0] /= newscale
        constraints_dynamic["angle"][0] -= newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

    # Here we look how is the turn-180
    target, stdev = constraints.get("angle", (0, None))
    odds = _get_odds(angle, target, stdev)

    # TODO: Small time is spent here (might be insignificant)
    # now we can use pcorr to guess the translation
    res = translation(im0, im2, filter_pcorr, odds, constraints)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = wrap_angle(angle, 360)

    # don't know what it does, but it alters the scale a little bit
    # scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)
    Dangle, Dscale = _get_precision(im0.shape, scale)

    res["scale"] = scale
    res["Dscale"] = Dscale
    res["Dangle"] = Dangle
    # 0.25 because we go subpixel now
    res["Dt"] = 0.25

    return res, bgval


def transform_img_dict(img, tdict, bgval=None, order=1, invert=False):
    """
    Wrapper of :func:`transform_img`, works well with the :func:`similarity`
    output.

    Args:
        img
        tdict (dictionary): Transformation dictionary --- supposed to contain
            keys "scale", "angle" and "tvec"
        bgval
        order
        invert (bool): Whether to perform inverse transformation --- doesn't
            work very well with the translation.

    Returns:
        np.ndarray: .. seealso:: :func:`transform_img`
    """
    scale = tdict["scale"]
    angle = tdict["angle"]
    tvec = np.array(tdict["tvec"])
    if invert:
        scale = 1.0 / scale
        angle *= -1
        tvec *= -1
    res = transform_img(img, scale, angle, tvec, bgval=bgval, order=order)
    return res


# TODO: Refactor this function, the current shape looks covoluted.
def frame_img(img, mask, dst, apofield=None):
    """
    Given an array, a mask (floats between 0 and 1), and a distance,
    alter the area where the mask is low (and roughly within dst from the edge)
    so it blends well with the area where the mask is high.
    The purpose of this is removal of spurious frequencies in the image's
    Fourier spectrum.

    Args:
        img (np.array): What we want to alter
        maski (np.array): The indicator what can be altered (0)
            and what can not (1)
        dst (int): Parameter controlling behavior near edges, value could be
            probably deduced from the mask.
    """
    radius = dst / 1.8

    convmask0 = mask + 1e-10

    krad_max = radius * 6
    convimg = img
    convmask = convmask0
    convimg0 = img
    krad0 = 0.8
    krad = krad0

    while krad < krad_max:
        convimg = ndi.gaussian_filter(convimg0 * convmask0, krad, mode="wrap")
        convmask = ndi.gaussian_filter(convmask0, krad, mode="wrap")
        convimg /= convmask

        convimg = convimg * (convmask - convmask0) + convimg0 * (
            1 - convmask + convmask0
        )
        krad *= 1.8

        convimg0 = convimg
        convmask0 = convmask

    if apofield is not None:
        ret = convimg * (1 - apofield) + img * apofield
    else:
        ret = convimg
        ret[mask >= 1] = img[mask >= 1]

    return ret


# Resize image's largest axis, by a division_factor (ratio is kept)
def resize_image(im, division_factor):
    # Use largest axis for resizing image
    largest_axis = 0
    if im.shape[1] > im.shape[0]:
        largest_axis = 1

    multiplication_factor = (
        int(im.shape[largest_axis] / division_factor) / im.shape[largest_axis]
    )

    return cv2.resize(
        im,
        (
            math.floor(im.shape[1] * multiplication_factor),
            math.floor(im.shape[0] * multiplication_factor),
        ),
    )


class im_reg:
    # Register im1 to im0 for Translation only
    def register_image_translation(self, im0, im1, scale_factor):
        translation_result = translation(
            resize_image(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY), scale_factor),
            resize_image(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), scale_factor),
        )

        height, width = im1.shape[:2]
        # Upscale offset, as it was calculated on a smaller image
        y_shift, x_shift = translation_result["tvec"] * scale_factor

        translation_matrix = np.float64(
            [
                [1, 0, x_shift],
                [0, 1, y_shift],
            ]
        )
        result = cv2.warpAffine(im1, translation_matrix, (width, height))

        return result.astype(np.uint8)

    # TODO: Implement with option of selecting what method to use (not yet successfully implemented)
    # Register im1 to im0 for Rotation, Scale, Translation (RST)
    def register_image_RST(
        self,
        im0,
        im1,
        scale_factor,
        numiter=5,
        order=3,
        constraints=None,
        filter_pcorr=0,
        exponent="inf",
    ):
        res, bgval = compute_similarity(
            resize_image(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY), scale_factor),
            resize_image(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), scale_factor),
            numiter,
            order,
            constraints,
            filter_pcorr,
            exponent,
        )
        print(res)

        scale = res["scale"]
        rot = res["angle"] * np.pi / 180  # Convert from degrees to radians
        y_shift, x_shift = res["tvec"] * scale_factor
        height, width = im1.shape[:2]

        # Src: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

        # Move image (translation)
        translation = np.float64(
            [
                [1, 0, x_shift],
                [0, 1, y_shift],
            ]
        )
        # Scale/rotate around center of image
        rotation = np.float64(
            [
                [np.cos(rot), np.sin(rot), 0],
                [-np.sin(rot), np.cos(rot), 0],
            ]
        )

        result = cv2.resize(im1, None, fx=scale, fy=scale)
        result = cv2.warpAffine(result, rotation, (width, height))
        result = cv2.warpAffine(result, translation, (width, height))

        return result.astype(np.uint8)
