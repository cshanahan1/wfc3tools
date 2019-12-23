# STDLIB
import os.path
import subprocess

# get the auto update version for the call to teal help
from .version import __version_date__, __version__

# STSCI
from stsci.tools import parseinput
from .util import error_code

__taskname__ = "wf3rej"


def wf3rej(input, output="", crrejtab="", scalense="", initgues="",
           skysub="", crsigmas="", crradius=0, crthresh=0,
           badinpdq=0, crmask=False, shadcorr=False, verbose=False,
           log_func=print):
    """

    Runs the calwf3 calibration subtask wf3rej on a single input WFC3 UVIS or IR
    image.

    The wf3rej program is used for both UVIS and IR images to combine multiple 
    exposures contained in a CR-SPLIT or REPEAT-OBS. In the full UVIS 
    calibration pipeline, `wf3rej` comes after the initial processing step 
    `wf3ccd`, which outputs a `blv_tmp.fits' file. UVIS input to `wf3rej` must 
    have the overscan pixels trimmed. In the IR pipeline, `wf3rej` is 
    Parameters
    ----------
    input : str
        Name of input file.
    output : str
        Name of the output FITS file.
    crrejtab : string
        Reference file name
    scalense : str
        Scale factor applied to noise.
    initgues : str
        Intial value estimate scheme (min|med)
    skysub : str
        How to compute the sky (none|mode|mean).
    crsigmas : str
        Rejection levels in each iteration.
    crradius : float 
        Cosmic ray expansion radius in pixels.
    crthresh : float
        Rejection propagation threshold.
    badinpdq : int
        Data quality flag bits to reject.
    crmask :   bool
        Flag CR in input DQ images?
    shadcorr : bool
        Perform shading shutter correction?
    verbose : bool, optional
        Print verbose time stamps?
    """

    call_list = ["wf3rej.e"]
    return_code = None

    infiles, dummy = parseinput.parseinput(input)
    if "_asn" in input:
        raise IOError("wf3rej does not accept association tables")
    if len(parseinput.irafglob(input)) == 0:
        raise IOError("No valid image specified")
    if len(parseinput.irafglob(input)) > 1:
        raise IOError("wf3rej can only accept 1 file for"
                      "input at a time: {0}".format(infiles))

    for image in infiles:
        if not os.path.exists(image):
            raise IOError("Input file not found: {0}".format(image))

    call_list.append(input)

    if output:
        call_list.append(str(output))

    if verbose:
        call_list.append("-v")
        call_list.append("-t")

    if (shadcorr):
        call_list.append("-shadcorr")

    if (crmask):
        call_list.append("-crmask")

    if (crrejtab != ""):
        call_list += ["-table", crrejtab]

    if (scalense != ""):
        call_list += ["-scale", str(scalense)]

    if (initgues != ""):
        options = ["min", "med"]
        if initgues not in options:
            print("Invalid option for intigues")
            return ValueError
        else:
            call_list += ["-init", str(initgues)]

    if (skysub != ""):
        options = ["none", "mode", "median"]
        if skysub not in options:
            print(("Invalid skysub option: %s") % (skysub))
            print(options)
            return ValueError
        else:
            call_list += ["-sky", str(skysub)]

    if (crsigmas != ""):
        call_list += ["-sigmas", str(crsigmas)]

    if (crradius >= 0.):
        call_list += ["-radius", str(crradius)]
    else:
        print("Invalid crradius specified")
        return ValueError

    if (crthresh >= 0.):
        call_list += ["-thresh", str(crthresh)]
    else:
        print("Invalid crthresh specified")
        return ValueError

    if (badinpdq >= 0):
        call_list += ["-pdq", str(badinpdq)]

    else:
        print("Invalid DQ value specified")
        return ValueError

    proc = subprocess.Popen(
        call_list,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    if log_func is not None:
        for line in proc.stdout:
            log_func(line.decode('utf8'))

    return_code = proc.wait()
    ec = error_code(return_code)
    if return_code:
        if ec is None:
            print("Unknown return code found!")
            ec = return_code
        raise RuntimeError("wf3rej.e exited with code {}".format(ec))
