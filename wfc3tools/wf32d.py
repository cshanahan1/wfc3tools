# get the auto update version for the call to teal help
from .version import __version_date__, __version__

# STDLIB
import os.path
import subprocess

# STSCI
from stsci.tools import parseinput
from .util import error_code

__taskname__ = "wf32d"


def wf32d(input, output=None, verbose=False, quiet=True, debug=False, 
          log_func=print):

    """
    Runs the calwf3 calibration subtask wf32d on a single input WFC3 UVIS
    image, which must be overscanned-trimmed (blv_tmp or crj_tmp).

    This routine contains the secondary steps for all the WFC3 UVIS
    channel data. These steps are:

      * DARKCORR: dark current subtraction
      * FLATCORR: flat-fielding
      * PHOTCORR: photometric keyword calculations
      * FLUXCORR: photometric normalization of the UVIS1 and UVIS2 chips

    wf32d processing is controlled by the values of keywords in the input
    image headers. Certain keywords, referred to as calibration switches, are
    used to control which calibration steps are performed. Reference file
    keywords indicate which reference files to use in the various calibration
    steps. Users who wish to perform custom reprocessing of their data may
    change the values of these keywords in the FITS file primary headers
    and then rerun the modified file through wf32d. See the WFC3 Data Handbook
    for a more complete description of these keywords and their values.

    Parameters
    ----------
    input : str
        Overscan-trimmed image (blv_tmp or crj_tmp).
    output : str
        Desired output file name. If not specified, will be the rootname of the
        input file appended with '_blv_tmp.fits'.
    verbose : bool, optional
        Print verbose time stamps.
    quiet : bool, optional
        Print messages only to trailer file.
    log_func: func()
        If not specified, the print function is used for logging to facilitate
        use in the Jupyter notebook. If None, no output will be printed or 
        passed to the logging function.

    Outputs
    -------
    <filename>_flt.fits : FITS file
        Calibrated UVIS image (e-).
    """


    call_list = ['wf32d.e']
    return_code = None

    if verbose:
        call_list += ['-v', '-t']

    if debug:
        call_list.append('-d')

    infiles, dummy = parseinput.parseinput(input)
    if "_asn" in input:
        raise IOError("wf32d does not accept association tables")
    if len(parseinput.irafglob(input)) == 0:
        raise IOError("No valid image specified")
    if len(parseinput.irafglob(input)) > 1:
        raise IOError("wf32d can only accept 1 file for"
                      "input at a time: {0}".format(infiles))

    for image in infiles:
        if not os.path.exists(image):
            raise IOError("Input file not found: {0}".format(image))

    call_list.append(input)

    if output:
        call_list.append(str(output))

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
        raise RuntimeError("wf32d.e exited with code {}".format(ec))
