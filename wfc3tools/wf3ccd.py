from .version import __version_date__, __version__

# STDLIB
import os.path
import subprocess

# STSCI
from stsci.tools import parseinput
from .util import error_code


__taskname__ = "wf3ccd"


def wf3ccd(input, output=None, verbose=False, quiet=True, log_func=print):

    """
    Runs the calwf3 calibration subtask wf3ccd on a single input WFC3 UVIS
    image.

    This routine contains the initial processing steps for all the WFC3 UVIS
    channel data. These steps are:

    * DQICORR - initializing the data quality array
    * ATODCORR - perform the a to d conversion correction
    * BLEVCORR - subtract the bias level from the overscan region
    * BIASCORR - subtract the bias image
    * FLSHCORR - subtract the post-flash image

    wf3ccd processing is controlled by the values of keywords in the input
    image headers. Certain keywords, referred to as calibration switches, are
    used to control which calibration steps are performed. Reference file
    keywords indicate which reference files to use in the various calibration
    steps. Users who wish to perform custom reprocessing of their data may
    change the values of these keywords in the FITS file primary headers
    and then rerun the modified file through wf3ccd. See the WFC3 Data Handbook
    for a more complete description of these keywords and their values.

    Parameters
    ----------
    input : str
        Single UVIS raw file.
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
    <filename>_blv_tmp.fits : FITS file
        Overscan-trimmed UVIS exposure (DN).
    """

    call_list = ['wf3ccd.e']
    return_code = None

    if verbose:
        call_list += ['-v', '-t']

    infiles, dummy = parseinput.parseinput(input)
    if "_asn" in input:
        raise IOError("wf3ccd does not accept association tables")
    if len(parseinput.irafglob(input)) == 0:
        raise IOError("No valid image specified")
    if len(parseinput.irafglob(input)) > 1:
        raise IOError("wf3ccd can only accept 1 file for"
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
        raise RuntimeError("wf3ccd.e exited with code {}".format(ec))
