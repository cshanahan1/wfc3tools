"""
This module contains tools to estimate and subtract a three-component
(zodiacal, HeI from earth, and scattered light) background model from WFC3 IR
grism (G102 or G141) data obtained in the same visit, or a single exposure.

There are three main external contributors to background in a WFC3 IR exposure:
zodiacal light, HeI from earth, and scattered light. Over the duration of a
single visit, i.e while HST is pointing at a specific target on the sky and
within a relatively short amount of time, the zodi and back-scattered light
from the sun do not vary. Unfortunately, we also have two other contributors to
light that enters HST that are variable: the HeI glow from the upper atmosphere
of the Earth (which is prevalent when HST points close to the Earth’s Limb)
as well as scattered light component (which has been shown to exist but whose
origin remains complicated and not fully determined).

The code takes a list of `raw.fits` observations and starts the process of
calibrating the data using `calwf3`, up to the creation of the `ima.fits` files
and omitting the final on-the-ramp fitting (CRCORR=OMIT). The latter cannot be
used when the signal varies temporally and any varying background (combination
of HeI and scattered light) results in the on-the-ramp fitting to fail. We
detect the spectra in each observation and mask each IMSET of the IMA files. We
then set up a system of equations where each pixel (which has not been masked)
is combination  of zodiacal, HeI, and scattered light. These three components
have different 2D structures and hence the contribution of each of them can be
estimated. We allow for the HeI and scattered light component to be varying
freely in each individual IMSET while the zodiacal light level is kept
constant. The number of unknown variables to solve for is therefore n*m*2 + n
if we have n observations, each with m IMSETS (the number of IMSETS needs not
being the same in all observations though). Even while heavily masked, each
IMSET contributes several thousands measurements so the system can be solved.
The code estimates the HeI and scattered light in each IMSET of each
observation, and those components are subtracted from the `ima.fits` file. We
then run `calwf3` again, this time starting with the `ima.fits` file to finish
the process and run the on-the-ramp-fitting. The final step is that of
re-estimating the zodiacal light level. While we already estimated that in the
previous step, the FLT files have a higher signal-to-noise and we instead again
use the mask we used to mask out the spectra and scale and subtract a single
zodiacal light level from all of the FLT files at the same time.

Use
---
    To run the full background estimation and subtraction process on a set of
    `raw.fits` files obtained during the same spectral element during the same
    visit:
    ::
        from wfc3tools import grism_back_sub

        # Initialize the SubBack object with path to data and obs id(s), and
        # path to directory with custom grism reference files (not $iref).
        sub = grism_back_sub.SubBack('/path/to/data/',
                                    ['icoi3na5q', 'icoi3na1q'],
                                    '/path/to/grism/ref/dir/')

        # Run the full background estimation and subtraction method
        sub.calc_and_subtract_background().

    To calculate zodi, scattered light, and HeI without subtracting these
    background values
    ::
        sub.calc_heI_zodi_scatter()

        # Inspect the calculated components. scatter + HeI have a value for
        # each read of each IMA, zodi has one value for the whole set.

        sub.zodi
        sub.heIs
        sub.scatter

    To generate and save diagnostic plot:
    ::
        sub.diagnostic_plot(save_fig=True)
Note
-----
    Before running any of the tools, make sure you have the necessary custom
    reference files downloaded into a directory that is seperate from the main
    $iref reference file directory. See `download_grism_ref_files`.

"""
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian1D
import glob
import numpy as np
import os
from photutils.background import Background2D
from photutils import detect_sources
from scipy import optimize
import shutil
import sys
import requests

from . import calwf3, wf3ir


def download_grism_ref_files(grism_ref_file_dir=''):

    import requests

    url = 'http://www.stsci.edu/~WFC3/ir-back-sub-files/'
    files = [f'{x}_{y}_v9.fits' for x in ['G102', 'G141'] for y in
             ['scatter', 'zodi', 'heI']]
    files += ['uc72113oi_pfl_patched2.fits', 'uc721143i_pfl_patched2.fits']

    for f in files:
        print(f'Downloading {os.path.join(grism_ref_file_dir, f)}.')
        r = requests.get(url+f, allow_redirects=True)
        open(os.path.join(grism_ref_file_dir, f), 'wb').write(r.content)


class SubBack():

    """
    Class containing methods for calculating time-variable background in WFC3
    IR Grism data.

    These tools rely upon a set of grism-specific reference files.
    `grism_ref_file_dir` should point to the directory where these files are
    located. All of the necessary files are hosted at
    http://www.stsci.edu/~WFC3/ir-back-sub-files. See
    `download_grism_ref_files`.

    Parameters
    ----------
    data_dir : str
        Path to directory where input files are located.
    obs_ids : str or list of str
        9-digit IPPPSSOOT ID(s) of raw files in `data_dir` that will be
        processed.

    Attributes
    ----------
    data_dir
    obs_ids
    raw_names : list of str
        List of rootnames (i.e <obs_id>_raw.fits) of input files.
    raw_paths : list of str
        List of full paths to all input raw files.
    grism : {'G102', 'G140'}
        Single grism, common to all input files.
    zodi_file : str
        Name of zodiacal background file.
    heI_file : str
        Name of HeI background file.
    scatter_file : str
        Name of scattered light background file.
    ff_file : str
        Name of custom flat field file to be used for flat field correction.
    zodi : float
        Zodi level fit to entire set. Set by calc_heI_zodi_scatter().
    heIs : dict
        Fit HeI levels for each read of each input file.
    scatter : dict
        Fit scattered light levels for each read of each input file.

    Methods
    --------
    calc_and_subtract_background(calwf3_verbose=False)
        Runs the full three-component (zodi, HeI, scattered light) background
        model calculation and subtraction on input `raw.fits` file(s).
    calc_heI_zodi_scatter()
        Calculates three-component (zodi, HeI, scattered light) background for
        input `raw.fits` file(s). Does not subtract background.
    diagnostic_plot(save_fig=False)
        Function to output diagnostic plots for each of the processed
        observations, plotting the median residuals in the final background
        subtracted FLT files (after applying the detection mask).
    """

    def __init__(self, data_dir, obs_ids, grism_ref_file_dir):

        self.data_dir = data_dir
        self.grism_ref_file_dir = grism_ref_file_dir

        if type(obs_ids) == 'str':
            obs_ids = [obs_ids]

        self.obs_ids = obs_ids

        self.raw_names = [f"{obs_id}_raw.fits" for obs_id in obs_ids]
        self.raw_paths = [os.path.join(data_dir, x) for x in self.raw_names]

        # get grism name, make sure it is common to all files
        grisms = [fits.getval(x, 'FILTER') for x in self.raw_paths]
        if len(set(grisms)) != 1:
            raise ValueError('Input files must use the same grism.')

        self.grism = grisms[0]

        self.zodi_file = f'{self.grism}_zodi_v9.fits'
        self.heI_file = f'{self.grism}_heI_v9.fits'
        self.scatter_file = f'{self.grism}_scatter_v9.fits'
        if self.grism == 'G102':
            self.ff_file = 'uc72113oi_pfl_patched2.fits'
        if self.grism == 'G140':
            self.ff_file = 'uc721143i_pfl_patched2.fits'

        self.__check_if_grism_ref_files_exist()
        self.orig_ff_files = self.__get_orig_ff_files()

        # initialize other attributes that are set by methods
        self.msk_paths = None
        self.flt_paths = None
        self.ima_paths = None
        self.zodi = None
        self.heIs = None
        self.scats = None

    def __check_if_grism_ref_files_exist(self):
        """ Checks if grism reference files have been downloaded to
            `grism_ref_file_dir`.
        """
        for x in [self.zodi_file, self.heI_file, self.scatter_file,
                  self.ff_file]:
            if not os.path.isfile(os.path.join(self.grism_ref_file_dir, x)):
                raise FileNotFoundError(f'Grism reference file {x} not in' +
                                        f'{self.grism_ref_file_dir}.' +
                                        'To download all reference files, '
                                        'see download_grism_ref_files')

    def __copy_flat_field_to_data_dir(self):

        """ Reference files are included with wfc3tools, but due to character
            limits need to be along side the data. This method copies the files
            from the wfc3tools directory to `dir`.
        """

        if not os.path.isfile(os.path.join(self.data_dir, self.ff_file)):
            print(f'Copying {self.ff_file} to data directory.')
            shutil.copy(os.path.join(self.grism_ref_file_dir, self.ff_file),
                        self.data_dir)

    def __get_orig_ff_files(self):
        """ Returns list of original flat field files in input data."""

        orig_ff_files = []
        for f in self.raw_paths:
            orig_ff_files.append(fits.getval(f, "PFLTFILE"))

        return orig_ff_files

    def calc_and_subtract_background(self, calwf3_verbose=False):
        """
        Main function to perform all the required steps to remove the time
        varying HeI and Scattered light component as well as the Zodi component
        from WFC3 IR G102 or G141 grism `raw.fits` file(s).

        Input files should be `raw.fits` files obtained in the same visit with
        the same grism (or a single `raw.fits` file).

        Parameters
        ----------
        calwf3_verbose : bool, optional
            If True, output from calwf3/wf3ir will be printed. If False, no
            output will be printed. Defaults to False.
        """

        self.__copy_flat_field_to_data_dir()

        # set calwf3 output behavior
        if calwf3_verbose is True:
            self._log_func = print
        if calwf3_verbose is False:
            self._log_func = None

        self.flt_paths = [self.__raw_to_flt(x) for x in self.raw_paths]
        self.ima_paths = [x.replace('raw.fits', 'ima.fits') for x in
                          self.raw_paths]
        self.msk_paths = [self.__create_msk(x) for x in self.flt_paths]

        self.calc_heI_zodi_scatter()

        self.__sub_HeI_Scat()

        self.flt_paths = [self.__ima_to_flt(x.replace('raw.fits', 'ima.fits'))
                          for x in self.raw_paths]
        self.__sub_Zodi()

    def __raw_to_flt(self, raw_path, tmp=False):
        """
        Function to run CALWF3 on a raw dataset. CRCORR is set to OMIT and
        FLATCORR is set to perform. The flat-field calibration file names are
        replaced with the ones included in this package and pointed to by
        `ff_file`.

        Parameters
        ----------
        raw_path : str
            Path to raw input file.

        Outputs
        -------
        <filename>_flt.fits, <filename>_ima.fits <filename>.tra: FITS file
            `calwf3` pipeline output files.

        """

        # change to dir w/ data due to character limits. remember original dir
        # to change back to after processing.
        orig_dir = os.getcwd()
        os.chdir(self.data_dir)

        raw_name = os.path.basename(raw_path)
        if tmp:
            shutil.copy(raw_name, 'temp_'+raw_name)
            raw_name = 'temp_'+raw_name

        # delete previous calwf3 output files
        calwf3_output_files = [raw_name.replace('raw.fits', f'{x}.fits')
                               for x in ['ima', 'flt']]

        for ff in calwf3_output_files:
            if os.path.isfile(ff):
                print(f'Removing existing output {ff}')
                os.remove(ff)
        print(f"Initial processing of {raw_name} with calwf3." +
              f"CRCORR = OMIT, PFLTFILE = {self.ff_file}")

        # set calibration switches and path to new PFLTFILE
        fin = fits.open(raw_path, mode="update")
        fin[0].header["CRCORR"] = "OMIT"
        fin[0].header["FLATCORR"] = "PERFORM"
        fin[0].header["PFLTFILE"] = self.ff_file
        fin.close()

        calwf3(raw_name, log_func=self._log_func)

        flt_name = raw_name.replace('raw.fits', 'flt.fits')

        # check if output flt exists
        if not os.path.isfile(flt_name):
            print("raw_to_flt() failed to generate ", flt_name)
            sys.exit(1)

        os.chdir(orig_dir)  # back to original directory

        return os.path.join(self.data_dir, flt_name)

    def __create_msk(self, flt_path, tmp=False):
        """
        Creates an image mask. The size of the detection kernel (1.25 pix),
        background_box (1014//6,2), threshold (0.05), and number of connected
        pixels (80) are fixed.

        Returns
        _______
        msk_name : str
            Path to the `msk.fits` file.
        """

        # parameters for mask creation
        # Nor found this particular combination of parameters works best, and
        # it is a better idea to keep them fixed.
        self._kernel_fwhm = 1.25
        self._background_box = (1014 // 6, 2)
        self._thr = 0.05
        self._npixels = 80
        # every flag except blob
        self._bit_mask = (7679)

        fin = fits.open(flt_path)
        image = fin["SCI"].data
        err = fin["ERR"].data
        dq = fin["DQ"].data
        dq = np.bitwise_and(dq, np.zeros(np.shape(dq), np.int16)
                            + self._bit_mask)

        g = Gaussian1D(mean=0., stddev=self._kernel_fwhm/2.35)
        x = np.arange(16.)-8
        a = g(x)
        kernel = np.tile(a, (16*int(self._kernel_fwhm + 1), 1)).T
        kernel = kernel/np.sum(kernel)

        b = Background2D(image, self._background_box)

        image = image-b.background
        threshold = self._thr * err

        image[dq > 0] = 0.  # np.nan

        mask = detect_sources(image, threshold, npixels=self._npixels,
                              filter_kernel=kernel).data
        ok = (mask == 0.) & (dq == 0)
        mask[~ok] = 1.

        segm = mask

        dq3 = fits.open(flt_path)["DQ"].data.astype(int)

        DQ = np.bitwise_and(dq3, np.zeros(np.shape(dq3), np.int) +
                            self._bit_mask)

        kernel = Gaussian2DKernel(x_stddev=1)
        segm = segm
        segm = convolve(segm, kernel)
        segm[segm > 1e-5] = 1.
        segm[segm <= 1e-5] = 0.
        segm[DQ > 0] = 1.

        msk_name = flt_path.replace('flt.fits', 'msk.fits')

        print(f'Writing {msk_name}')
        fits.writeto(msk_name, segm, overwrite=True)

        return msk_name

    def __get_mask(self, flt_path):
        """
        Function to create a mask (set to 0 for no detection and 1 for
        detection) appropriate to mask WFC3 slitless data. Generates a
        segmentation map to create an image mask.

        Parameters
        ----------
        flt_path : str
            Path to input `flt.fits` file.

        """
        fin = fits.open(flt_path)
        image = fin["SCI"].data
        err = fin["ERR"].data
        dq = fin["DQ"].data
        dq = np.bitwise_and(dq, np.zeros(np.shape(dq), np.int16)
                            + self._bit_mask)

        g = Gaussian1D(mean=0., stddev=self._kernel_fwhm/2.35)
        x = np.arange(16.) - 8
        a = g(x)
        kernel = np.tile(a, (16*int(self._kernel_fwhm+1), 1)).T
        kernel = kernel/np.sum(kernel)

        b = Background2D(image, self._background_box)

        image = image-b.background
        threshold = self._thr * err

        image[dq > 0] = 0.  # np.nan

        mask = detect_sources(image, threshold, npixels=self._npixels,
                              filter_kernel=kernel).data
        ok = (mask == 0.) & (dq == 0)
        mask[~ok] = 1.

        return mask

    def __ima_to_flt(self, ima_path):
        """
        Function to run CALWF3 on an exisiting IMA file. CRCORR is set to
        PERFORM.

        Attributes
        ----------
        ima_name string containing the name of the IMA file to process

        Output
        ------
        string containing the name of the FLT file that has been created
        """

        print(f"\nRunning wf3ir on {os.path.basename(ima_path)}")

        # change to dir w/ data due to character limits. remember original
        # to change back to after processing.
        orig_dir = os.getcwd()
        os.chdir(self.data_dir)

        ima_name = os.path.basename(ima_path)

        fin = fits.open(ima_name, mode="update")
        fin[0].header["CRCORR"] = "PERFORM"
        fin.close()

        obs_id = ima_name.split("_ima.fits")[0]
        flt_name = f"{obs_id}_flt.fits"

        if os.path.isfile(flt_name):
            os.remove(flt_name)
        tmp_name = f"{obs_id}_ima_ima.fits"
        if os.path.isfile(tmp_name):
            os.remove(tmp_name)
        tmp_name = f"{obs_id}_ima_flt.fits"
        if os.path.isfile(tmp_name):
            os.remove(tmp_name)

        wf3ir(ima_name, log_func=self._log_func)

        shutil.move(tmp_name, flt_name)
        tmp_name = f"{obs_id}_ima_ima.fits"
        if os.path.isfile(tmp_name):
            os.remove(tmp_name)

        os.chdir(orig_dir)  # back to original directory

        return os.path.join(self.data_dir, flt_name)

    def calc_heI_zodi_scatter(self, calwf3_verbose=False):
        """
        Function to estimate the Zodi, HeI, and Scatter levels in each IMSET of
        an IMA file. A set of IMA files can be processed at once and the Zodi
        level is assumed to be identical in all of them. The HeI and Scatter
        levels are allowed to vary freely. See code by R. Ryan in Appendix of
        ISR WFC3 2015-17 for details.
        """
        self.__copy_flat_field_to_data_dir()
        print('Calculating HeI, Zodi, and Scatter.')
        tmp = False

        # work in 'temp' mode if this is being run alone to calculate
        # background levels without subtracting. Because the calculation
        # requires generating output files, this ensures no existing output
        # from previous subtraction runs is deleted,
        if self.msk_paths is None:  # generate flts/masks if not done.
            if calwf3_verbose is True:
                self._log_func = print
            else:
                self._log_func = None
            flt_paths = [self.__raw_to_flt(x, tmp=True) for x in
                         self.raw_paths]
            msk_paths = [self.__create_msk(x, tmp=True) for x in flt_paths]
            tmp = True  # work with temp files to not overwrite
            print("Temp files will be generated to calculate background.")
        else:
            msk_paths = self.msk_paths

        border = 0  # num of columns to avoid on the left and right hand side

        ima_paths = [x.replace('raw.fits', 'ima.fits') for x in self.raw_paths]
        if tmp:
            ima_paths = [x.replace(os.path.basename(x),
                         'temp_'+os.path.basename(x)) for x in ima_paths]

        nimas = len(ima_paths)

        # We drop the last ext/1st read
        nexts = [fits.open(ima)[-1].header["EXTVER"] for ima in ima_paths]

        sl = slice(5, 1014+5)  # slice for data (same for x, y)

        # load background files
        zodi = fits.open(os.path.join(self.grism_ref_file_dir,
                         self.zodi_file))[1].data
        heI = fits.open(os.path.join(self.grism_ref_file_dir,
                        self.heI_file))[1].data
        scatter = fits.open(os.path.join(self.grism_ref_file_dir,
                            self.scatter_file))[1].data

        data0s = []
        err0s = []
        samp0s = []
        dq0s = []
        dqm0s = []
        masks = []
        for j, _ in enumerate(ima_paths):
            obs_id = ima_paths[j][0:9]
            mask = fits.open(msk_paths[j])[0].data
            masks.append([mask for ext in range(1, nexts[j])])
            hdu = fits.open(ima_paths[j])
            data0s.append([hdu["SCI", ext].data[sl, sl] for ext in
                          range(1, nexts[j])])
            err0s.append([hdu["ERR", ext].data[sl, sl] for ext in
                          range(1, nexts[j])])
            dq0s.append([hdu["DQ", ext].data[sl, sl] for ext in
                        range(1, nexts[j])])

        dqm0s = [[np.bitwise_and(dq0, np.zeros(np.shape(dq0), np.int16)
                  + self._bit_mask) for dq0 in dq0s[j]] for j in range(nimas)]

        ok = (np.isfinite(zodi)) & (np.isfinite(heI)) & (np.isfinite(scatter))
        zodi[~ok] = 0.
        heI[~ok] = 0.
        scatter[~ok] = 0.

        # Setting up image weights
        whts = []
        for j, _ in enumerate(ima_paths):
            whts_j = []
            for i in range(len(err0s[j])):
                err = err0s[j][i]
                err[err <= 1e-6] = 1e-6
                w = 1./err**2
                w[~ok] = 0.
                whts_j.append(w)
            whts.append(whts_j)

        nflt = sum(nexts)
        npar = 2*nflt+1
        print("\nSolving for Zodi and HeI, Scatter values.")
        v = np.zeros(npar, np.float)
        m = np.zeros([npar, npar], np.float)
        ii = -1
        for j, _ in enumerate(ima_paths):
            whts[j] = np.array(whts[j])
            data0s[j] = np.array(data0s[j])
            masks[j] = np.array(masks[j])
            dqm0s[j] = np.array(dqm0s[j])
            whts[j][~np.isfinite(data0s[j])] = 0.
            data0s[j][~np.isfinite(data0s[j])] = 0.
            whts[j][masks[j] > 0] = 0.
            whts[j][dqm0s[j] != 0] = 0.

            for i in range(len(data0s[j])):
                ii = ii + 1

                img = data0s[j][i]
                wht = whts[j][i]

                if border > 0:
                    wht[0:border] = 0.
                    wht[-border:0] = 0.

                # Populate up matrix and vector
                v[ii] = np.sum(wht*data0s[j][i]*heI)
                v[-1] += np.sum(wht*data0s[j][i]*zodi)
                m[ii, ii] = np.sum(wht*heI*heI)
                m[ii, -1] = np.sum(wht*heI*zodi)
                m[-1, ii] = m[ii, -1]
                m[-1, -1] += np.sum(wht*zodi*zodi)
                v[ii+nflt] = np.sum(wht*data0s[j][i]*scatter)
                m[ii+nflt, ii+nflt] = np.sum(wht*scatter*scatter)
                m[ii, ii+nflt] = np.sum(wht*heI*scatter)
                m[ii+nflt, -1] = np.sum(wht*zodi*scatter)
                m[ii+nflt, ii] = m[ii, ii+nflt]
                m[-1, ii+nflt] = m[ii+nflt, -1]

        res = optimize.lsq_linear(m, v)
        x = res.x

        Zodi = x[-1]
        HeIs = {}
        Scats = {}

        ii = -1
        for j, _ in enumerate(data0s):
            HeIs[ima_paths[j]] = {}
            Scats[ima_paths[j]] = {}
            for i in range(len(data0s[j])):
                ii = ii + 1
                idd = os.path.basename(ima_paths[j])
                print(f"{idd} IMSET {i} He: {x[ii]} S: {x[ii+nflt]}")
                HeIs[ima_paths[j]][i+1] = x[ii]
                Scats[ima_paths[j]][i+1] = x[ii+nflt]
        print(f'Zodi, Scale: {Zodi}')

        self.zodi = Zodi
        self.heIs = HeIs
        self.scats = Scats

        if tmp:
            for obs in self.obs_ids:
                for f in glob.glob(os.path.join(self.data_dir,
                                                f'temp*{obs}*')):
                    print(f'Removing temp file {f}')
                    os.remove(f)

    def __sub_HeI_Scat(self):
        """
        Function to subtract the appropriately scaled HeI and Scatter light
        models from each of the IMSET of the `ima.fits` files included in the
        `heIs` and `scats` dictionaries. Header keywords are populated to
        reflect the amount of HeI and Scattered light subtracted.

        """
        scatter = fits.open(os.path.join(self.grism_ref_file_dir,
                            self.scatter_file))[1].data
        heI = fits.open(os.path.join(self.grism_ref_file_dir,
                        self.heI_file))[1].data

        for f in self.heIs.keys():

            print(f"\nSubtracting HeI and Scattered light from {f}")
            fin = fits.open(f, mode="update")
            sl = slice(5, 1014+5)  # slice for data (same for x, y)

            for ev in self.heIs[f].keys():

                dat = fin["SCI", ev].data[sl, sl]

                print(f"IMSET: {ev}. subtracting He: {self.heIs[f][ev]} S:" +
                      f"{self.scats[f][ev]}")
                fin["SCI", ev].data[sl, sl] = dat - self.heIs[f][ev]*heI \
                    - self.scats[f][ev]*scatter
                # update header
                hdr = fin["SCI", ev].header
                hdr[f"HeI_{ev}"] = (self.heIs[f][ev],
                                    "HeI level subtracted (e-)")
                hdr[f"Scat_{ev}"] = (self.scats[f][ev],
                                     "Scat level estimated (e-)")

            fin.close()

    def __sub_Zodi(self):
        """
        Function to subtract the Zodi component from an `flt.fits` file.
        Updates header of `flt.fits` file to add the keyword `Zodi` with the
        value that was subtracted.

        Attributes
        ----------
        flt_name String containing the name of the FLT file to process

        Output
        ------
        None

        """

        for i, flt_path in enumerate(self.flt_paths):
            print(f'\nSubtracting Zodi from {flt_path}')

            flt_name = os.path.basename(flt_path)
            fin = fits.open(flt_path, mode="update")
            zodi = fits.open(os.path.join(self.grism_ref_file_dir,
                             self.zodi_file))[1].data
            print("Zodi, Scale", self.zodi)
            fin["SCI"].data = fin["SCI"].data - zodi*self.zodi
            fin["SCI"].header["Zodi"] = (self.zodi, "Zodi level (e-)")
            fin.close(output_verify="ignore")

    def diagnostic_plot(self, save_fig=False):

        """
        Function to output diagnostic plots for each of the processed
        observations, plotting the median residuals in the final background
        subtracted FLT files (after applying the detection mask).

        Parameters
        ----------
        save_fig : bool, optional
            To save <filename>_diag.png in the same directory as the input
            data. Defaults to False.
        Outputs
        -------
        <filename>_diag.png : png file
            Diagnostic plot, in directory with input data.
            Only output if save_fig=True.
        """

        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (10, 2.5*len(self.obs_ids))
        fig = plt.figure()
        plt.clf()
        for i, obs in enumerate(self.obs_ids):
            plt.subplot(len(self.obs_ids), 1, i + 1)
            hdu = fits.open(self.flt_paths[i])
            d = hdu[1].data
            dq = hdu["DQ"].data
            dq = np.bitwise_and(dq, np.zeros(np.shape(dq), np.int16) +
                                self._bit_mask)
            m = fits.open(self.msk_paths[i])[0].data
            ok = (m == 0) & (np.isfinite(d))
            d[m > 0] = np.nan
            plt.plot(np.nanmedian(d, axis=0), label=obs)
            plt.grid()
            plt.ylabel("e-/s")
            plt.xlabel("col")
            plt.xlim(0, 1014)
            plt.ylim(-0.02, 0.02)
            plt.legend()
        oname = "{}_diag.png".format(self.obs_ids[0][0:6])
        oname = os.path.join(self.data_dir, oname)
        if save_fig:
            print(f'Saving {oname}')
            plt.savefig(oname)
