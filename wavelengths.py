import numpy as np
import matplotlib.pyplot as plt
from astroquery.nist import Nist
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter

linfitter = LinearLSQFitter()
wlmodel = Linear1D()

def initial_wl(calibration_spectra, xlims, bad_pixel_mask, guess_pixels, guess_wl, npix_cutout, manual_scale=1.01):

  """
  This function is meant to help us get an initial feel for the wavelength solution for a calibration spectra when given a set of guessed pixels of lines and associated wavelengths of those lines



  Inputs (6, 7 if needed):
        calibration_spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was made using spectra.spectra_producer)
        
        xlims: (tuple) a set of xmin/xmax values to create an xaxis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)
        
        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image (can be found using spectra.point_finder)
        
        guess_wl: (list) the corresponding wavelengths that line up with the integer pixel values from the guess_pixels list
        
        npix_cutout: (int) a number of x-axis values to +/- cut around when measuring and adjusting guessed x-axis values

        manual_scale: (float) a number used to scale the points representing "guesses" on the graphs produced by this function (default is 1.01)
  Returns (1):
        wavelengths: (np.ndarray) a completed wavelength solution that considers only the guessed lines
        
  """
  consideration = np.median(calibration_spectra)**manual_scale
  xaxis = np.arange(xlims[0], xlims[1], step=1)
  improved_xval_guesses = [np.average(xaxis[~bad_pixel_mask][g-npix_cutout:g+npix_cutout],
                                          weights=calibration_spectra[g-npix_cutout:g+npix_cutout])
                               for g in guess_pixels]
  
  linear_fit_wlmodel = linfitter(model=wlmodel, x=improved_xval_guesses, y=guess_wl)
  wavelengths = linear_fit_wlmodel(xaxis[~bad_pixel_mask]) * u.AA
  fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(12,4))
  ax1.plot(wavelengths, calibration_spectra)
  ax1.plot(guess_wl, [consideration]*len(guess_wl), 'x', label="Guessed Values");
  ax1.set_xlabel("Wavelength($\AA$)")
  ax1.set_ylabel("Intensity")
  ax1.legend()

  ax2.plot(guess_pixels, guess_wl, 'o')
  ax2.plot(xaxis[~bad_pixel_mask], wavelengths, '-')
  ax2.set_ylabel("$\lambda(x)$")
  ax2.set_xlabel("x (pixels)")
  return wavelengths


def solution_fitter():
  minwave = wavelengths_alt.min()
  maxwave = wavelengths_alt.max()
  neon_lines = Nist.query(minwav=minwave,
                        maxwav=maxwave,
                        wavelength_type='vac+air',
                        linename='Ne I')
