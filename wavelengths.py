import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
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
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)
        
        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image (can be found using spectra.point_finder)
        
        guess_wl: (list) the corresponding wavelengths that line up with the integer pixel values from the guess_pixels list
        
        npix_cutout: (int) a number of x-axis values to +/- cut around when measuring and adjusting guessed x-axis values

        manual_scale: (float) a number used to scale the points representing "guesses" on the graphs produced by this function (default is 1.01)
  Returns (2):
        wavelengths: (np.ndarray) a completed wavelength solution that considers only the guessed lines

        linear_fit_wlmodel: an initial wavelength solution that considers only the guessed lines
        
  """
  consideration = np.median(calibration_spectra)**manual_scale
  xaxis = np.arange(xlims[0], xlims[1], step=1)
  improved_xval_guesses = [np.average(xaxis[~bad_pixel_mask][g-npix_cutout:g+npix_cutout],
                                          weights=calibration_spectra[g-npix_cutout:g+npix_cutout])
                               for g in guess_pixels]
  
  linear_fit_wlmodel = linfitter(model=wlmodel, x=improved_xval_guesses, y=guess_wl)
  wavelengths = linear_fit_wlmodel(xaxis[~bad_pixel_mask]) * u.AA
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
  ax1.plot(wavelengths, calibration_spectra)
  ax1.plot(guess_wl, [consideration]*len(guess_wl), 'x', label="Guessed Values");
  ax1.set_xlabel("Wavelength($\AA$)")
  ax1.set_ylabel("Intensity")
  ax1.legend()

  ax2.plot(guess_pixels, guess_wl, 'o')
  ax2.plot(xaxis[~bad_pixel_mask], wavelengths, '-')
  ax2.set_ylabel("$\lambda(x)$")
  ax2.set_xlabel("x (pixels)")
  return wavelengths, linear_fit_wlmodel


def wavelength_solver(spectra, xlims, bad_pixel_mask, initial_wl_soln, fit_model, guess_wl, guess_pixels, intensity_scaling = 2):
  """
  This function is meant to help us get a true wavelength solution after doing initial fitting using guessed pixels-wavelength pairs

  *Note* This function DOES NOT check every single Neon line in the range from the first wavelength fit. Instead, this function locates the NIST information of the lines used in making guesses
  and simply uses those to fit a new wavelength solution. This avoids the lists being incorrect sizes and having errors at every step.


  Inputs (7, 8 if needed):
        spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was the same used in using wavelengths.intial_wl
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)
        
        initial_wl_soln: this is the returned wavelength solution from the wavelengths.intial_wl function

        fit_model: this is the returned fit model from the wavelengths.intial_wl function
        guess_wl: (list) the corresponding wavelengths that line up with the integer pixel values from the guess_pixels list

        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image
        
        intensity_scaling: (int) a scaling factor for the displayed neon lines from NIST, larger values plot them at larger y-axis values (default is 2)
  Returns (1):
        fit_model_with_true_neon: corrected model which uses the NIST lists
        
  """
  xaxis = np.arange(xlims[0], xlims[1], step=1)

  minwave = initial_wl_soln.min()
  maxwave = initial_wl_soln.max()
  neon_lines = Nist.query(minwav=minwave,
                          maxwav=maxwave,
                          wavelength_type='vac+air',
                          linename='Ne I')
  ne_wl_only_good = neon_lines['Observed']
  ne_rel_only_good = np.array([float(x) for x in neon_lines['Rel.']])
    
  wavelengths = np.array(ne_wl_only_good)
  df = pd.DataFrame(wavelengths)
    
  used_for_guesses = []
    
  for i in range(len(guess_wl)):
      checker = guess_wl[i]
      checked = np.abs((df - checker))
      val = checked.loc[checked[0] == np.min(checked)].index[0]
      used_for_guesses.append(val)
        
  ne_keep_final = ne_wl_only_good[used_for_guesses]
  ne_rel_only_good = ne_rel_only_good[used_for_guesses]
  ne_rel_intens = (ne_rel_only_good / ne_rel_only_good.max() * spectra.max())

  ne_pixel_vals = fit_model.inverse(ne_keep_final)

  xvals_ne_guess = np.concatenate([guess_pixels,
                                  ne_pixel_vals])
  waves_ne_guess = np.concatenate([guess_wl, ne_keep_final])
    
  fit_model_with_true_neon = linfitter(model=wlmodel,
                              x=xvals_ne_guess,
                              y=waves_ne_guess)
  wavelength_model = fit_model_with_true_neon(xaxis[~bad_pixel_mask]) * u.AA

  print("Original Fit\n" + str(fit_model) + "\n")
  print("Fit Using NIST as Well\n" + str(fit_model_with_true_neon) + "\n")
    
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,14))
  ax1.plot(initial_wl_soln, spectra)
  ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x')
  ax1.set_ylabel("Intensity")
  ax1.set_xlabel("Wavelength ($\AA$)")
    
  ax2.plot(wavelength_model, spectra)
  ax2.vlines(ne_keep_final, np.min(spectra), np.max(spectra), 'r', alpha=0.45, linestyle='--')
  for wl in ne_keep_final:
      plt.text(wl+4, np.max(spectra)-np.std(spectra), str(wl) +"$\AA$", rotation=90, ha='right', va='top')
  ax2.set_ylim(np.min(spectra), np.max(spectra));
  ax2.set_xlabel("Wavelength ($\AA$)");
  ax2.set_title("Calibration Neon Lamp")
  return fit_model_with_true_neon
