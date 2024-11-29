import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.table import Column
from astroquery.nist import Nist
from astropy.modeling.models import Linear1D, Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter

linfitter = LinearLSQFitter()
wlmodel = Linear1D()
polymodel2 = Polynomial1D(degree=2)

def initial_wl(calibration_spectra, xlims, bad_pixel_mask, guess_wl, guess_pixels, npix_cutout, manual_scale=1.01):

  """
  
  This function is meant to help us get an initial feel for the wavelength solution for a calibration spectra when given a set of guessed pixels of lines and associated wavelengths of those lines

  *Note, since this function simply takes *guesses* and turns them into a solution, one can provide pixel/wavelength pairs for non-Neon lines too*

  Inputs (6, 7 if needed):
        calibration_spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was made using spectra.spectra_producer)
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)
        
        guess_wl: (list) the corresponding wavelengths that line up with the integer pixel values from the guess_pixels list
        
        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image (can be found using spectra.point_finder)
      
        npix_cutout: (int) a number of x-axis values to +/- cut around when measuring and adjusting guessed x-axis values

        manual_scale: (float) a number used to scale the points representing "guesses" on the graphs produced by this function (default is 1.01)
  Returns (3):
        wavelengths: (np.ndarray) a completed wavelength solution that considers only the guessed lines
        
        improved_xval_guesses: (np.ndarray) the improved x-axis values using weighted averaging

        linear_fit_wlmodel: (astropy.modeling.fitting.LinearLSQFitter) an initial linear wavelength solution that considers only the guessed lines
        
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
  return wavelengths, improved_xval_guesses, linear_fit_wlmodel


def wavelength_lin_solver(spectra, xlims, bad_pixel_mask, improved_xval_guesses, initial_wl_soln, fit_model, guess_wl, guess_pixels, intensity_scaling = 2, argon=False, argon_wls=[], argon_pixels=[]):
  """
  This function is meant to help us get a true *linear* wavelength solution after doing initial fitting using guessed pixels-wavelength pairs

  *Note* This function DOES NOT check every single Neon line in the range from the first wavelength fit. Instead, this function locates the
  NIST information of the lines used in making guesse and simply uses those to fit a new wavelength solution. 
  This avoids the lists being incorrect sizes and having errors at every step.


  Inputs (8, 13 if needed):
        spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was the same used in using wavelengths.intial_wl
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)

        improved_xval_guesses: (np.ndarray) the improved x-axis values using weighted averaging from the wavelengths.intial_wl function

        initial_wl_soln: (np.ndarray) this is the returned wavelength solution from the wavelengths.intial_wl function

        fit_model: (astropy.modeling.fitting.LinearLSQFitter) this is the returned fit model from the wavelengths.intial_wl function
        
        guess_wl: (list) the corresponding **NEON** wavelengths that line up with the integer pixel values from the guess_pixels list

        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image
        
        intensity_scaling: (int) a scaling factor for the displayed neon lines from NIST, larger values plot them at larger y-axis values (default is 2)

        argon: (bool) set to True if you also want to fit a set of Argon lines into the spectra as well

        argon_wls: (list) list of wavelengths of Argon lines if argon is set to True

        argon_pixels: (list) list of pixels of associated Argon lines if argon is set to True
        
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
  a = np.array([(True if "*" in str(val) else False) for val in neon_lines['Rel.'].value])
  ne_keep = (neon_lines['Rel.'] != "*") & (~a)

  ne_wl_only_good = neon_lines['Observed'][ne_keep]
  ne_rel_only_good = np.array([float(x) for x in neon_lines['Rel.'][ne_keep]])
    
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
  wavelength_model1 = fit_model_with_true_neon(xaxis[~bad_pixel_mask]) * u.AA

  if (argon!=True):
    print("Original Fit\n" + str(fit_model) + "\n")
    print("Fit Using NIST as Well\n" + str(fit_model_with_true_neon) + "\n")
    
    fig = plt.figure(layout="constrained")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax1 = plt.subplot2grid((3,4), (0, 0), colspan=4, rowspan=1)
    ax2 = plt.subplot2grid((3,4), (1, 0), colspan=4, rowspan=1)
    ax3 = plt.subplot2grid((3,4), (2, 0), colspan=4, rowspan=1)
    

    ax1.plot(initial_wl_soln, spectra)
    ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x')
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength ($\AA$)")
      
    ax2.plot(wavelength_model1, spectra)
    ax2.vlines(ne_keep_final, np.min(spectra), np.max(spectra), 'r', alpha=0.45, linestyle='--')
    for wl in ne_keep_final:
        ax2.text(wl+4, np.max(spectra)-np.std(spectra), str(wl) +"$\AA$", rotation=90, ha='right', va='top')
    ax2.set_ylim(np.min(spectra), np.max(spectra));
    ax2.set_xlabel("Wavelength ($\AA$)");
    ax2.set_title("Calibration Neon Lamp")
    
    residuals_guesses = np.array(guess_wl)  - fit_model(improved_xval_guesses)
    residuals_NIST = np.array(ne_keep_final)  - fit_model_with_true_neon(improved_xval_guesses)
    
    ax3.plot(improved_xval_guesses, residuals_guesses, 'x', label="Guesses")
    ax3.plot(improved_xval_guesses, residuals_NIST, '+', label="Guesses+NIST")

    ax3.set_xlabel("Pixel Coordinate")
    ax3.set_ylabel("Wavelength residual ($\AA$)");
    ax3.set_title("Residuals")
    ax3.legend()
    
    return fit_model_with_true_neon
  else:
    argon_lines = Nist.query(minwav=minwave,
                            maxwav=maxwave,
                            wavelength_type='vac+air',
                            linename='Ar I')
    b = np.array([(True if "*" in str(val) else False) for val in argon_lines['Rel.'].value])
    ar_keep = (argon_lines['Rel.'] != "*") &  (~argon_lines['Rel.'].mask) & (~b)
    ar_wl_only_good = argon_lines['Observed'][ar_keep]
    ar_rel_only_good = np.array([float(x) for x in argon_lines['Rel.'][ar_keep]])
          
    wavelengths2 = np.array(ar_wl_only_good)
    df2 = pd.DataFrame(wavelengths2)
          
    used_for_guesses2 = []
    
    for i in range(len(argon_wls)):
        checker2 = argon_wls[i]
        checked2 = np.abs((df2 - checker2))
        val2 = checked2.loc[checked2[0] == np.min(checked2)].index[0]
        used_for_guesses2.append(val2)

    ar_keep_final = Column(ar_wl_only_good[used_for_guesses2])
    
    ar_rel_only_good = ar_rel_only_good[used_for_guesses2]
    ar_rel_intens = (ar_rel_only_good / ar_rel_only_good.max() * spectra.max())
    
    ar_pixel_vals = fit_model.inverse(ar_keep_final)
    
    xvals_ar_guess = np.concatenate([ne_pixel_vals, ar_pixel_vals])
    waves_ar_guess = np.concatenate([ne_keep_final, ar_keep_final])
      
    fit_model_with_argon_neon = linfitter(model=wlmodel,
                                x=xvals_ar_guess,
                                y=waves_ar_guess)
    wavelength_model2 = fit_model_with_argon_neon(xaxis[~bad_pixel_mask]) * u.AA
    print("Original Fit\n" + str(fit_model) + "\n")
    print("Fit Using NIST (+Argon) and no Guesses\n" + str(fit_model_with_argon_neon) + "\n")
    
    fig = plt.figure(layout="constrained")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax1 = plt.subplot2grid((3,4), (0, 0), colspan=4, rowspan=1)
    ax2 = plt.subplot2grid((3,4), (1, 0), colspan=4, rowspan=1)
    ax3 = plt.subplot2grid((3,4), (2, 0), colspan=4, rowspan=1)
    
    ax1.plot(initial_wl_soln, spectra)
    ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x', label="Neon")
    ax1.plot(ar_keep_final, ar_rel_intens*intensity_scaling, '+', label="Argon")
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength ($\AA$)")
    ax1.legend()
      
    ax2.plot(wavelength_model2, spectra)
    ax2.vlines(ar_keep_final, np.min(spectra), np.max(spectra), 'orange', alpha=0.25, linestyle='--', label="Argon")
    ax2.vlines(ne_keep_final, np.min(spectra), np.max(spectra), 'r', alpha=0.25, linestyle='--', label="Neon")
    for wl in ne_keep_final:
        ax2.text(wl+4, np.max(spectra)-np.std(spectra), str(wl) +"$\AA$", rotation=90, ha='right', va='top')
    for wl2 in ar_keep_final:
        ax2.text(wl2+4, np.max(spectra)-np.std(spectra), str(wl2) +"$\AA$", rotation=90, ha='right', va='top')
    
    ax2.set_ylim(np.min(spectra), np.max(spectra));
    ax2.set_xlabel("Wavelength ($\AA$)");
    ax2.set_title("Calibration Neon Lamp")
    ax2.legend()
      
    guessed_wl = guess_wl + argon_wls
    residuals_guesses = np.array(guessed_wl)  - fit_model(improved_xval_guesses)
    residuals_NIST = np.array(waves_ar_guess)  - fit_model_with_true_neon(improved_xval_guesses)
    
    ax3.plot(improved_xval_guesses, residuals_guesses, 'x', label="Guesses")
    ax3.plot(improved_xval_guesses, residuals_NIST, '+', label="Guesses+NIST")

    ax3.set_xlabel("Pixel Coordinate")
    ax3.set_ylabel("Wavelength residual ($\AA$)");
    ax3.set_title("Residuals")
    ax3.legend()
    
    return fit_model_with_argon_neon
  
def wavelength_polynomial_solver(spectra, xlims, bad_pixel_mask, improved_xval_guesses, initial_wl_soln, fit_model, guess_wl, guess_pixels, intensity_scaling = 2, argon=False, argon_wls=[], argon_pixels=[]):
  """
  This function is meant to help us get a true *polynomial* wavelength solution after doing initial fitting using guessed pixels-wavelength pairs

  *Note* This function DOES NOT check every single Neon line in the range from the first wavelength fit. Instead, this function locates the NIST information of the lines used in making guesses
  and simply uses those to fit a new wavelength solution. This avoids the lists being incorrect sizes and having errors at every step.


  Inputs (8, 12 if needed):
        spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was the same used in using wavelengths.intial_wl
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)

        improved_xval_guesses: (np.ndarray) the improved x-axis values using weighted averaging from the wavelengths.intial_wl function

        initial_wl_soln: (np.ndarray) this is the returned wavelength solution from the wavelengths.intial_wl function

        fit_model: (astropy.modeling.fitting.LinearLSQFitter) this is the returned fit model from the wavelengths.intial_wl function
        
        guess_wl: (list) the corresponding **NEON** wavelengths that line up with the integer pixel values from the guess_pixels list

        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image
        
        intensity_scaling: (int) a scaling factor for the displayed neon lines from NIST, larger values plot them at larger y-axis values (default is 2)

        argon: (bool) set to True if you also want to fit a set of Argon lines into the spectra as well

        argon_wls: (list) list of wavelengths of Argon lines if argon is set to True

        argon_pixels: (list) list of pixels of associated Argon lines if argon is set to True
        
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
  a = np.array([(True if "*" in str(val) else False) for val in neon_lines['Rel.'].value])
  ne_keep = (neon_lines['Rel.'] != "*") & (~a)

  ne_wl_only_good = neon_lines['Observed'][ne_keep]
  ne_rel_only_good = np.array([float(x) for x in neon_lines['Rel.'][ne_keep]])
    
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
    
  fit_model_with_true_neon = linfitter(model=polymodel2,
                              x=xvals_ne_guess,
                              y=waves_ne_guess)
  wavelength_model1 = fit_model_with_true_neon(xaxis[~bad_pixel_mask]) * u.AA

  if (argon!=True):
    print("Original Linear Fit\n" + str(fit_model) + "\n")
    print("Polynomial Fit Using NIST as Well\n" + str(fit_model_with_true_neon) + "\n")
    fig = plt.figure(layout="constrained")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax1 = plt.subplot2grid((3,4), (0, 0), colspan=4, rowspan=1)
    ax2 = plt.subplot2grid((3,4), (1, 0), colspan=4, rowspan=1)
    ax3 = plt.subplot2grid((3,4), (2, 0), colspan=4, rowspan=1)

    ax1.plot(initial_wl_soln, spectra)
    ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x')
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength ($\AA$)")
      
    ax2.plot(wavelength_model1, spectra)
    ax2.vlines(ne_keep_final, np.min(spectra), np.max(spectra), 'r', alpha=0.45, linestyle='--')
    for wl in ne_keep_final:
        ax2.text(wl+4, np.max(spectra)-np.std(spectra), str(wl) +"$\AA$", rotation=90, ha='right', va='top')
    ax2.set_ylim(np.min(spectra), np.max(spectra));
    ax2.set_xlabel("Wavelength ($\AA$)");
    ax2.set_title("Calibration Neon Lamp")
    
    residuals_guesses = np.array(guess_wl)  - fit_model(improved_xval_guesses)
    residuals_NIST = np.array(ne_keep_final)  - fit_model_with_true_neon(improved_xval_guesses)
    
    ax3.plot(improved_xval_guesses_alt, residuals_guesses, 'x')
    ax3.plot(improved_xval_guesses_alt, residuals_NIST, '+')
    
    ax3.set_xlabel("Pixel Coordinate")
    ax3.set_ylabel("Wavelength residual ($\AA$)");
    ax3.set_title("Residuals")
    ax3.legend()
    return fit_model_with_true_neon
    
  else:
    argon_lines = Nist.query(minwav=minwave,
                            maxwav=maxwave,
                            wavelength_type='vac+air',
                            linename='Ar I')
    b = np.array([(True if "*" in str(val) else False) for val in argon_lines['Rel.'].value])
    ar_keep = (argon_lines['Rel.'] != "*") &  (~argon_lines['Rel.'].mask) & (~b)
    ar_wl_only_good = argon_lines['Observed'][ar_keep]
    ar_rel_only_good = np.array([float(x) for x in argon_lines['Rel.'][ar_keep]])
          
    wavelengths2 = np.array(ar_wl_only_good)
    df2 = pd.DataFrame(wavelengths2)
          
    used_for_guesses2 = []
    
    for i in range(len(argon_wls)):
        checker2 = argon_wls[i]
        checked2 = np.abs((df2 - checker2))
        val2 = checked2.loc[checked2[0] == np.min(checked2)].index[0]
        used_for_guesses2.append(val2)
    
    ar_keep_final = Column(ar_wl_only_good[used_for_guesses2])
    
    ar_rel_only_good = ar_rel_only_good[used_for_guesses2]
    ar_rel_intens = (ar_rel_only_good / ar_rel_only_good.max() * spectra.max())
    
    ar_pixel_vals = fit_model.inverse(ar_keep_final)
    
    xvals_ar_guess = np.concatenate([ne_pixel_vals, ar_pixel_vals])
    waves_ar_guess = np.concatenate([ne_keep_final, ar_keep_final])
      
    fit_model_with_argon_neon = linfitter(model=polymodel2,
                                x=xvals_ar_guess,
                                y=waves_ar_guess)
    wavelength_model2 = fit_model_with_argon_neon(xaxis[~bad_pixel_mask]) * u.AA
    print("Original Linear Fit\n" + str(fit_model) + "\n")
    print("Polynomial Fit Using NIST (+Argon) and no Guesses\n" + str(fit_model_with_argon_neon) + "\n")
    
    fig = plt.figure(layout="constrained")
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax1 = plt.subplot2grid((3,4), (0, 0), colspan=4, rowspan=1)
    ax2 = plt.subplot2grid((3,4), (1, 0), colspan=4, rowspan=1)
    ax3 = plt.subplot2grid((3,4), (2, 0), colspan=4, rowspan=1)
    
    ax1.plot(initial_wl_soln, spectra)
    ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x', label="Neon")
    ax1.plot(ar_keep_final, ar_rel_intens*intensity_scaling, '+', label="Argon")
    ax1.set_ylabel("Intensity")
    ax1.set_xlabel("Wavelength ($\AA$)")
    ax1.legend()
      
    ax2.plot(wavelength_model2, spectra)
    ax2.vlines(ar_keep_final, np.min(spectra), np.max(spectra), 'orange', alpha=0.25, linestyle='--', label="Argon")
    ax2.vlines(ne_keep_final, np.min(spectra), np.max(spectra), 'r', alpha=0.25, linestyle='--', label="Neon")
    for wl in ne_keep_final:
        ax2.text(wl+4, np.max(spectra)-np.std(spectra), str(wl) +"$\AA$", rotation=90, ha='right', va='top')
    for wl2 in ar_keep_final:
        ax2.text(wl2+4, np.max(spectra)-np.std(spectra), str(wl2) +"$\AA$", rotation=90, ha='right', va='top')
    
    ax2.set_ylim(np.min(spectra), np.max(spectra));
    ax2.set_xlabel("Wavelength ($\AA$)");
    ax2.set_title("Calibration Neon Lamp")
    ax2.legend()
      
    guessed_wl = guess_wl + argon_wls
    residuals_guesses = np.array(guessed_wl)  - fit_model(improved_xval_guesses)
    residuals_NIST = np.array(waves_ar_guess)  - fit_model_with_argon_neon(improved_xval_guesses)
    
    ax3.plot(improved_xval_guesses, residuals_guesses, 'x', label="Guesses")
    ax3.plot(improved_xval_guesses, residuals_NIST, '+', label="Guesses+NIST")

    ax3.set_xlabel("Pixel Coordinate")
    ax3.set_ylabel("Wavelength residual ($\AA$)");
    ax3.set_title("Residuals")
    ax3.legend()

    return fit_model_with_argon_neon
def wavelength_argon_solver(spectra, xlims, bad_pixel_mask, improved_xval_guesses, initial_wl_soln, fit_model, guess_wl, guess_pixels, poly=False, intensity_scaling = 2):
  """
  This function is meant to help us get a true *linear or polynomial* wavelength solution after doing initial fitting using guessed pixels-wavelength pairs for spectra
  which ONLY HAVE ARGON LINES IN THEM

  *Note* This function DOES NOT check every single Argon line in the range from the first wavelength fit. Instead, this function locates the
  NIST information of the lines used in making guesse and simply uses those to fit a new wavelength solution. 
  This avoids the lists being incorrect sizes and having errors at every step.


  Inputs (8, 10 if needed):
        spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was the same used in using wavelengths.intial_wl
        
        xlims: (tuple) a set of xmin/xmax values to create an x-axis with
        
        bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)

        improved_xval_guesses: (np.ndarray) the improved x-axis values using weighted averaging from the wavelengths.intial_wl function

        initial_wl_soln: (np.ndarray) this is the returned wavelength solution from the wavelengths.intial_wl function

        fit_model: (astropy.modeling.fitting.LinearLSQFitter) this is the returned fit model from the wavelengths.intial_wl function
        
        guess_wl: (list) the corresponding **Argon** wavelengths that line up with the integer pixel values from the guess_pixels list

        guess_pixels: (list) a set of guessed pixels (MUST BE integers) in the calibration image

        poly: (bool) set this to True if you want to fit using only Argon lines but with a 2-term polynomial instead of linear fit
        
        intensity_scaling: (int) a scaling factor for the displayed neon lines from NIST, larger values plot them at larger y-axis values (default is 2)
        
  Returns (1):
        fit_model_with_true_argon: corrected model which uses the NIST lists
        
  """
     
  xaxis = np.arange(xlims[0], xlims[1], step=1)

  minwave = initial_wl_soln.min()
  maxwave = initial_wl_soln.max()
  argon_lines = Nist.query(minwav=minwave,
                          maxwav=maxwave,
                          wavelength_type='vac+air',
                          linename='Ar I')
  b = np.array([(True if "*" in str(val) else False) for val in argon_lines['Rel.'].value])
  ar_keep = (argon_lines['Rel.'] != "*") &  (~argon_lines['Rel.'].mask) & (~b)
  ar_wl_only_good = argon_lines['Observed'][ar_keep]
  ar_rel_only_good = np.array([float(x) for x in argon_lines['Rel.'][ar_keep]])
        
  wavelengths2 = np.array(ar_wl_only_good)
  df2 = pd.DataFrame(wavelengths2)
        
  used_for_guesses2 = []
  
  for i in range(len(guess_wl)):
      checker2 = guess_wl[i]
      checked2 = np.abs((df2 - checker2))
      val2 = checked2.loc[checked2[0] == np.min(checked2)].index[0]
      used_for_guesses2.append(val2)

  ar_keep_final = Column(ar_wl_only_good[used_for_guesses2])
  
  ar_rel_only_good = ar_rel_only_good[used_for_guesses2]
  ar_rel_intens = (ar_rel_only_good / ar_rel_only_good.max() * spectra.max())
  
  ar_pixel_vals = fit_model.inverse(ar_keep_final)
  
  xvals_ar_guess = np.concatenate([guess_pixels, ar_pixel_vals])
  waves_ar_guess = np.concatenate([guess_wl, ar_keep_final])
  if (poly!=True):  
    fit_model_with_argon_neon = linfitter(model=wlmodel,
                                x=xvals_ar_guess,
                                y=waves_ar_guess)
  else:
    fit_model_with_argon_neon = linfitter(model=polymodel2,
                                x=xvals_ar_guess,
                                y=waves_ar_guess)
    
  wavelength_model2 = fit_model_with_argon_neon(xaxis[~bad_pixel_mask]) * u.AA
  print("Original Fit\n" + str(fit_model) + "\n")
  print("Fit Using NIST (+Argon) and no Guesses\n" + str(fit_model_with_argon_neon) + "\n")
  
  fig = plt.figure(layout="constrained")
  fig.set_figheight(12)
  fig.set_figwidth(12)
  ax1 = plt.subplot2grid((3,4), (0, 0), colspan=4, rowspan=1)
  ax2 = plt.subplot2grid((3,4), (1, 0), colspan=4, rowspan=1)
  ax3 = plt.subplot2grid((3,4), (2, 0), colspan=4, rowspan=1)
  
  ax1.plot(initial_wl_soln, spectra)
  ax1.plot(ne_keep_final, ne_rel_intens*intensity_scaling, 'x', label="Neon")
  ax1.plot(ar_keep_final, ar_rel_intens*intensity_scaling, '+', label="Argon")
  ax1.set_ylabel("Intensity")
  ax1.set_xlabel("Wavelength ($\AA$)")
  ax1.legend()
    
  ax2.plot(wavelength_model2, spectra)
  ax2.vlines(ar_keep_final, np.min(spectra), np.max(spectra), 'orange', alpha=0.25, linestyle='--', label="Argon")
  for wl2 in ar_keep_final:
      ax2.text(wl2+4, np.max(spectra)-np.std(spectra), str(wl2) +"$\AA$", rotation=90, ha='right', va='top')
  
  ax2.set_ylim(np.min(spectra), np.max(spectra));
  ax2.set_xlabel("Wavelength ($\AA$)");
  ax2.set_title("Calibration Neon Lamp")
  ax2.legend()
    
  guessed_wl = guess_wl + argon_wls
  residuals_guesses = np.array(guessed_wl)  - fit_model(improved_xval_guesses)
  residuals_NIST = np.array(waves_ar_guess)  - fit_model_with_true_neon(improved_xval_guesses)
  
  ax3.plot(improved_xval_guesses, residuals_guesses, 'x', label="Guesses")
  ax3.plot(improved_xval_guesses, residuals_NIST, '+', label="Guesses+NIST")

  ax3.set_xlabel("Pixel Coordinate")
  ax3.set_ylabel("Wavelength residual ($\AA$)");
  ax3.set_title("Residuals")
  ax3.legend()
  return fit_model_with_true_argon

  
def inverse_polymodel(wl_list, wl_model, xlims, bad_pixel_mask, backwards=False):
    """
      This function is meant to help us get the inverse pixel values for a given polynomial wavelenght 
      
      Inputs (4, 5 if needed):
            spectra: (np.ndarray) the spectra of a calibration lamp (assumed that it was the same used in using wavelengths.intial_wl
            
            wl_list: (list) a list of wavelengths that need to be turned into pixel values
            
            wl_model: (np.ndarray) the fitted polynomial model
            
            xlims: (tuple) a set of xmin/xmax values to create an x-axis with
            
            bad_pixel_mask: (np.ma.maskedarray) this is an array of x-axis values being masked over for one reason or another (also from spectra.spectra_producer)
    
            backward: (bool) set this value to True in the event the wavelength solution is backwards for a certain image. If you do not set this to True for a backwards
                        solution, then the pixel values we take from the solution will be flipped
            
      Returns (1):
            interp_wls: (list) an interpolated pixel value list

    """
    xvals = xaxis[~bad_pix_mask]
    wavelengths = wl_model(xvals)
    if(backwards==True):
        return np.interp(wl_list, wavelengths[::-1], xvals[::-1])
    else:
      return np.interp(wl_list, wavelengths, xvals) 
