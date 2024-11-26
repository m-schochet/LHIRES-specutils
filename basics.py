import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter, LMLSQFitter
from astropy.visualization import ZScaleInterval as zscale


"""
If you run these functions inside of a jupyter notebook, useful matplotlib settings include:

plt.rcParams['figure.dpi'] = 100  #quality of images, larger values takes longer to plot
plt.rcParams['image.origin'] = 'lower'  # we want to show images, not matrices, so we set the origin to the lower-left
plt.matplotlib.style.use('dark_background')  # Optional configuration: if run, this will look nice on dark background notebooks

*** Also, before using any function as shown below, please import your fits images and assign them to variables as below: ***
    
    # image = fits.getdata(path-to-fits-file)

*** where path-to-fits-file should be replaced with the path on your machine to the image surrounded by quotations ***


Fits Used

  wlmodel: 1st-order linear (wavelength solution)
  polymodel2 = 2-term polynomial (used in traces)
  polymodel3 = 3-term polynomial (used in traces)

Fitters

  linfitter = Linear least-squares algorithm
  lmlfitter = Levenberg–Marquardt nonlinear least-squares algorithm
  
"""
wlmodel = Linear1D()
polymodel2 = Polynomial1D(degree=2)
polymodel3 = Polynomial1D(degree=3)
linfitter = LinearLSQFitter()
lmlfitter = LMLSQFitter(calc_uncertainties=True)

def image_reduction(object_image, list_flats, list_bias=None, list_dark=None):
    """
    
    This function is meant to take in a list of optical spectra fits files and perform median flat division from science images. 
    Ideally, we would also have bias frames to subtract from the flats as well as darks to subtract away dark current from the 
    science images. However if you did not take darks, the function does not require them to run

    Inputs:
        object_image: (np.ndarray) image to be reduced (must be a fits file that has already been placed into a variable with fits.getdata)
        
        list_flats: (list) list of flat frames (each one must be a fits file that has already been placed into a variable with fits.getdata)
        
        *Note*
            if there is not a list of flats, simply use [] as the flat list input

        Optional:
            list_bias: (list) list of bias frames (each one must be a fits file that has already been placed into a variable with fits.getdata)
            
            list_dark: (list) list of dark frames (each one must be a fits file that has already been placed into a variable with fits.getdata)

    Returns: 
        science_image: (np.ndarray) the reduced science image of an object
    """
    if(list_flats != []):
        master_flat = np.median(list_flats)
        
        if((list_bias != None) & (list_dark != None)):
            master_bias = np.median(list_bias)
            master_dark = np.median(list_darks)
            
            for flat in enumerate(list_flats):
                flat = flat - master_bias   
                
            master_flat = np.median(list_flats)
    
            science_image = (object_image - master_dark) / master_flat
            
        elif(list_bias != None):
            master_bias = np.median(list_bias)
            
            for flat in enumerate(list_flats):
                flat = flat - mean_bias   
                
            master_flat = np.median(list_flats)
    
            science_image = (object_image - master_bias) / master_flat
            
        elif(list_dark != None):    
            master_dark = np.median(list_darks)
    
            science_image = (object_image - master_dark) / master_flat
            
        else:
            science_image = (object_image) / master_flat
    
        return science_image
    elif(list_flats == []):
        science_image = object_image
        return science_image

def scales(obj_image):
    """
    This function is meant to return the vmin-vmax values with zscale for a certain image
    
    Inputs (1):
        obj_image: (np.ndarray) image to be plotted (must be a fits file that has already been placed into a variable with fits.getdata)
    
    Returns (1): 
        vmin-vmax pairs
           
    """
    zscaler = zscale()
    scaler = zscaler.get_limits(obj_image)
    vmax = math.ceil(scaler[1])
    vmin = int(scaler[0])
    return (vmin, vmax)

def plotter(obj_image, obj_name, manual_vscales=None, obj_type="detector-direct", specification=None):
    """
    This function is meant to take an image and determine the optimal scaling to plot it
    
    Inputs (2, or up to 5):
        obj_image: (np.ndarray) image to be plotted (must be a fits file that has already been placed into a variable with fits.getdata)

        obj_name: (str) the name of the object (to be displayed on plot title)

        manual_vscales: (tuple) a tuple of manual vmin/vmax scalings for the plotter function
        
        obj_type: (str) the type of image you are plotting. Options are: "detector-direct (default), wavelengths, frequencies

        specification: (astropy.units) if plotting wavelengths or frequencies on the x-axis, specify the astropy.units of the x-axis
    Returns (1): 
        Nothing, this function is for plotting
           
    """
    zscaling = scales(obj_image)
    if(manual_vscales!=None):
        vmin = manual_vscales[0]
        vmax = manual_vscales[1]
    else:
        vmin = zscaling[0]
        vmax = zscaling[1]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(obj_image, norm='log', vmin=vmin, vmax=vmax)
    ax.set_title(obj_name, fontsize=20)
    if(obj_type=="detector-direct"):
        ax.set_xlabel("x-axis", fontsize=10)
        ax.set_ylabel("y-axis", fontsize=10)
    elif((obj_type=="wavelengths") | (obj_type=="frequencies")):
        unit = str(specification.unit)
        if(obj_type=="frequencies"):
            ax.set_xlabel("Frequency (" + unit + ")", fontsize=10)
            ax.set_ylabel("Intensity", fontsize=10)
        elif(obj_type=="wavelengths"):
            ax.set_xlabel("Wavelength (" + unit + ")", fontsize=10)
            ax.set_ylabel("Intensity", fontsize=10)

def tracer(obj_image, min_y, max_y, model, npix, vmin, vmax, aspect=0, npix_bot=None, hot_pix_min_cut=None, hot_pix_max_cut=None, plot_cutouts=False):
     
    """
    This function is meant to help us determine the trace of our object. After running this function, the trace is plotted, and so if there are
    hot pixels present, feel free to rerun this function with the hot pixel inputs specified
        
    Inputs (7 required, 12 including optional):
        obj_image: (np.ndarray) image to be traced (must be a fits file that has already been placed into a variable with fits.getdata)
        
        min_y: (int) what is the minimum y-axis value from where the weighted pixels should be judged
        
        max_y: (int) what is the maximum y-axis value from where the weighted pixels should be judged
        
        model: (astropy.modeling.models) use this to determine the fitter you want to use
            (expected either 2-term or 3-term polynomial [polymodel2/polymodel3], but other astropy.modeling.models can work)
        
        npix: (int) number of pixels to be cut out +/- to determine the weights of the trace 
            (if npix_bot is supplied, this is the number of pixels to be cut from the top)

        vmin: (int) vmin scaling on the images for plot_cutouts

        vmax: (int) vmax scaling on the images for plot_cutouts

        aspect: (int) how scaled should the cutout plots be
        
        plot_cutouts: (boolean) set to True if you want to see the effect of getting the weights
        
        Optional: 
            hot_pix_min_cut: (int) if the image has hot pixels, use this to select where those should be cut off (below on y-axis)
            
            hot_pix_max_cut: (int) if the image has hot pixels, use this to select where those should be cut off (above on y-axis)
            
            *Note*
                if these above optional parameters are given, the function needs to be called with four returned variables
                (both the weighted y-axis values, fit trace, mean weights, and the bad pixels mask)    
    
            npix_bot: (int) if the image needs different cuts of pixels on the top and bottom, use this to indicate the number of pixels to be cut on the bottom

    Returns (5, optional 6):
        *without hot pixel cut outs (5)*
            fit_model: (astropy.models.Polynomial1D) the trace of our object
            
            mean_trace_profile: (np.array) the weights of the trace for making spectra
            
            xvals: (np.array) the x-axis of this image

            weighted_yaxis_values: (np.array) the weighted y-axis used to make the images

            npix_ret: (tuple) pixels cut from below and above for use in spectra weighting 
            
        *with hot pixel cut outs (6)* 
            bad_pixels: (np.ma.maskedarray) the hot pixel mask
    """
    # Instantiating everything
    image_array = np.array(obj_image)
    
    yaxis = np.repeat(np.arange(min_y, max_y)[:,None],
                      image_array.shape[1], axis=1)
    xvals = np.arange(image_array.shape[1])
    
    weighted_yaxis_values = np.average(yaxis, axis=0, weights=image_array[min_y:max_y,:])

    # Determining trace
    if ((hot_pix_min_cut != None) | (hot_pix_max_cut != None)):
        if (hot_pix_min_cut != None):
            if (hot_pix_max_cut != None):
                bad_pixels = (weighted_yaxis_values > hot_pix_max_cut) | (weighted_yaxis_values < hot_pix_min_cut)
            else:
                bad_pixels = (weighted_yaxis_values < hot_pix_min_cut)
        elif (hot_pix_max_cut != None):
            bad_pixels = (weighted_yaxis_values > hot_pix_max_cut)
        
        fit_model = linfitter(model, xvals[~bad_pixels], weighted_yaxis_values[~bad_pixels])
        print(fit_model)
        
        plt.plot(xvals[~bad_pixels], weighted_yaxis_values[~bad_pixels], 'x')
        plt.plot(xvals[~bad_pixels], fit_model(xvals[~bad_pixels]))
        plt.title("Traced spectra on weighted y-values")

        trace = fit_model(xvals[~bad_pixels])
        if(npix_bot != None):
            cutouts = np.array([image_array[int(yval)-npix_bot:int(yval)+npix, ii]
                                    for yval, ii in zip(trace, xvals[~bad_pixels])])
            npix_ret = (npix_bot, npix)
        else:
            cutouts = np.array([image_array[int(yval)-npix:int(yval)+npix, ii]
                                for yval, ii in zip(trace, xvals[~bad_pixels])])
            npix_ret = (npix, npix)
        mean_trace_profile = cutouts.mean(axis=0)
        
        # Plotting trace
        if(plot_cutouts==True):
            fig = plt.figure(figsize=(12,8))
            ax1 = plt.subplot(1,2,1)
            ax1.imshow(image_array[int((trace-npix)[0]):int((trace+npix)[0]),:], 
                       extent=[0,image_array.shape[1],int((trace-npix)[0]),int((trace+npix)[0])],vmin=vmin, vmax=vmax)
            ax1.set_aspect(aspect)
            ax1.set_title("We go from this...")
            ax2 = plt.subplot(1,2,2)
            ax2.imshow(cutouts.T, vmin=vmin, vmax=vmax)
            ax2.set_title("...to this")
            ax2.set_aspect(aspect)
        
        return fit_model, mean_trace_profile, xvals, weighted_yaxis_values, npix_ret, bad_pixels
    
    else:
        fit_model = linfitter(model, xvals, weighted_yaxis_values)
        print(fit_model)
        
        plt.plot(xvals, weighted_yaxis_values, 'x')
        plt.plot(xvals, fit_model(xvals))
        plt.title("Traced spectra on weighted y-values")

        trace = fit_model(xvals)
        if(npix_bot != None):
            cutouts =  np.array([image_array[int(yval)-npix_bot:int(yval)+npix, ii]
                            for yval, ii in zip(trace, xvals)])
            npix_ret = (npix_bot, npix)
        else:
            cutouts = np.array([image_array[int(yval)-npix:int(yval)+npix, ii]
                                for yval, ii in zip(trace, xvals)])
            npix_ret = (npix, npix)
        mean_trace_profile = cutouts.mean(axis=0)

        #Plotting trace
        if(plot_cutouts==True):
            fig = plt.figure(figsize=(12,8))
            ax1 = plt.subplot(1,2,1)
            ax1.imshow(image_array[int((trace-npix)[0]):int((trace+npix)[0]),:], 
                       extent=[0,image_array.shape[1],int((trace-npix)[0]),int((trace+npix)[0])],vmin=vmin, vmax=vmax)
            ax1.set_aspect(aspect)
            ax1.set_title("We go from this...")
            ax2 = plt.subplot(1,2,2)
            ax2.imshow(cutouts.T, vmin=vmin, vmax=vmax)
            ax2.set_title("...to this")
            ax2.set_aspect(aspect)
        return fit_model, mean_trace_profile, xvals, weighted_yaxis_values, npix_ret

def residuals(model, xvals, yvals, bad_pixel_mask=None):
    """
    This function is meant to plot residuals from a certain wavelength trace
        
    Inputs (2 required, 3 if using a mask of bad pixels):
        model: (astropy.models) the fit trace of our object 
        
        xvals: (np.array) the x-axis of this traced image

        yvals: (np.array) (please use) weighted y-axis used to make the trace

        bad_pixel_mask: (np.ma.maskedarray) the masks of bad pixels for the image
        
    Returns (5, optional 6):
        nothing, this function simply plots residuals
    """   
    plt.figure(figsize=(8,4))
    plt.plot(xvals[~bad_pixel_mask],
          yvals[~bad_pixel_mask] - model(xvals[~bad_pixel_mask]), 'x')
    plt.title("Residuals", fontsize=20)
    plt.xlabel("x-axis", fontsize=10)
    plt.ylabel("Residual (data-model)", fontsize=10)
