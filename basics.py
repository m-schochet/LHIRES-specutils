import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter, LMLSQFitter


"""
If you run these functions inside of a jupyter notebook, useful matplotlib settings include:

plt.rcParams['figure.dpi'] = 100  #quality of images, larger values takes longer to plot
plt.rcParams['image.origin'] = 'lower'  # we want to show images, not matrices, so we set the origin to the lower-left
plt.matplotlib.style.use('dark_background')  # Optional configuration: if run, this will look nice on dark background notebooks

*** Also, before using any function as shown below, please import your fits images and assign them to variables as below: ***
    
    # image = fits.getdata(path-to-fits-file)

*** where path-to-fits-file should be replaced with the path on your machine to the image surrounded by quotations ***

"""

"""

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
        
def plotter(obj_image, yval=None):
    """
    This function is meant to take an image and determine the optimal scaling to plot it
    
    Inputs:
        obj_image: (np.ndarray) image to be plotted (must be a fits file that has already been placed into a variable with fits.getdata)

        Optional:
            y_val: (int) yval to calculate minimum/maximum scaling values for the plot
            
            *Note*
                This is for use if the plotter does not automatically accurately scale the image using min/maxing
                such that a specific y-axis value needs to be given for accurate scaling

    Returns: 
        Nothing, this function is for plotting
           
    """
    if(yval != None):
        middle_y_axis = yval
    else:
        middle_y_axis = int(np.shape(obj_image[:,0])[0]/2)
    
    histogram = plt.hist(obj_image[middle_y_axis].flatten(), bins='auto')
    plt.clf()
    vmin = histogram[1].min()
    vmax = histogram[1].max()

    plt.imshow(obj_image, norm='log', vmin=vmin, vmax=vmax)

def tracer(obj_image, min_y, max_y, model, npix, npix_bot=None, hot_pix_min_cut=None, hot_pix_max_cut=None, plot_cutouts=False):
     
    """
    This function is meant to help us determine the trace of our object. After running this function, the trace is plotted, and so if there are
    hot pixels present, feel free to rerun this function with the hot pixel inputs specified
        
    Inputs:
        obj_image: (np.ndarray) image to be traced (must be a fits file that has already been placed into a variable with fits.getdata)
        
        min_y: (int) what is the minimum y-axis value from where the weighted pixels should be judged
        
        max_y: (int) what is the maximum y-axis value from where the weighted pixels should be judged
        
        model: (astropy.modeling.models) use this to determine the fitter you want to use
            (expected either 2-term or 3-term polynomial [polymodel2/polymodel3], but other astropy.modeling.models can work)
        
        npix: (int) number of pixels to be cut out +/- to determine the weights of the trace 
            (if npix_bot is supplied, this is the number of pixels to be cut from the top)
        
        plot_cutouts: (boolean) set to True if you want to see the effect of getting the weights
        
        Optional: 
            hot_pix_min_cut: (int) if the image has hot pixels, use this to select where those should be cut off (below on y-axis)
            
            hot_pix_max_cut: (int) if the image has hot pixels, use this to select where those should be cut off (above on y-axis)
            
            *Note*
                if these above optional parameters are given, the function needs to be called with four returned variables
                (both the weighted y-axis values, fitted trace, mean weights, and the bad pixels mask)    
    
            npix_bot: (int) if the image needs different cuts of pixels on the top and bottom, use this to indicate the number of pixels to be cut on the bottom

    Returns:
        *without hot pixel cut outs* (3)
            fitted_model: the trace of our object (Polynomial1D)
            
            mean_trace_profile: the weights of the trace for making spectra (Array)
            
            npix_ret: pixels cut from below and above for use in spectra weighting (tuple)

        *with hot pixel cut outs* (4)
            bad_pixels: the hot pixel mask (MaskedArray)
            
            fitted_model: the trace of our object (Polynomial1D)
            
            mean_trace_profile: the weights of the trace for making spectra (Array)
            
            npix_ret: pixels cut from below and above for use in spectra weighting (tuple)
    """
    # Instantiating everything
    image_array = np.array(obj_image)
    image_array = image_array - np.median(image_array)
    
    yaxis = np.repeat(np.arange(min_y, max_y)[:,None],
                      image_array.shape[1], axis=1)
    xvals = np.arange(image_array.shape[1])
    
    weighted_yaxis_values = np.average(yaxis, axis=0, weights=image_array[min_y:max_y,:])

    # Determining trace
    if ((hot_pix_min_cut != None) & (hot_pix_max_cut != None)):
        bad_pixels = (weighted_yaxis_values > hot_pix_max_cut) | (weighted_yaxis_values < hot_pix_min_cut)
        
        fitted_model = linfitter(model, xvals[~bad_pixels], weighted_yaxis_values[~bad_pixels])
        print(fitted_model)
        
        plt.plot(xvals[~bad_pixels], weighted_yaxis_values[~bad_pixels], 'x')
        plt.plot(xvals[~bad_pixels], fitted_model(xvals[~bad_pixels]))
        plt.title("Traced spectra on weighted y-values")

        trace = fitted_model(xvals[~bad_pixels])
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
                       extent=[0,image_array.shape[1],int((trace-npix)[0]),int((trace+npix)[0])],vmin=0, vmax=100)
            ax1.set_aspect(20)
            ax1.set_title("We go from this...")
            ax2 = plt.subplot(1,2,2)
            ax2.imshow(cutouts.T, vmin=0, vmax=100)
            ax2.set_title("...to this")
            ax2.set_aspect(20)
        
        return bad_pixels, fitted_model, mean_trace_profile, npix_ret
    
    else:
        fitted_model = linfitter(model, xvals, weighted_yaxis_values)
        print(fitted_model)
        
        plt.plot(xvals, weighted_yaxis_values, 'x')
        plt.plot(xvals, fitted_model(xvals))
        plt.title("Traced spectra on weighted y-values")

        trace = fitted_model(xvals)
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
                       extent=[0,image_array.shape[1],int((trace-npix)[0]),int((trace+npix)[0])],vmin=0, vmax=100)
            ax1.set_aspect(20)
            ax1.set_title("We go from this...")
            ax2 = plt.subplot(1,2,2)
            ax2.imshow(cutouts.T, vmin=0, vmax=100)
            ax2.set_title("...to this")
            ax2.set_aspect(20)
        return fitted_model, mean_trace_profile, npix_ret
