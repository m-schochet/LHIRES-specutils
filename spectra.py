import numpy as np
import matplotlib.pyplot as plt
import astropy

def spectra_producer(obj_image:np.ndarray, fit_model, mean_weights:np.ndarray, npix_tup:tuple, xlims=None, size=(10, 6), 
    plot_spectra=False, bad_pix_mask=None, obj_name=""):

    """
    This function is meant to help us create a spectra and check to make sure it looks alright. 
   
    ** Note that for the time being, this function can only get spectra if it is cutting out a portion along the y-axis 
    that is the same size as the npix_tup from the basics.py/tracer function. Future functionality will include the ability to
    downselect a smaller portion of the original spectrum **

    
    Inputs (4 needed, 8 possible):
        obj_image: (np.ndarray) image to use for the spectra (must be a fits file that has already been placed into a variable with fits.getdata)
        
        fit_model: (astropy.modeling.models) this is a returned model trace, can be gotten from the basics.py/tracer function
        
        mean_weights: (np.ndarray) this is the mean_trace_profile returned variable from basics.py/tracer, used to weight our spectra
        
        npix_tup: (tuple) use this to input the npix_ret tuple from basics.py/tracer so that we can cut out the spectra

        size: (tuple) use this to change the size of the plotted spectra (Default is 10,6)
        
        plot_spectra: (boolean) set to True if you want to see the spectra

        obj_name: (string) insert the name of the object for a plot title herex
            
        Optional: 
            bad_pix_mask: (np.ma.maskedarray) if there is a mask of bad pixels from the basics.py/tracer function, use this as an input of that mask

    Returns:
        spectra: (np.ndarray) the spectra of the opject
        
    """
    
    image_array = np.array(obj_image)
    image_array = image_array - np.median(image_array)
    xvals = np.arange(image_array.shape[1])
    xvals=xvals[xlims[0]:xlims[1]]
        
    if bad_pix_mask is not None and bad_pix_mask.any():
        trace = fit_model(xvals[~bad_pix_mask])
        spectra = np.array([np.average(image_array[int(yval)-npix_tup[0]:int(yval)+npix_tup[1], ii],
                            weights = mean_weights)
                                for yval, ii in zip(trace, xvals[~bad_pix_mask])])
    else:
        trace = fit_model(xvals)
        spectra = np.array([np.average(image_array[int(yval)-npix_tup[0]:int(yval)+npix_tup[1], ii],
                            weights = mean_weights)
                                for yval, ii in zip(trace, xvals)])

        
    if(plot_spectra==True):
        fig = plt.figure(figsize=size)
        ax1 = fig.add_subplot(111)
        ax1.plot(spectra)
        ax1.set_title("Spectra " +obj_name)
        
    return spectra

def point_finder(spectra, xaxis, mask, size=(8,4)):
    """
    This function is meant to help us display the spectra and be able to interact with the image to determine the location of lines (pixel values)
    
    ** Note that for this function to run, one needs to set the widget matplotlib backend. This requires an install of ipympl as well as pyqt. Without these 
    packages, the function will error**
    Inputs (3 needed, 4 possible)
        spectra: (np.ndaray) the spectra of the opject
        
        xaxis: (np.ndarray) the xaxis values of this spectra
        
        mask: (np.ma.maskedarray) if there is a mask of bad pixels from the basics.py/tracer function, use this as an input of that mask (otherwise simply use None)

        size: (tuple) variable to change the size of the plotted image
    Returns:
        Nothing, simply a plotter
        

    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    spec = ax.plot(xaxis[~mask], spectra)
    pos = []
    def onclick(event):
         pos.append([event.xdata,event.ydata])
    fig.canvas.mpl_connect('button_press_event', onclick)
