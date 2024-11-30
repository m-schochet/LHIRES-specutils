def central_wl(obj_image, fit_model, xlims=None, obj_name=None, micrometer=None):

    """
    This function is meant to help us create a spectra and check to make sure it looks alright. 
   
    ** Note that for the time being, this function can only get spectra if it is cutting out a portion along the y-axis 
    that is the same size as the npix_tup from the basics.py/tracer function. Future functionality will include the ability to
    downselect a smaller portion of the original spectrum **

    
    Inputs (2 needed, 4 possible):
        obj_image: (np.ndarray) image to use for the spectra (must be a fits file that has already been placed into a variable with fits.getdata)
        
        fit_model: (astropy.modeling.polynomial.Polynomial1D OR astropy.modeling.functional_models.LinearID) this is a returned wavelength solution model,
                    can be gotten from the wavelengths.py functions
                    
        Optional:
          
            xlims: (tuple) a tuple only used when the xaxis is shortened (i.e., a trace extends only partially through the image)

            obj_name: (string) insert the name of the object

            micrometer: (float) insert the micrometer setting to display
        
    Returns:
        cent_wl: (np.ndarray) the central wavelength of that image
        
    """
    
    image_array = np.array(obj_image)
    xvals = np.arange(image_array.shape[1])
    if(xlims is not None):
      xvals=xvals[xlims[0]:xlims[1]]
    lengther = len(xvals)
    cent_pix = lengther/2
    
    wl_at_cent = fit_model(cent_pix)*u.AA
    if((micrometer is not None) & (obj_name is not None)):
        print(str(obj_name) + " at micrometer setting " + str(micrometer) + " gives a central wavelength of " + str(wl_at_cent))
    return wl_at_cent
