from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy import units as u
import numpy as np

linfitter = LinearLSQFitter()
wlmodel = Linear1D()


def central_wl(obj_image, fit_model, xlims=None, obj_name=None, micrometer=None):

    """
    This function is meant to help us get the central wavelength of an image which has been fit to a wavelength solution
   
    
    Inputs (2 needed, 4 possible):
        obj_image: (np.ndarray) image to use for the spectra (must be a fits file that has already been placed into a variable with fits.getdata)
        
        fit_model: (astropy.modeling.polynomial.Polynomial1D OR astropy.modeling.functional_models.Linear1D) this is a returned wavelength solution model,
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


def get_central_wls(list_wls, list_settings):

    """
    This function is meant get a solution function whereby you can insert a micrometer setting and get back out a central wavelength 
    
    Inputs (2 needed):
        list_wls: (list) a list of central wavelengths (NOTE: These MUST be astropy unit objects that have had their .value taken)
        
        list_settings: (list) associated micrometer settings to determine to central wavelength
        
        
    Returns (1):
        micrometer_model: (astropy.modeling.functional_models.Linear1D) fit model which intakes micrometer settings and outputs central wavelengths
        
    """
    central_wl_model = linfitter(model=wlmodel, x=list_settings, y=list_wls)
    return central_wl_model



def get_micrometer(list_wls, list_settings):

    """
    This function is meant get a solution function whereby you can insert a central wavelength and get back out a micrometer setting 
    
    Inputs (2 needed):
        list_wls: (list) a list of central wavelengths (NOTE: These MUST be astropy unit objects that have had their .value taken)
        
        list_settings: (list) associated micrometer settings to determine to central wavelength
        
        
    Returns (1):
        central_wl_model: (astropy.modeling.functional_models.Linear1D) fit model which intakes micrometer settings and outputs central wavelengths
        
    """
    micrometer_model = linfitter(model=wlmodel, x=list_wls, y=list_settings)
    return micrometer_model
