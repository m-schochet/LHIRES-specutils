# LHIRES Spectroscopy Utilities (LHIRES-specutils)

Hello and welcome! 

Before continuing any further, **make sure that spectra used for these tools are done using the typical High Resolution grating on the [SHELYAK LHIRES-III](https://www.shelyak.com/produit/spectroscope-lhires-iii/?lang=en) spectrograph .**

This github provides a set of functions and data reduction tools put together for interaction with spectra produced. In this github you will find the following.

## requirements.txt
This file serves as a reminder of the specific general python packages needed to run these functions.

## basics.py
This is the essential basic functions used to reduce and get data from LHIRES spectra. Inside here are a number of functions

	a) image_reduction 
		This function does image reduction on a 
		spectra (or list of spectra) if provided 
		a list of flats (required), darks (optional),
		and biases (optional). If none of these 
		files are available, this function isn't super useful
	
	b) scales
		This function returns a tuple of vmin/vmax values 
		per the astropy Zscale function
	
	c) plotter	
		This function simply plots an LHIRES spectra from 
		the fits file
	
	d) tracer	
		This function is the culmination of the basics.py file, 
		and it determines an accurate trace for whatever object is
		observed through an LHIRES slit and plots it (optionally). The
		returned variables from this function are essentially
		important for making spectra and wavelength solutions 

	e) residuals
		This function simply determines and plots the residuals 
		from a trace, and similar to plotter is just a plotting
		function

## spectra.py
This file serves as the main spectra creation file, and it uses the variables from functions in basics.py to create/plot/return spectra. Inside are the following functions:

	a) spectra_producer
		This is the main function for creating spectra, and it
		returns a single spectra as well as optionally plotting 
		the spectra 	
	
	b) point_finder
		This function plots the spectra produced in spectra_producer
		as an interactive widget so that pixel-wavelength pairs can
		be determined in the process of making a wavelength solution

***NOTE: The function above requires the matplotlib widget backend, which requires the installation of the [ipympl](https://matplotlib.org/ipympl/) package for the images to display properly. If using this function, ensure this package is installed.***

**As a result, in any cell where the point_finder function is called, one will necessarily have to run**
		
	matplotlib.use("widget")

**at the top of the cell. After this cell is run and the specific point-wavelength pairs are selected, the next cell plotting a non-interactive figure should run**

	matplotlib.use("inline")

**or else every other plot will display as an interactive plot, which uses more memory and may slow down the associated Jupyter notebook**


## wavelengths.py
This file serves as a the wavelength solution maker using all of the variables and information deduced from the basics.py and spectra.py files. 

*Note: The LHIRES-III spectrograph calibration lamps are a combination of Argon/Neon. The Neon lines cover the entire visible spectrum, but the Argon lines are useful especially on the blue end of the spectrum where Neon lines become sparse*

Inside here are a number the functions:

	a) intial_wl
		This function helps us determine and initial "guess"
		wavelength solution using only a list of associated pixel
		values and wavelengths of the associated line. 

**Note: This function since it only uses guesses *does not* discriminate between Neon or Argon lines. The list of guessed pixels and wavelengths can combine these two**

	b) wavelength_solver
		This function is the bulk of this file, and it takes in a
		ton of the previously made variables and returns a 
		wavelength solution. 

**Note: This function since it requires a split between Neon or Argon line-pixel guess pairs. Without this, the function cannot fit properly**

~~**Note 2: As of**~~

	commit 758a5dd
 
~~**This function can only fit linear wavelength solutions. This should be amended soon and allow for slightly curved two-term polynomial or higher solutions**~~

**Addendum to Note 2: There are now 3 separate wavelength_solver functions in wavelengths.py as of**

	commit e8d0a77
 
**wavelength_lin_solver fits a linear solution to a pair of Neon/Argon lines; wavelength_polynomial_solver does the same but with a 2-term polynomial; wavelength_argon_solver meanwhile only fits a selection of Argon lines to either a polynomial or linear solution. This solution is not *clean* but it will work for the time being**
