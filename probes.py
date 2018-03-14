import treecorr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from astropy.io import ascii
import pickle
import os
import regions
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import KMeans

class ExclusionZones():

    def __init__(self, field, subfield, basePath='/Users/ih64/Desktop/qualstuff/'):
        self.basePath = basePath
        self.field = field
        self.subfield = subfield
        self.exRegPath = os.path.join(self.basePath, 'regfiles',self.field+self.subfield+'R.all.2009.reg')
        self.edgeRegPath = os.path.join(self.basePath, 'regfiles',self.field+self.subfield+'.edge.reg')
        self.imagePath = os.path.join(self.basePath, 'Stacks',self.field, self.subfield, 'dlsmake.fits')
        
        self.wcs = self._getWCS(self.imagePath)
        self.regionList = self._parseExReg(self.exRegPath)
        self.regionList += self._parseEdgeReg(self.edgeRegPath)
        
    def _getWCS(self,filename):
        hdu = fits.open(filename)
        hdu.close()
        w = WCS(hdu[0].header)
        return w
    
    def _parseExReg(self,filename):

        with open(filename,'r') as f:
            lines = f.readlines()

        reglist = []

        for line in lines:
            if not line.startswith('#'):
                if line.startswith('circle'):
                    baseLine = line.partition('# color=green pos=')
                    x, y, r = baseLine[0][baseLine[0].find('(') +1 : baseLine[0].find(')')].split(',')
                    ra, dec = baseLine[-1][baseLine[-1].find('(') +1 : baseLine[-1].find(')')].split(',')
                    circ = regions.CircleSkyRegion(center=SkyCoord(np.float(ra), np.float(dec), unit='deg'),
                                 radius=Angle(.265*np.float(r), 'arcsec'))
                    reglist.append(circ)

                elif line.startswith('box'):
                    baseLine = line.partition('# color=green ')
                    x, y, w, h = baseLine[0][baseLine[0].find('(') +1:baseLine[0].find(')')].split(',')
                    ra, dec = baseLine[-1].rstrip().split()
                    ra = np.float(ra[3:])
                    dec = np.float(dec[4:])
                    rec = regions.RectangleSkyRegion(center=SkyCoord(ra, dec, unit='deg'),
                                       width=Angle(.265*np.float(w), 'arcsec'), height=Angle(.265*np.float(h), 'arcsec'),
                                       angle=Angle(270, 'deg'))
                    reglist.append(rec)

        return reglist
    
    def _parseEdgeReg(self,filename):
        pixRegions = regions.read_ds9(filename)
        skyRegions = [r.to_sky(self.wcs) for r in pixRegions]
        return skyRegions
    
    def flagPoints(self, ra, dec, unit='rad'):
        containsList = [i.contains(SkyCoord(ra, dec, unit=unit), self.wcs) for i in self.regionList]
        #tells you indx of ra dec coords that fall inside any regions
        pointMask = np.where(~np.any(containsList, axis=0))
        
        ra_clean, dec_clean = ra[pointMask], dec[pointMask]
        return ra_clean, dec_clean

def makeWTheta(table, debug=False):
    '''
    given a astropy table with ra and dec columns, compute w of theta and make a plot
    '''

    #make a random catalog so we can compute the 2 point correlation
    rand_ra, rand_dec = genRandoms(table['alpha']*(np.pi/180), table['delta']*(np.pi/180))
    
    
    #sanitize randoms
    for subfield in ('p11','p12', 'p13', 'p21', 'p22', 'p23', 'p31', 'p32', 'p33'):
        ez = ExclusionZones('F2',subfield)
        rand_ra, rand_dec = ez.flagPoints(rand_ra, rand_dec, unit='rad')
        print('finished %s' % subfield)
    
    #create the treecorr data catalog
    cat = astpyToCorr(table)

    #calculate w of theta given our sanitized randoms and catalog data
    xi, varxi, sig, r = doTreeCorr(cat, rand_ra, rand_dec)
    
    if debug:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
        ax1.scatter(cat.ra * 180/np.pi, cat.dec * 180/np.pi, color='blue', s=0.1)
        ax1.scatter(rand_ra * 180/np.pi, rand_dec * 180/np.pi, color='green', s=0.1)
        ax1.set_xlabel('RA (degrees)')
        ax1.set_ylabel('Dec (degrees)')
        ax1.set_title('Randoms on top of data')

        # Repeat in the opposite order
        ax2.scatter(rand_ra * 180/np.pi, rand_dec * 180/np.pi, color='green', s=0.1)
        ax2.scatter(cat.ra * 180/np.pi, cat.dec * 180/np.pi, color='blue', s=0.1)
        ax2.set_xlabel('RA (degrees)')
        ax2.set_ylabel('Dec (degrees)')
        ax2.set_title('Data on top of randoms')

    plt.show()

    return {"xi":xi, "varxi": varxi, "sig":sig, "r":r, "rand_ra": rand_ra, "rand_dec": rand_dec}

def genRandoms(ra, dec, debug=True):
    ra_min = np.min(ra)
    ra_max = np.max(ra)
    dec_min = np.min(dec)
    dec_max = np.max(dec)
    ntot = ra.size

    if debug:
        print('ra range = %f .. %f' % (ra_min, ra_max))
        print('dec range = %f .. %f' % (dec_min, dec_max))

    rand_ra = np.random.uniform(ra_min, ra_max, 6*ntot)
    rand_sindec = np.random.uniform(np.sin(dec_min), np.sin(dec_max), 6*ntot)
    rand_dec = np.arcsin(rand_sindec)
    return rand_ra, rand_dec

def astpyToCorr(table):
    cat = treecorr.Catalog(ra=table['alpha'].data, dec=table['delta'].data,
                         ra_units='deg', dec_units='deg', g1=table['e1'], g2=table['e2'])
    return cat

def doTreeCorr(cat, rand_ra, rand_dec):
    dd = treecorr.NNCorrelation(min_sep=0.01, max_sep=2, bin_size=0.2, sep_units='degrees')
    dd.process(cat)
    rand = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='radians', dec_units='radians')
    rr = treecorr.NNCorrelation(min_sep=0.01, max_sep=2, bin_size=0.2, sep_units='degrees')
    rr.process(rand)

    r = np.exp(dd.meanlogr)

    dr = treecorr.NNCorrelation(min_sep=0.01, max_sep=2, bin_size=0.2, sep_units='degrees')
    dr.process(cat, rand)

    xi, varxi = dd.calculateXi(rr, dr)
    sig = np.sqrt(varxi)
    return xi, varxi, sig, r

def makePlot(xi, varxi, sig, r):
    plt.style.use('seaborn-poster')
    plt.plot(r, xi, color='blue')
    plt.plot(r, -xi, color='blue', ls=':')
    plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', lw=0.1, ls='')
    plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', lw=0.1, ls='')
    leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$\theta$ (degrees)')

    plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
    plt.xlim([0.01,2])
    plt.show()
    return

        