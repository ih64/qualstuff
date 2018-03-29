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
        self.centRegPath = os.path.join(self.basePath, 'regfiles',self.field+self.subfield+'Center.reg')
        self.imagePath = os.path.join(self.basePath, 'Stacks',self.field, self.subfield, 'dlsmake.fits')
        
        self.wcs = self._getWCS(self.imagePath)
        self.regionList = self._parseExReg(self.exRegPath)
        self.regionList += self._parseEdgeReg(self.edgeRegPath)
        self.regionCenter = self._readCenterReg(self.centRegPath)[0]
        
    def _getWCS(self,filename):
        hdu = fits.open(filename)
        hdu.close()
        w = WCS(hdu[0].header)
        return w
    
    def _readCenterReg(self,filename):
        cntReg = regions.read_ds9(filename)
        return cntReg

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
                                width=Angle(.265*np.float(w), 'arcsec'),
                                height=Angle(.265*np.float(h), 'arcsec'),
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

    def interiorPoints(self, ra, dec, unit):
        containsMask = self.regionCenter.contains(SkyCoord(ra, dec, unit=unit), self.wcs)
        #only keep points inside the center region
        #were cutting out any points that land in gutters here
        return containsMask


def calcProbes(table, debug=False):
    '''
    given a astropy table with ra and dec columns, compute w of theta and make a plot
    '''

    xi_list = []
    sig_list = []
    r_list = []
    Coffset_list = []
    
    
    for subfield in ('p11','p12', 'p13', 'p21', 'p22', 'p23', 'p31', 'p32', 'p33'):

        #iterate sources subfield by subfield
        subTable = table[table['subfield'] == 'F2'+subfield]

        #make a random catalog so we can compute the 2 point correlation
        rand_ra, rand_dec = genRandoms(subTable['alpha']*(np.pi/180), subTable['delta']*(np.pi/180))

        #sanatize randoms
        ez = ExclusionZones('F2',subfield)
        rand_ra, rand_dec = ez.flagPoints(rand_ra, rand_dec, unit='rad')
        #deal w gutter regions
        randsInsideMask = ez.interiorPoints(rand_ra, rand_dec, unit='rad')
        rand_ra = rand_ra[randsInsideMask]
        rand_dec = rand_dec[randsInsideMask]

        #sanitize data
        dataInsideMask = ez.interiorPoints(subTable['alpha'], subTable['delta'], unit='deg')
        #deal w gutter regions
        subTable = subTable[dataInsideMask]
        #create the treecorr data catalog
        cat = astpyToCorr(subTable)

        #calculate w of theta given our sanitized randoms and catalog data
        xi, sig, r, Coffset = getWTheta(cat, rand_ra, rand_dec)
        xi_list.append(xi)
        sig_list.append(sig)
        r_list.append(r)
        Coffset_list.append(Coffset)

        if debug:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
            ax1.scatter(cat.ra * 180/np.pi, cat.dec * 180/np.pi, color='blue', s=0.1)
            ax1.scatter(rand_ra* 180/np.pi,
                rand_dec* 180/np.pi, color='green', s=0.1)
            ax1.set_xlabel('RA (degrees)')
            ax1.set_ylabel('Dec (degrees)')
            ax1.set_title('Randoms on top of data')

            # Repeat in the opposite order
            ax2.scatter(rand_ra * 180/np.pi,
                rand_dec * 180/np.pi, color='green', s=0.1)
            ax2.scatter(cat.ra * 180/np.pi, cat.dec * 180/np.pi, color='blue', s=0.1)
            ax2.set_xlabel('RA (degrees)')
            ax2.set_ylabel('Dec (degrees)')
            ax2.set_title('Data on top of randoms')

            plt.show()

        print('finished %s' % subfield)

    return {"xi":xi_list, "sig":sig_list, "r":r_list,
        "rand_ra": rand_ra, "rand_dec": rand_dec, "Coffset":Coffset_list}

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
    """
    turn an astropy table into a treecorr catalog
    anticipating certain format form astropy catalog
    """
    cat = treecorr.Catalog(ra=table['alpha'].data, dec=table['delta'].data,
                         ra_units='deg', dec_units='deg', g1=table['e1'], g2=table['e2'])
    return cat

def calcC(RR):
    NN = RR.weight
    theta = np.exp(RR.meanlogr)

    numerator = NN*np.power(theta, -.8)

    C = numerator.sum()/NN.sum()
    return C


def getWTheta(cat, rand_ra, rand_dec):
    """
    calculate the angular two point correlation function using the landay-sazlay estimator

    note: rand_ra and rand_dec should sample the same space on the sky as the data
        to accurately calculate w of theta
    
    parameters
    cat: treecorr catalog of galaxies we will calculate w of theta for.
    rand_ra: numpy array. uniformly random sampled coordinates in RA space. 
    rand_dec: numpy array. uniformly random sampled coordinates in DEC space

    returns:
    xi: numpy array. the angular two point correlation function
    sig: numpy array. xi's std dev noise estimated from treecor. underestimated error 
    r: numpy array of angular bins xi is calculated for
    """

    dd = treecorr.NNCorrelation(min_sep=0.1, max_sep=50, nbins=15, sep_units='arcmin')
    dd.process(cat)
    rand = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='radians', dec_units='radians')
    rr = treecorr.NNCorrelation(min_sep=0.1, max_sep=50, nbins=15, sep_units='arcmin')
    rr.process(rand)

    r = np.exp(dd.meanlogr)

    dr = treecorr.NNCorrelation(min_sep=0.1, max_sep=50, nbins=15, sep_units='arcmin')
    dr.process(cat, rand)

    xi, varxi = dd.calculateXi(rr, dr)
    sig = np.sqrt(varxi)

    Coffset = calcC(rr)
    return xi, sig, r, Coffset

def getGGL(lensCat, sourceCat):
    """
    calculate galaxy galaxy lensing

    parameters
    lensCat: TreeCorr catalog of lens galaxies. must have positions and shear specified 
    sourceCat: TreeCorr catalog of source galaxies. must have positions and shear specified

    returns
    GGL : galaxy galaxy lens treecorr object. 
        holds information on tangentail shear for lenses
    nullGGL : galaxy galaxy lens treecorr object. 
        swap shear and lens planes and calculate tangential shear
        nice null test for photo-zs
    """
    GGL = treecorr.NGCorrelation(min_sep=0.1, max_sep=100, nbins=20, sep_units='arcmin')
    GGL.process(lensCat, sourceCat)

    nullGGL = treecorr.NGCorrelation(min_sep=0.1, max_sep=100, nbins=20, sep_units='arcmin')
    nullGGL.process(sourceCat, lensCat)
    return GGL, nullGGL

def getKmeans(ra, dec, n_clusters=16):
    """
    use kmeans algorithm to find clusters in data
    useful for jacknife

    input:
    ra: numpy array. ra positions for data
    dec: numpy array. dec positions for data

    returns:
    kmeans: numpy array, same length as ra and dec. 
    tells you which cluster a point in ra,dec
    belongs too
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(np.stack([ra, dec], axis=-1))
    return kmeans
    

def makePlot(xi, sig, r):
    plt.style.use('seaborn-poster')
    plt.plot(r, xi, color='blue')
    plt.plot(r, -xi, color='blue', ls=':')
    plt.errorbar(r[xi>0], xi[xi>0], yerr=sig[xi>0], color='blue', ls='')
    plt.errorbar(r[xi<0], -xi[xi<0], yerr=sig[xi<0], color='blue', ls='')
    leg = plt.errorbar(-r, xi, yerr=sig, color='blue')

    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.xlabel(r'$\theta$ (arcmin)')

    plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
#    plt.xlim([0.01,2])
    plt.show()
    return

        