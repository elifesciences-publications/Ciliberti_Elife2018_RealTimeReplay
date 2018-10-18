import numpy as np
import scipy as sp
from os import path
import abc

from compressed_decoder.gmmcompression.fkmixture import MixtureClass as Mixture
import fklab.segments as seg

        
def load_mixture( filepath ):
    '''
        Returns a mixture saved on disk
    '''
    m = Mixture()
    return m.fromfile( filepath )

def save_mixture( density, filepath ):
    '''
        Save the gmm mixture containied in a compressed density
    '''
    m = density._mixture
    m.tofile( filepath )

def create_covariance_array( bw_behav_cm, bw_ampl_mV, n_wires=4 ):
    """Returns array of covariances for compressed gmm
    """
    
    covars = [ bw_behav_cm**2 ]
    for i in range(n_wires):
        covars.append( (bw_ampl_mV * 1000)**2 )
        
    return np.array( covars )


def scaling_factor_trunc_gauss( h, cutoff=0.05 ):
    """Returns scaling factor for trunated gaussian kernel.
    
    Parameters
    ----------
    h : 1D array-like
        vector of standard deviations
    cutoff : float
        probability to cut off
    
    Returns
    -------
    scale : float
        scaling factor
    cutoff: float
        cut off in standard deviations
    
    """
    h = np.array( h, copy=False ).ravel()
    n = len(h)
    s = (2.0*np.pi)**(-n/2.0)
    s /= np.product(h)
    s /= (1-cutoff)**n
    cutoff = -sp.stats.norm.ppf( cutoff/2.0 ) #cutoff in st.dev.
    return s, cutoff


#density class minimum interface:
# property: ndim
# method: evaluate( points )
# method: evaluate_grid( grid, points )
# method: evaluate_marginal( points, dim )

class DensityBase(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def ndim(self):
        pass
    
    @abc.abstractproperty
    def nsamples(self):
        pass
    
    @abc.abstractmethod
    def evaluate(self, points):
        pass
    
    @abc.abstractmethod
    def evaluate_grid(self, grid, points):
        pass
    
    @abc.abstractmethod
    def evaluate_marginal(self, points, dim):
        pass

class MergingCompressionDensity(DensityBase):
    
    def __init__(self, samples=None, ndim=None, sample_covariance = 1.0,\
        method='bandwidth', threshold=1.0, name='unspecified'):
        #samples: (nsamples,ndim) array
        
        if samples is None and ndim is None:
            ndim = 1 #by default one dimensional density
            nsamples = 0
        elif ndim is None:
            samples = np.array( samples )
            if samples.ndim<2:
                samples = samples.reshape( (samples.size,1) )
            elif samples.ndim>2:
                raise ValueError
            nsamples, ndim = samples.shape
        else:
            nsamples = 0
        
        self._ndim = ndim
        
        self._mixture = Mixture( ndim=ndim )
        
        self.method = method
        self.threshold = threshold
        self.sample_covariance = sample_covariance
        
        if nsamples>0:
            self.addsamples( samples )
        
        self.name = name
        
    def ncomponents(self):
        return self._mixture.ncomponents
    
    def addsamples(self, samples):
        
        samples = np.array( samples )
        if samples.ndim<2:
            samples = samples.reshape( (samples.size,1) )
        elif samples.ndim>2:
            raise ValueError
        nsamples, ndim = samples.shape
        
        if ndim!=self._ndim:
            raise ValueError
        
        samples = np.ascontiguousarray( samples )
        
        if self._method in ['full','bandwidth','constant']:
            self._mixture.merge_samples( samples, threshold=self._threshold, covariance_match = self._method )
        else:
            self._mixture.add_samples( samples )
    
    def evaluate(self, points ):
        
        return self._mixture.evaluate( np.ascontiguousarray(points) )
    
    def evaluate_grid( self, grid, points ):
        
        return self._mixture.evaluate_grid(  np.ascontiguousarray(grid), np.ascontiguousarray(points) )
    
    def grid_accumulator( self, grid ):
        
        return self._mixture.build_grid_accumulator( np.ascontiguousarray(grid) )
    
    def evaluate_grid_multi( self, grid, points ):
        
        grid_acc = self._mixture.build_grid_accumulator( np.ascontiguousarray(grid) )
        return self._mixture.evaluate_grid_multi( grid_acc, np.ascontiguousarray(points) )
    
    def evaluate_marginal(self, points, dims):
        m = self._mixture.marginalize( dims )
        x = m.evaluate( points )
        return x
    
    def load_from_mixturefile( self, filepath ):
        '''
            Load densitiy using an available mixture
            
            method = string of the method sp
        '''
        if not path.isfile( filepath ):
            raise ValueError("Path to mixture does not exist.")
        m = load_mixture( filepath )

        self._mixture = m
        
        if m.ndim < 1:
            raise ValueError("Loaded mixture has zero dimensions.")
            
        self._ndim = m.ndim        
        
#        if len(covariance) < m.ndim:
#            raise ValueError("Covariance has too few dimensions.")
            
#        self.sample_covariance = covariance[:m.ndim]          
        self.ncomponents = m.ncomponents
    
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self,value):
        value = str(value)
        if value not in ['none','full','bandwidth','constant']:
            raise ValueError
        
        self._method = value
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self,value):
        value = float(value)
        if value<0:
            raise ValueError
        
        self._threshold = value
    
    @property
    def ndim(self):
        return self._mixture.ndim
    
    @property
    def nsamples(self):
        return self._mixture.sum_n
        
    @property
    def n_samples(self):
        return 
    
    @property
    def sample_covariance(self):
        #TODO: add method to get sample covariance as numpy array to mixture class
        return self._sample_covariance
    
    @sample_covariance.setter
    def sample_covariance(self, value):
        value = np.asarray( value, dtype=np.float64 )
        if value.size==1:
            value = float(value) * np.diag( np.ones( self._ndim ) )
        elif value.ndim==1 and len(value)==self._ndim:
            value = np.diag( value )
        elif value.ndim==2 and value.shape[0]==self._ndim and value.shape[1]==self._ndim:
            pass
        else:
            raise ValueError
        
        self._mixture.set_sample_covariance( value )
        
        self._sample_covariance = value


class DistanceLUT:
    def __init__(self, index, lut):
        self._index = index
        self._lut = lut
        #TODO: check index vector and lut array
        
    def convert2index(self, x, kind='nearest'):
        return sp.interpolate.interp1d( range(len(self._index)), self._index, kind=kind )(x)
    
    def convert2value(self, x):
        return self._index[ np.int(x) ]
    
    @property
    def lut(self):
        return self._lut
        

class KDEDecoder(object):
    def __init__(self, behaviordata, spikedata, trainingtime, behavior_grid, offset=0):
        #behaviordata: MergingCompressionDensity object
        #spikedata: List of MixeKDE objects, where covariates are positioned in first dimensions
        #spikedata must have ndim >= behaviordata ndim
        #kernels, bandwidths, datatypes, distance for covariate dimensions should be the same for behaviordata and spikedata
        
        if not isinstance(behaviordata,DensityBase):
            raise ValueError
        
        if isinstance(spikedata, DensityBase):
            spikedata = [spikedata]
        
        if not all( isinstance(x,DensityBase) for x in spikedata ):
            raise ValueError
        
        self._behaviordata = behaviordata
        self._spikedata = spikedata
        
        if trainingtime < 0.1:
            raise ValueError("Training time is too low")
        self._trainingtime = trainingtime
        self._behavior_grid = behavior_grid
        self._offset = offset
        
        self._occupancy = self._behaviordata.evaluate( self._behavior_grid )
        self._mu = [ x.nsamples/self._trainingtime for x in self._spikedata ]
        
        self._spike_rate = [ x.evaluate_marginal( self._behavior_grid, np.arange(behavior_grid.shape[1], dtype=np.uint16) ) for x in self._spikedata ]
        self._spike_rate = [self._offset + x*y/self._occupancy for x,y in zip(self._mu,self._spike_rate)]
        
        self._spike_rate_sum = sum( self._spike_rate )
        
    
    def decode_single_bin(self, spikedata=[], binsize=0.25):
        
        p = [x.evaluate_grid(self._behavior_grid,y) if len(y)>0 else np.zeros( (0,len(self._behavior_grid)) ) for x,y in zip(self._spikedata,spikedata)]
        p = [np.sum( np.log( x + self._offset*self._occupancy/y ), axis=0 ) for x,y in zip( p,self._mu) ]
        
        nspikes = np.sum( [ x.shape[0] for x in spikedata ] )
        
        P = sum( p )
        
        logposterior = P - np.log(self._occupancy)*nspikes - binsize*self._spike_rate_sum
        
        posterior = np.exp( logposterior - np.max(logposterior) )
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    
    def decode(self, spikedata, spiketime, bins):
        
        #spikedata is list of arrays with size Nspikes x Ndim
        #spiketime is list of vectors of size Nspikes
        #bins is segment like
        
        bins = seg.Segment( bins )
        if bins.hasoverlap():
            raise ValueError("Bins have overlap")
        
        posterior = np.zeros( (len(self._behavior_grid), len(bins)), dtype=np.float64 )
        
        total_n = 0.
        
        #loop through each source
        for t,data,density,mu in zip(spiketime,spikedata,self._spikedata,self._mu):
            
            #select all data in requested time bins
            #`inseg` is True(False) for each spike time that is (not) contained within a bin
            #`n` is for each bin the number of spike times it contains
            inseg, n, _ = bins.contains( t )
            
            total_n += n
            
            d = data[inseg]
            
            cumulative_spikes = np.cumsum(n)
            
            #process spikes in blocks
            maxblocksize = round(5000000. / len(self._behavior_grid)) #limit the size of the output array to 5M elements
            
            startbin = 0
            n_processed_spikes = 0
            
            offset = self._offset * self._occupancy / mu
            
            while startbin<len(bins):
                
                #find next bin, such that number of spike from start to end bin is <= maxblocksize
                endbin = np.searchsorted( cumulative_spikes, maxblocksize + n_processed_spikes, 'right' )
                
                start_spike_index = n_processed_spikes
                end_spike_index = cumulative_spikes[ endbin - 1 ]
                
                p = density.evaluate_grid( self._behavior_grid,d[start_spike_index:end_spike_index] )

                p = np.split( p, cumulative_spikes[startbin:endbin]-n_processed_spikes, axis=0 )

                p = [ np.sum( np.log( x + offset ), axis=0 ) if len(x)>0 else 0. for x in p[:-1] ]
                
                for idx,kk in enumerate( range(startbin,endbin) ):
                    posterior[:,kk] += p[idx]
                
                startbin = endbin
                n_processed_spikes = end_spike_index
                
            
        posterior = posterior - np.log(self._occupancy[:,None])*total_n[None,:]\
            - self._spike_rate_sum[:,None]*bins.duration[None,:]
        
        posterior = np.exp( posterior - np.max(posterior,axis=0) )
        posterior = posterior / np.sum(posterior,axis=0)
        
        return posterior


class Decoder(object):
    
    def __init__(self, behavior_density, joint_density, trainingtime, behavior_grid, offset=0):
        #behavior_density: MergingCompressionDensity object
        #joint_density: List of MergingCompressionDensity objects, where covariates are positioned in first dimensions
        #joint_density must have ndim >= behaviordata ndim
        #kernels, bandwidths, datatypes, distance for covariate dimensions should be the same for behaviordata and spikedata
        
        if not isinstance(behavior_density, MergingCompressionDensity):
            raise ValueError
        
        if not all( isinstance(x,DensityBase) for x in joint_density ):
            raise ValueError
        
        self.behavior_density = behavior_density
        self.joint_density = joint_density
        self.T = trainingtime
        self.grid = behavior_grid
        self.offset = offset
        self.last_pax = None
        
        self.uncorrected_pix = self.behavior_density.evaluate( self.grid )
        # self.mu = [ tt.shape[1]/self.T for tt in self.joint_density ]
        self.mu = np.array( [ tt.nsamples/self.T for tt in self.joint_density ] )
        
        self.px = np.array( [ d.evaluate_marginal( self.grid, np.array([0], dtype=np.uint16) )\
                             if d.ncomponents > 0 else np.zeros(len(behavior_grid)) for d in joint_density ] )
        
        self.last_split_log_pax = None
        self.last_log_pax = None  
        self.last_test_spikes = None
        self.offset_pix = None
        
        self._lx_computed = False
        self._lx = None
        self._pix_computed = False
        self._pix = None
        
        
    def pix( self, factor=12 ):
        '''
            compute occupancy adjusting for unoccupied points
        '''
        
        if not self._pix_computed:
            u_pix = self.uncorrected_pix
            offset_pix = u_pix.mean()/factor
            self.offset_pix = offset_pix
            self._pix = u_pix + offset_pix
            self._pix_computed = True
        
        return self._pix
        
    def lx( self ):
        '''
            compute spike rate
        '''
        if not self._lx_computed:
            self._lx = np.array( [ m*p_x/self.pix() + self.offset for m,p_x in zip(self.mu, self.px)])   
            self._lx_computed = True
        
        return self._lx  
        
    def pax( self, test_spikes, tt_included ):
        '''
            test_spikes contains only the spike amplitudes
        '''
        pax = []
        j = 0
        for i, d in enumerate(self.joint_density):
            if tt_included[i]:
                if len(test_spikes[j].T) > 0:
                    pax.append( d.evaluate_grid_multi( self.grid, test_spikes[j][:-1].T ) )
                else:
                    pax.append( np.empty( (0, len(self.grid)) ) )    
                j += 1
        self.last_pax = pax        
        return pax
        
    def decode_single_bin(self, spikedata=[], binsize=0.25):
        
        p = [x.evaluate_grid(self._behavior_grid,y) if len(y)>0 else\
            np.zeros( (0,len(self._behavior_grid)) ) for x,y in zip(self._spikedata,spikedata)]
        p = [np.sum( np.log( x + self._offset*self._occupancy/y ), axis=0 ) for x,y in zip( p,self._mu) ]
        
        nspikes = np.sum( [ x.shape[0] for x in spikedata ] )
        
        P = sum( p )
        
        logposterior = P - np.log(self._occupancy)*nspikes - binsize*self._spike_rate_sum
        
        posterior = np.exp( logposterior - np.max(logposterior) )
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    
    def decode( self, spike_features, bins, tt_included, spike_ampl_mask, bin_size,\
    sf_keys=["time", "value"] ):
        """
        
        Parameters
        ----------
         spike_features : list of of dictionaries
             contains all spike features of the dataset
        bins : list of Segments
            time bins to be used to decoding
        tt_included : list of booleans
        spike_ampl_mask : list of 1d boolean arrays
        sf_keys : 2-el list of string of the dictionaries
            the 1st refers to time, the 2nd one to spike amplitudes
         
        Returns
        -------
        posterior, logpos, n_spikes, n_tt
         
         """
        
        bins = seg.Segment( bins )
        if bins.hasoverlap():
            raise ValueError("Bins have overlap")
            
        assert( len(spike_ampl_mask) <= len(tt_included) )
        assert( len(spike_features) == len(tt_included) )   
        assert( len(spike_ampl_mask) == sum(tt_included) )         
            
        # check bin size
        bin_sizes = np.unique( bins.duration[None,:] )
        if not np.all( np.isclose( bin_sizes, bin_size) ):
            raise ValueError("Not all bins have the requested bin size")

        test_spikes = []
        n_spikes = []
        j = 0
        
        for i, tt in enumerate( spike_features ):    
            
            if tt_included[i]:
                
                sel_test_spikes = bins.contains(tt[sf_keys[0]])[0]
                second = tt[sf_keys[0]][sel_test_spikes] 
                # carefully handle single-amplitude case
                if len(spike_ampl_mask[j]) == 1 and spike_ampl_mask[j][0] == True:
                    first = np.expand_dims( tt[sf_keys[1]][sel_test_spikes].T, axis=0 )
                else:
                    first = tt[sf_keys[1]][sel_test_spikes].T[spike_ampl_mask[j]]
                test_spikes.append( np.vstack( (first, second) ) )  
                n_spikes.append( bins.contains(tt[sf_keys[0]])[1] )
                j += 1
                
        n_spikes = np.array(n_spikes)
        self.last_test_spikes = test_spikes        
        
        n_tt = len(test_spikes) # all tetrodes with enough encoding spikes
        pax = self.pax( test_spikes, tt_included )
        log_pax = np.array( [ np.log( _pax + self.offset * self.pix() / _mu )\
                             for _pax, _mu in zip(pax, self.mu[tt_included]) ] )
        self.last_log_pax = log_pax
        cs = np.cumsum( n_spikes, axis=1 )
        split_log_pax = [ np.split( log_pax[i], cs[i, :-1] ) for i in range(n_tt) ]
        self.last_split_log_pax = split_log_pax
        nbins = cs.shape[1]
        logpos_spikes = np.zeros( (nbins, len(self.grid)) )
        logpos_sources = np.zeros_like(logpos_spikes)
        for bin_index in range(nbins):
            for s in range(n_tt):
                logpos_spikes[bin_index, :] += np.sum(split_log_pax[s][bin_index], axis=0)
            logpos_sources[bin_index, :] = \
                np.log( self.pix() ) * np.sum( n_spikes, axis=0 )[bin_index] +\
                    bin_size * np.sum( self.lx()[tt_included], axis=0 ) # tt_included not necessary if offset is very small
        logpos = logpos_spikes - logpos_sources
        
        # get true posteriors (i.e. normalize the distribution)
        posterior = np.exp( logpos - np.nanmax( logpos, axis=1, keepdims=True ) )
        posterior = posterior / np.nansum(posterior, axis=1, keepdims=True )

        return posterior, logpos, n_spikes, n_tt
    
    
    def lazy_decode( self, spike_features, bins, tt_included, spike_ampl_mask, bin_size,\
    sf_keys=["time", "value"] ):
        """
        
        Use last decoded pax to generate posteriors.
        
        Parameters
        ----------
         spike_features : list of of dictionaries
             contains all spike features of the dataset
        bins : list of Segments
            time bins to be used to decoding
        tt_included : list of booleans
        spike_ampl_mask : list of 1d boolean arrays
        sf_keys : 2-el list of string of the dictionaries
            the 1st refers to time, the 2nd one to spike amplitudes
         
        Returns
        -------
        posterior, logpos, n_spikes, n_tt
         
         """
        
        if self.last_split_log_pax is None:
            raise ValueError("pax has never been computed")
        
        bins = seg.Segment( bins )
        if bins.hasoverlap():
            raise ValueError("Bins have overlap")
            
        assert( len(spike_ampl_mask) <= len(tt_included) )
        assert( len(spike_features) == len(tt_included) )   
        assert( len(spike_ampl_mask) == sum(tt_included) )         
            
        # check bin size
        bin_sizes = np.unique( bins.duration[None,:] )
        if not np.all( np.isclose( bin_sizes, bin_size) ):
            raise ValueError("Not all bins have the requested bin size")

        test_spikes = []
        n_spikes = []
        j = 0
        
        for i, tt in enumerate( spike_features ):    
            
            if tt_included[i]:
                
                sel_test_spikes = bins.contains(tt[sf_keys[0]])[0]
                second = tt[sf_keys[0]][sel_test_spikes] 
                # carefully handle single-amplitude case
                if len(spike_ampl_mask[j]) == 1 and spike_ampl_mask[j][0] == True:
                    first = np.expand_dims( tt[sf_keys[1]][sel_test_spikes].T, axis=0 )
                else:
                    first = tt[sf_keys[1]][sel_test_spikes].T[spike_ampl_mask[j]]
                test_spikes.append( np.vstack( (first, second) ) )  
                n_spikes.append( bins.contains(tt[sf_keys[0]])[1] )
                j += 1
                
        n_spikes = np.array(n_spikes)       
        
        n_tt = len(test_spikes) # all tetrodes with enough encoding spikes
        cs = np.cumsum( n_spikes, axis=1 )
        nbins = cs.shape[1]
        logpos_spikes = np.zeros( (nbins, len(self.grid)) )
        logpos_sources = np.zeros_like(logpos_spikes)
        for bin_index in range(nbins):
            for s in range(n_tt):
                logpos_spikes[bin_index, :] += np.sum(self.last_split_log_pax[s][bin_index], axis=0)
            logpos_sources[bin_index, :] = \
                np.log( self.pix() ) * np.sum( n_spikes, axis=0 )[bin_index] +\
                    bin_size * np.sum( self.lx()[tt_included], axis=0 ) # tt_included not necessary if offset is very small
        logpos = logpos_spikes - logpos_sources
        
        posterior = np.exp( logpos - np.nanmax( logpos, axis=1, keepdims=True ) )
        posterior = posterior / np.nansum(posterior, axis=1, keepdims=True )

        return posterior, logpos, n_spikes, n_tt