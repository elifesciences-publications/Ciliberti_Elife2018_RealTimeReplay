import numpy as np
cimport numpy as np

from libc.stdint cimport uint32_t, uint16_t, uint8_t

cdef extern from "covariance.h":
    ctypedef struct CovarianceMatrix:
        uint32_t refcount
        uint16_t ndim
    
    CovarianceMatrix* covariance_create( uint16_t ndim )
    void covariance_delete( CovarianceMatrix* cov )
    
    void covariance_set_full( CovarianceMatrix* cov, double* values )
    void covariance_print( CovarianceMatrix* cov )
    
cdef extern from "component.h":
    ctypedef struct GaussianComponent:
        uint32_t refcount
        uint16_t ndim
        double weight
        double *mean
        CovarianceMatrix *covariance
    
    GaussianComponent* component_create_empty( uint16_t ndim )
    void component_delete( GaussianComponent* component )
    void component_get_mean( GaussianComponent* component, double* mean)
    void component_get_covariance_array( GaussianComponent* component, double* covariance)


cdef extern from "mixture.h":
    ctypedef struct Mixture:
        uint32_t refcount
        double sum_n
        double sum_weights
        uint16_t ndim
        uint32_t ncomponents
        uint32_t buffersize
        GaussianComponent **components
        CovarianceMatrix *samplecovariance
    
    Mixture* mixture_create( int )
    void mixture_delete( Mixture* mixture)
    
    int mixture_save_to_file( Mixture* mixture, char* filename )
    Mixture* mixture_load_from_file( char* filename )
    void mixture_update_cache( Mixture* mixture )    
    
    void mixture_set_samplecovariance( Mixture* mixture, CovarianceMatrix* cov)
    CovarianceMatrix* mixture_get_samplecovariance( Mixture* mixture)
    void mixture_addsamples( Mixture* mixture, double* means, uint32_t nsamples, uint16_t ndim )
    
    void mixture_evaluate( Mixture* mixture, double* points, uint32_t npoints, double* result)
    void mixture_evaluategrid( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* points, uint32_t npoints, uint16_t npointdim, uint16_t* pointdim, double* result )
    
    void mixture_evaluate_diagonal( Mixture* mixture, double* points, uint32_t npoints, double* result)
  
    void mixture_evaluategrid_diagonal( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* testpoint, uint16_t npointsdim, uint16_t* pointsdim, double* output )
    void mixture_prepare_grid_accumulator( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* grid_acc )
    void mixture_evaluategrid_diagonal_multi( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output )
    
    Mixture* mixture_compress( Mixture* mixture, double threshold, uint8_t weighted_hellinger )
    void mixture_merge_samples( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    void mixture_merge_samples_constant_covariance( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    void mixture_merge_samples_match_bandwidth( Mixture* mixture, double* means, uint32_t npoints, double threshold )
    
    void mixture_get_means( Mixture* mixture, double* means )
    void mixture_get_scaling_factors( Mixture* mixture, double* scales )
    void mixture_get_weights( Mixture* mixture, double* weights )
    void mixture_get_covariances( Mixture* mixture, double* covariances )
    
    double compute_distance( GaussianComponent* c1, GaussianComponent* c2 )
    void moment_match( GaussianComponent** components, uint32_t ncomponents, GaussianComponent* output, uint8_t normalize )
    
    Mixture* mixture_marginalize( Mixture* mixture, uint16_t ndim, uint16_t* dims )
    
    double hellinger_single( GaussianComponent** components, uint32_t n, GaussianComponent* model, uint8_t weighted)

cdef class CovarianceClass:
    cdef CovarianceMatrix* _c_covariance
    def __init__(self, np.ndarray[double, ndim=2, mode="c"] data):
        nr = data.shape[0]
        nc = data.shape[1]
        assert nr==nc
        assert nr>0
        
        self._c_covariance = covariance_create(nr)
        covariance_set_full( self._c_covariance, &data[0,0] )
        
        if self._c_covariance is NULL:
            raise MemoryError
    
    def __dealloc__(self):
        if self._c_covariance is not NULL:
            #print("deleting covariance matrix")
            covariance_delete(self._c_covariance)
    
    def show(self):
        covariance_print( self._c_covariance )
    
    @property
    def refcount(self):
        return self._c_covariance.refcount
    
    @property
    def ndim(self):
        return self._c_covariance.ndim
    
        
cdef class ComponentClass:
    cdef GaussianComponent* _c_component
    def __init__(self, ndim=1):
        self._c_component = component_create_empty(ndim)
        if self._c_component is NULL:
            raise MemoryError
    
    def __dealloc__(self):
        if self._c_component is not NULL:
            #print("deleting component")
            component_delete(self._c_component)
    
    @property
    def refcount(self):
        return self._c_component.refcount
    
    @property
    def ndim(self):
        return self._c_component.ndim
    
    @property
    def weight(self):
        return self._c_component.weight
    
    @property
    def mean(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ndim), dtype=np.float64, order="C" )
        component_get_mean(self._c_component, &result[0] )
        return result
    
    @property
    def covariance(self):
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.empty( (self.ndim,self.ndim), dtype=np.float64, order="C" )
        component_get_covariance_array(self._c_component, &result[0,0] )
        return result
    


cdef class MixtureClass:
    cdef Mixture* _c_mixture
    def __cinit__(self, ndim=1):
        if ndim==0:
            self._c_mixture = NULL
        else:
            self._c_mixture = mixture_create(ndim) 
            if self._c_mixture is NULL:
                raise MemoryError
        
    def __dealloc__(self):
        if self._c_mixture is not NULL:
            #print("deleting mixture")
            mixture_delete(self._c_mixture)
    
    def marginalize( self, np.ndarray[uint16_t, ndim=1, mode="c"] dims ):
        result = MixtureClass(ndim=0)
        result._c_mixture = mixture_marginalize( self._c_mixture, len(dims), &dims[0] )
        return result
    
    def hellinger( self, ComponentClass c, index=0, n=0, weighted=1):
        index = int(index)
        n = int(n)
        weighted = int(weighted)
        
        assert index>=0 and index<self.ncomponents
        
        if n==0:
            n = self.ncomponents - index
        
        assert n>0 and n<=(self.ncomponents-index)
        
        assert isinstance( c, ComponentClass )
        
        #cdef GaussianComponent* comp = c._c_component
        
        return hellinger_single( &self._c_mixture.components[index], n, c._c_component, weighted )
        
    
    def distance(self, c1, c2 ):
        c1 = int(c1)
        c2 = int(c2)
        
        assert c1>=0 and c1<self.ncomponents
        assert c2>=0 and c2<self.ncomponents
        
        return compute_distance( self._c_mixture.components[c1], self._c_mixture.components[c2] )
    
    def moment_match_components(self, index=0, n=0, normalize=1 ):
        
        index = int(index)
        n = int(n)
        normalize = int(normalize)
        
        assert index>=0 and index<self.ncomponents
        
        if n==0:
            n = self.ncomponents - index
        
        assert n>0 and n<=(self.ncomponents-index)
        
        c = ComponentClass( self.ndim )
        
        moment_match( &self._c_mixture.components[index], n, c._c_component, normalize )
        
        return c
    
    def set_sample_covariance(self, np.ndarray[double, ndim=2, mode="c"] data):
        
        c = CovarianceClass( data )
        mixture_set_samplecovariance( self._c_mixture, c._c_covariance )
        del c
    
    def add_samples(self, np.ndarray[double, ndim=2, mode="c"] means ):
        
        nsamples = means.shape[0]
        ndim = means.shape[1]
        
        mixture_addsamples( self._c_mixture, &means[0,0], nsamples, ndim )
    
    def compress(self,threshold=0.01, weighted_hellinger=True):
        
        m = mixture_compress( self._c_mixture, threshold, weighted_hellinger )
        result = MixtureClass(ndim=0)
        result._c_mixture = m
        
        return result
    
    def merge_samples( self, np.ndarray[double, ndim=2, mode="c"] samples, threshold=1.0, covariance_match = 'full'):
        nsamples = samples.shape[0]
        ndim = samples.shape[1]
        
        assert ndim==self.ndim
        
        if covariance_match == 'constant':
            mixture_merge_samples_constant_covariance( self._c_mixture, &samples[0,0], nsamples, threshold )
        elif covariance_match == 'bandwidth':
            mixture_merge_samples_match_bandwidth( self._c_mixture, &samples[0,0], nsamples, threshold )
        else: #'full'
            mixture_merge_samples( self._c_mixture, &samples[0,0], nsamples, threshold )
    
    
    def evaluate(self, np.ndarray[double, ndim=2, mode="c"] x, diagonal=False):
        
        npoints = x.shape[0]
        ndim = x.shape[1]
        
        assert ndim==self.ndim
        
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.zeros( npoints, dtype=np.float64, order='C' )
        
        if npoints<1:
            return result
        
        if diagonal:
            mixture_evaluate_diagonal( self._c_mixture, &x[0,0], npoints, &result[0])
        else:
            mixture_evaluate( self._c_mixture, &x[0,0], npoints, &result[0])
        
        return result
        
    def build_grid_accumulator(self, np.ndarray[double, ndim=2, mode="c"] grid):
        
        ngrid = grid.shape[0]
        ngriddim = grid.shape[1]
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] griddim
        griddim = np.arange( ngriddim, dtype=np.uint16 )
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] grid_acc
        grid_acc = np.zeros( (self.ncomponents,ngrid), dtype=np.float64, order="C" )
        
        mixture_prepare_grid_accumulator( self._c_mixture, &grid[0,0], ngrid, ngriddim, &griddim[0], &grid_acc[0,0] )
        
        return grid_acc
    
    def evaluate_grid_multi(self, np.ndarray[double, ndim=2, mode="c"] grid_acc, np.ndarray[double, ndim=2, mode="c"] x ):
        
        ngrid = grid_acc.shape[1]
        
        npoints = x.shape[0]
        npointsdim = x.shape[1]
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.zeros( (npoints,ngrid), dtype=np.float64, order="C" )
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] pointsdim
        pointsdim = np.arange( self.ndim-npointsdim, self.ndim, dtype=np.uint16 )
        
        mixture_evaluategrid_diagonal_multi( self._c_mixture, &grid_acc[0,0], ngrid, &x[0,0], npoints, npointsdim, &pointsdim[0], &result[0,0] )
        
        return result
    
    def evaluate_grid(self, np.ndarray[double, ndim=2, mode="c"] grid_acc, np.ndarray[double, ndim=2, mode="c"] x):
        
        ngrid = grid_acc.shape[1]
        npointsdim = x.shape[1]
        
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.zeros( (1,ngrid), dtype=np.float64, order="C" )
        
        cdef np.ndarray[np.uint16_t, ndim=1, mode="c"] pointsdim
        pointsdim = np.arange( self.ndim-npointsdim, self.ndim, dtype=np.uint16 )
        
        mixture_evaluategrid_diagonal( self._c_mixture, &grid_acc[0,0], ngrid, &x[0,0], npointsdim, &pointsdim[0], &result[0,0] )
        
        return result
    
    def update_cache( self ):
        mixture_update_cache( self._c_mixture )
    
    @property
    def ncomponents(self):
        return self._c_mixture.ncomponents
    
    @property
    def ndim(self):
        return self._c_mixture.ndim
    
    @property
    def refcount(self):
        return self._c_mixture.refcount
    
    @property
    def sum_n(self):
        return self._c_mixture.sum_n
    
    @property
    def sum_weights(self):
        return self._c_mixture.sum_weights
    
    @property
    def means(self):
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] result
        result = np.empty( (self.ncomponents,self.ndim), dtype=np.float64, order="C" )
        mixture_get_means( self._c_mixture, &result[0,0] )
        return result
    
    @property
    def scaling_factors(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ncomponents), dtype=np.float64, order="C" )
        mixture_get_scaling_factors( self._c_mixture, &result[0] )
        return result
    
    @property
    def weights(self):
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] result
        result = np.empty( (self.ncomponents), dtype=np.float64, order="C" )
        mixture_get_weights( self._c_mixture, &result[0] )
        return result
    
    @property
    def covariances(self):
        cdef np.ndarray[np.double_t, ndim=3, mode="c"] result
        result = np.empty( (self.ncomponents, self.ndim, self.ndim), dtype=np.float64, order="C" )
        mixture_get_covariances( self._c_mixture, &result[0,0,0] )
        return result
    
    def tofile(self, filename):
        return mixture_save_to_file( self._c_mixture, filename )
    
    @classmethod
    def fromfile(cls, filename):
        result = MixtureClass(ndim=0)
        result._c_mixture = mixture_load_from_file( filename )
        # update scaling factors
        result.update_cache()        
        
        return result
    
    
    
        
