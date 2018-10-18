import yaml
import numpy as np
import ast

__all__ = ['read_yaml_header', 'load_file']

def read_yaml_header( stream ):
    """Reads YAML header from stream.
    
    Parameters
    ----------
    stream : open file
    
    Returns
    -------
    header : YAML node
    headersize
    
    """
    
    s = stream.read(4)
    if (s!="---\n"):
        raise Exception("Invalid yaml header")
    
    header = ""
    
    while ( s!="...\n" ):
        s = stream.readline()
        header += s
    
    return yaml.load(header), stream.tell()


def data_description_to_dtype( desc ):
    """Converts data description string to numpy stype.
    
    Parameters
    ----------
    desc : string
    
    Returns
    -------
    dtype
    
    """
    
    dt = []
    
    for field in desc:
        
        parts = field.split(" ")
        
        if len(parts)!=3:
            raise Exception("Invalid field description")
        
        if parts[1]=='string': parts[1]='str'
        
        parts[2] = parts[2].replace('[','(').replace(']',')')
        
        dt.append( (parts[0], (getattr(np,parts[1]),ast.literal_eval(parts[2]))) )
    
    return np.dtype( dt )


def load_file( filename ):
    """Loads data from serialized Falcon data file.
    
    Parameters
    ----------
    filename : string
    
    Returns
    -------
    data : YAML node or numpy array;
    
    header : YAML node
    
    """
    
    with open( filename, 'r' ) as fid:
        
        #load header
        header, headersize = read_yaml_header( fid )
        
        if header["encoding"]=="binary":
            dt = data_description_to_dtype( header["data"] )
            data = np.memmap( filename, dtype=dt, mode='r', offset=headersize )
        elif header["encoding"]=="yaml":
            data = yaml.load( fid )
        else:
            raise Exception("Unknown encoding")
        
    return data, header
    

def load_spike_data( filename ):
    """
    Loads Falcon serialized Spike data.
    
    Parameters
    ----------
    filename : string
    
    Returns
    ---------
    spike_data : dictionary with three keys:
                "n_spikes": numpy array of size (n_packets, ) containing the # detected spikes per packet,
                "times": numpy array of size (n_spikes, ) containing all time points [in sec]
                    of the detected spikes (derived from the hw timestamps),
                "amplitudes": numpy array of size (n_spikes, n_channels) containing all amplitudes
                    of each detected spike
    tot_n_spikes : uint64_t
    n_channels : uint8_t
    original_data : YAML node or numpy array
    header : YAML node
    
    """

    data, header = load_file (filename)

    n_channels = int(header['data'][-1][-2])    
    spike_data = {}


    if header['format'] == 'FULL': 

        spike_data['n_spikes'] = data['n_detected_spikes']
        tot_n_spikes = sum(spike_data['n_spikes'])
        spike_data['times'] = np.zeros(tot_n_spikes)
        spike_data['amplitudes'] = np.zeros( (tot_n_spikes, n_channels) )

        sp = 0 # loops over spikes in spike_data
        
        for p, ns in enumerate(spike_data['n_spikes']):
            spike_data['times'][sp:sp+ns] = data['TS_detected_spikes'][p][:ns]/1e6
            spike_data['amplitudes'][sp:sp+ns, :] = data['spike_amplitudes'][p][:ns]
            sp += ns
            
    if header['format'] == 'COMPACT':
        tot_n_spikes = len(data)
        spike_data['times'] = data['TS_detected_spikes'] / 1e6
        spike_data['amplitudes'] = data['spike_amplitudes']

    return spike_data, tot_n_spikes, n_channels, data, header 
    
    
def load_likelihood_data( filename ):
    """
    Loads Falcon serialized Likelihood Data.
    
    Parameters
    ----------
    filename : string
    
    Return
    ---------
   likelihood_data : dictionary with four keys:
                "log_likelihood": numpy array of size (n_packets, grid_size),
                    it contains the values of the likelihood in logarithmic space
                    of each LikelihoodData item;
                "n_spikes": numpy array of size (n_packets),
                    it contains the number of spikes of each LikelihoodData item
                "time_bin": numpy array of size (n_packets),
                    it contains the time bin in [ms] that was used to compute
                    each LikelihoodData item
                "likelihood": numpy array of size (n_packets, grid_size),
                    it contains the values of the likelihood in linear space
                    of each LikelihoodData item; this field is empty if data
                    was not saved using FULL mode
    grid_size : int
    
    original_data : YAML node or numpy array
    
    header : YAML node
    
    """

    data, header = load_file (filename)

    if header['format'] == 'COMPACT':
        grid_size = int(header['data'][2][-4:-1]) # only works with grid sizes of 3 digits   
    else: #TODO: add deserialization from header only
        assert( header['format'] == 'FULL')
        grid_size = int(header['data'][7][-4:-1]) # only works with grid sizes of 3 digits   
        assert (grid_size == int(header['data'][8][-4:-1]))
        
    n_bins = len(data)
    
    likelihood_data = {}

    if (header['format'] == 'FULL') or (header['format'] == 'COMPACT'): 
        likelihood_data['log_likelihood'] = data['log_likelihood']
        likelihood_data['n_spikes'] = data['n_spikes']
        likelihood_data['time_bin_ms'] = data['time_bin_ms']
            
    if header['format'] == 'FULL':
        likelihood_data['likelihood'] = data['likelihood']

    return likelihood_data, grid_size, n_bins, data, header 
