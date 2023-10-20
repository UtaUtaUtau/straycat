import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import sys
import os
import pyworld as world # Vocoder
import numpy as np # Numpy <3
from numba import njit, vectorize, float64, optional # JIT compilation stuff (and ufuncs)
import soundfile as sf # WAV read + write
import scipy.signal as signal # for filtering
import scipy.interpolate as interp # Interpolator for feats
import resampy # Resampler (as in sampling rate stuff)
import re

version = '0.2.2'
help_string = '''usage: straycat in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

Resamples using the WORLD Vocoder.

arguments:
\tin_file\t\tPath to input file.
\tout_file\tPath to output file.
\tpitch\t\tThe pitch to render on.
\tvelocity\tThe consonant velocity of the render.

optional arguments:
\tflags\t\tThe flags of the render.
\toffset\t\tThe offset from the start of the render area of the sample. (default: 0)
\tlength\t\tThe length of the stretched area in milliseconds. (default: 1000)
\tconsonant\tThe unstretched area of the render in milliseconds. (default: 0)
\tcutoff\t\tThe cutoff from the end or from the offset for the render area of the sample. (default: 0)
\tvolume\t\tThe volume of the render in percentage. (default: 100)
\tmodulation\tThe pitch modulation of the render in percentage. (default: 0)
\ttempo\t\tThe tempo of the render. Needs to have a ! at the start. (default: !100)
\tpitch_string\tThe UTAU pitchbend parameter written in Base64 with RLE encoding. (default: AA)'''

notes = {'C' : 0, 'C#' : 1, 'D' : 2, 'D#' : 3, 'E' : 4, 'F' : 5, 'F#' : 6, 'G' : 7, 'G#' : 8, 'A' : 9, 'A#' : 10, 'B' : 11} # Note names lol
note_re = re.compile(r'([A-G]#?)(-?\d+)') # Note Regex for conversion
default_fs = 44100 # UTAU only really likes 44.1khz
fft_size = world.get_cheaptrick_fft_size(default_fs, world.default_f0_floor) # It's just 2048 but you know
cache_ext = '.sc.npz' # cache file extension

# Giving it better range
f0_floor = world.default_f0_floor 
f0_ceil = 1760

# Flags
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't', 'A', 'B', 'G', 'P', 'S', 'p']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

# Utility functions
@vectorize([float64(float64, float64, float64)], nopython=True)
def smoothstep(edge0, edge1, x):
    """Smoothstep function from GLSL that works with numpy arrays."""
    x = (x - edge0) / (edge1 - edge0)
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return 3*x*x - 2*x*x*x

@vectorize([float64(float64, float64, float64)], nopython=True)
def clip(x, x_min, x_max):
    """Clips function. Faster than np.clip somehow"""
    if x < x_min:
        return x_min
    if x > x_max:
        return x_max
    return x

@vectorize([float64(float64, float64)], nopython=True)
def bias(x, a):
    """Element-wise Schlick bias function."""
    if a == 0:
        return 0
    if a == 1:
        return 1
    return x / ((1 / a - 2) * (1 - x) + 1)

def highpass(x, fs=44100, cutoff=3000, order=1):
    """Butterworth highpass with doubled order because of sosfiltfilt."""
    nyq = 0.5 * fs
    cut = cutoff / nyq
    sos = signal.butter(order, cut, btype='high', output='sos')
    return signal.sosfiltfilt(sos, x)

def lowpass(x, fs=44100, cutoff=16000, order=1):
    """Butterworth lowpass with doubled order because of sosfiltfilt."""
    nyq = 0.5 * fs
    cut = cutoff / nyq
    sos = signal.butter(order, cut, btype='low', output='sos')
    return signal.sosfiltfilt(sos, x)

# Pitch string interpreter
def to_uint6(b64):
    """Convert one Base64 character to an unsigned integer.

    Parameters
    ----------
    b64 : str
        The Base64 character.

    Returns
    -------
    int
        The equivalent of the Base64 character as an integer.
    """
    c = ord(b64) # Convert based on ASCII mapping
    if c >= 97:
        return c - 71
    elif c >= 65:
        return c - 65
    elif c >= 48:
        return c + 4
    elif c == 43:
        return 62
    elif c == 47:
        return 63
    else:
        raise Exception

def to_int12(b64):
    """Converts two Base64 characters to a signed 12-bit integer.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    int
        The equivalent of the Base64 characters as a signed 12-bit integer (-2047 to 2048)
    """
    uint12 = to_uint6(b64[0]) << 6 | to_uint6(b64[1]) # Combined uint6 to uint12
    if uint12 >> 11 & 1 == 1: # Check most significant bit to simulate two's complement
        return uint12 - 4096
    else:
        return uint12

def to_int12_stream(b64):
    """Converts a Base64 string to a list of integers.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    list
        The equivalent of the Base64 string if split every 12-bits and interpreted as a signed 12-bit integer.
    """
    res = []
    for i in range(0, len(b64), 2):
        res.append(to_int12(b64[i:i+2]))
    return res

def pitch_string_to_cents(x):
    """Converts UTAU's pitchbend argument to an ndarray representing the pitch offset in cents.

    Parameters
    ----------
    x : str
        The pitchbend argument.

    Returns
    -------
    ndarray
        The pitchbend argument as pitch offset in cents.
    """
    pitch = x.split('#') # Split RLE Encoding
    res = []
    for i in range(0, len(pitch), 2):
        # Go through each pair
        p = pitch[i:i+2]
        if len(p) == 2:
            # Decode pitch string and extend RLE
            pitch_str, rle = p
            res.extend(to_int12_stream(pitch_str))
            res.extend([res[-1]] * int(rle))
        else:
            # Decode last pitch string without RLE if it exists
            res.extend(to_int12_stream(p[0]))
    res = np.array(res, dtype=np.int32)
    if np.all(res == res[0]):
        return np.zeros(res.shape)
    else:
        return np.concatenate([res, np.zeros(1)])

# Pitch conversion
def note_to_midi(x):
    """Note name to MIDI note number."""
    note, octave = note_re.match(x).group(1, 2)
    octave = int(octave) + 1
    return octave * 12 + notes[note]

def midi_to_hz(x):
    """MIDI note number to Hertz using equal temperament. A4 = 440 Hz."""
    return 440 * np.exp2((x - 69) / 12)

##def hz_to_midi(x):
##    return 12 * np.log2(x / 440) + 69

# WAV read/write
def read_wav(loc):
    """Read audio files supported by soundfile and resample to 44.1kHz if needed. Mixes down to mono if needed.

    Parameters
    ----------
    loc : str or file
        Input WAV file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    x, fs = sf.read(loc)
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1)

    if fs != default_fs:
        x = resampy.resample(x, fs, default_fs)

    return x

def save_wav(loc, x):
    """Save data into a WAV file.

    Parameters
    ----------
    loc : str or file
        Output WAV file.

    x : ndarray
        Audio data in 44.1kHz within [-1, 1].

    Returns
    -------
    None
    """
    sf.write(loc, x, default_fs, 'PCM_16')

# Processing WORLD things
@njit(float64(float64[:], optional(float64), optional(float64)))
def _jit_base_frq(f0, f0_min, f0_max):
    q = 0
    avg_frq = 0
    tally = 0
    N = len(f0)

    if f0_min is None:
        f0_min = f0_floor

    if f0_max is None:
        f0_max = f0_ceil
    
    for i in range(N):
        if f0[i] >= f0_min and f0[i] <= f0_max:
            if i < 1:
                q = f0[i+1] - f0[i]
            elif i == N - 1:
                q = f0[i] - f0[i-1]
            else:
                q = (f0[i+1] - f0[i-1]) / 2
            weight = 2 ** (-q * q)
            avg_frq += f0[i] * weight
            tally += weight

    if tally > 0:
        avg_frq /= tally
    return avg_frq

def base_frq(f0, f0_min=None, f0_max=None):
    """Get average F0 with a stronger bias on flatter areas. 

    Parameters
    ----------
    f0 : list or ndarray
        Array of F0 values.

    f0_min : float, optional
        Lower F0 limit.

    f0_max : float, optional
        Upper F0 limit.

    Returns
    -------
    float
        Average F0.
    """
    return _jit_base_frq(f0, f0_min, f0_max)

class Resampler:
    """
    A class for the UTAU resampling process.

    Attributes
    ----------
    in_file : str
        Path to input file.

    out_file : str
        Path to output file.

    pitch : str
        The pitch of the note.

    velocity : str or float
        The consonant velocity of the note.

    flags : str
        The flags of the note.

    offset : str or float
        The offset from the start for the render area of the sample.

    length : str or int
        The length of the stretched area in milliseconds.

    consonant : str or float
        The unstretched area of the render.

    cutoff : str or float
        The cutoff from the end or from the offset for the render area of the sample.

    volume : str or float
        The volume of the note in percentage.

    modulation : str or float
        The modulation of the note in percentage.

    tempo : str
        The tempo of the note.

    pitch_string : str
        The UTAU pitchbend parameter.

    Methods
    -------    
    render(self):
        The rendering workflow. Immediately starts when class is initialized.

    get_features(self):
        Gets the WORLD features either from a cached file or generating it if it doesn't exist.

    generate_features(self, features_path):
        Generates WORLD features and saves it for later.

    resample(self, features):
        Renders a WAV file using the passed WORLD features.
    """
    def __init__(self, in_file, out_file, pitch, velocity, flags='', offset=0, length=1000, consonant=0, cutoff=0, volume=100, modulation=0, tempo='!100', pitch_string='AA'):
        """Initializes the renderer and immediately starts it.

        Parameters
        ---------
        in_file : str
            Path to input file.

        out_file : str
            Path to output file.

        pitch : str
            The pitch of the note.

        velocity : str or float
            The consonant velocity of the note.

        flags : str
            The flags of the note.

        offset : str or float
            The offset from the start for the render area of the sample.

        length : str or int
            The length of the stretched area in milliseconds.

        consonant : str or float
            The unstretched area of the render.

        cutoff : str or float
            The cutoff from the end or from the offset for the render area of the sample.

        volume : str or float
            The volume of the note in percentage.

        modulation : str or float
            The modulation of the note in percentage.

        tempo : str
            The tempo of the note.

        pitch_string : str
            The UTAU pitchbend parameter.
        """
        self.in_file = in_file
        self.out_file = out_file
        self.pitch = note_to_midi(pitch)
        self.velocity = float(velocity)
        self.flags = {k : int(v) if v else None for k, v in flag_re.findall(flags.replace('/', ''))}
        self.offset = float(offset)
        self.length = int(length)
        self.consonant = float(consonant)
        self.cutoff = float(cutoff)
        self.volume = float(volume)
        self.modulation = float(modulation)
        self.tempo = float(tempo[1:])
        self.pitchbend = pitch_string_to_cents(pitch_string)

        self.render()
    
    def render(self):
        """The rendering workflow. Immediately starts when class is initialized.

        Parameters
        ----------
        None
        """
        features = self.get_features()
        self.resample(features)

    def get_features(self):
        """Gets the WORLD features either from a cached file or generating it if it doesn't exist.

        Parameters
        ----------
        None

        Returns
        -------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.
        """
        # Setup cache path file
        loc, file = os.path.split(self.in_file)
        fname, _ = os.path.splitext(file)
        features_path = os.path.join(loc, fname + cache_ext)
        features = None

        if 'G' in self.flags.keys():
            logging.info('G flag exists. Forcing feature generation.')
            features = self.generate_features(features_path)
        elif os.path.exists(features_path):
            # Load if it exists
            logging.info(f'Reading {fname}{cache_ext}.')
            features = np.load(features_path)
        else:
            # Generate if not
            logging.info(f'{fname}{cache_ext} not found. Generating features.')
            features = self.generate_features(features_path)

        return features

    def generate_features(self, features_path):
        """Generates WORLD features and saves it for later.

        Parameters
        ----------
        features_path : str or file
            The path for caching the features.

        Returns
        -------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.
        """
        x = read_wav(self.in_file)
        logging.info('Generating F0.')
        f0, t = world.harvest(x, default_fs, f0_floor=f0_floor, f0_ceil=f0_ceil)
        base_f0 = base_frq(f0)
        
        logging.info('Generating spectral envelope.')
        sp = world.cheaptrick(x, f0, t, default_fs)
        mgc = world.code_spectral_envelope(sp, default_fs, 64)
        
        logging.info('Generating aperiodicity.')
        ap = world.d4c(x, f0, t, default_fs, threshold=0.25)
        bap = world.code_aperiodicity(ap, default_fs)
        
        logging.info('Saving features.')
        
        features = {'base' : base_f0, 'f0' : f0, 'mgc' : mgc, 'bap' : bap}
        np.savez_compressed(features_path, **features)

        return features
    
    def resample(self, features):
        """Renders a WAV file using the passed WORLD features.

        Parameters
        ----------
        features : dict
            A dictionary of the F0, MGC, BAP, and average F0.

        Returns
        -------
        None
        """
        if self.out_file == 'nul':
            logging.info('Null output file. Skipping...')
            return
        
        # Convert percentages to decimal
        vel = np.exp2(1 - self.velocity / 100) # convel is more a multiplier...
        vol = self.volume / 100
        mod = self.modulation / 100

        logging.info('Decoding WORLD features.')
        # Recalculate spectral envelope and aperiodicity
        sp = world.decode_spectral_envelope(features['mgc'], default_fs, fft_size)
        ap = world.decode_aperiodicity(features['bap'], default_fs, fft_size)

        # Turn F0 to offset map for modulation
        base_f0 = features['base']
        f0 = features['f0']
        f0[f0 == 0] = base_f0
        f0_off = f0 - base_f0
        
        # Calculate temporal positions
        t_area = np.arange(len(f0)) * 0.005

        logging.info('Calculating timing.') # use seconds instead of 5ms terms cuz someone gave me negative offsets </3
        start = self.offset / 1000 # start time
        end = self.cutoff / 1000 # end time
        if self.cutoff < 0: # deal with relative end time
            end = start - end
        else:
            end = t_area[-1] - end
        con = start + self.consonant / 1000 # consonant

        logging.info('Preparing interpolators.')
        # Make interpolators to render new areas
        f0_off_interp = interp.UnivariateSpline(t_area, f0_off, s=0, ext='const')
        sp_interp = interp.Akima1DInterpolator(t_area, sp)
        ap_interp = interp.Akima1DInterpolator(t_area, ap)

        # Make new temporal positions array for stretching
        t_consonant = np.linspace(start, con, num=int(vel * self.consonant / 5), endpoint=False) # temporal positions of the unstretched area. can be stretched because of velocity
        # stretched area only needs to stretch if the length required is longer than the stretch area
        length_req = self.length / 1000
        stretch_length = end - con
        if stretch_length > length_req:
            con_idx = int(200 * con) # position of consonant in the temporal positions array ??
            len_idx = int(200 * length_req) # length of length required by 5ms frames
            t_stretch = t_area[con_idx:con_idx+len_idx]
        else:
            t_stretch = np.linspace(con, end, num=int(200 * length_req))
        
        t_render = clip(np.concatenate([t_consonant, t_stretch]), 0, t_area[-1]) # concatenate and clip for interpolation
        con = len(t_consonant) # new placement of the consonant, now in 5ms frame terms...
        
        logging.info('Interpolating WORLD features.')
        # Interpolate render area
        f0_off_render = f0_off_interp(t_render)
        sp_render = sp_interp(t_render)
        ap_render = clip(ap_interp(t_render), 0, 1) # aperiodicity freaks out if not within [0, 1] range

        # Calculate new temporal positions for tuning
        t = np.arange(len(sp_render)) * 0.005

        logging.info('Calculating pitch.')
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96)
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(clip(t, 0, t_pitch[-1]))
        
        logging.info('Checking flags.')
        # Flag interpretation area
        ### BEFORE HZ CONVERSION FLAGS ###
        # Pitch offset flag
        if 't' in self.flags.keys():
            pitch_render += self.flags['t'] / 100

        # Convert pitch to Hertz and add F0 offset for modulation
        f0_render = midi_to_hz(pitch_render) + f0_off_render * mod

        ### BEFORE RENDER FLAGS ###
        # Vocal Fry flag
        if 'fe' in self.flags.keys():
            logging.info('Adding vocal fry.')
            fry = self.flags['fe'] / 1000
            fry_len = 0.075
            fry_offset = 0
            fry_pitch = f0_floor
            if 'fl' in self.flags.keys(): # check length flag
                fry_len = max(self.flags['fl'] / 1000, 0.001)

            if 'fo' in self.flags.keys():
                fry_offset = self.flags['fo'] / 1000

            if 'fp' in self.flags.keys():
                fry_pitch = max(self.flags['fp'], 0)
            
            # Prepare envelope
            t_fry = t - t[con] - fry_offset # temporal positions centered around the consonant shifted by offset
            amt = smoothstep(-fry - fry_len / 2, -fry + fry_len / 2, t_fry) * smoothstep(fry_len / 2, -fry_len / 2, t_fry) #fry envelope

            f0_render = f0_render * (1 - amt) + fry_pitch * amt # mix low F0 for fry

        # Gender/Formant shift flag
        if 'g' in self.flags.keys():
            logging.info('Shifting formants.')
            gender = np.exp2(self.flags['g'] / 120)

            freq_x = np.linspace(0, 1, fft_size // 2 + 1) # map spectral envelope by frequency instead of time
            sp_render_interp = interp.Akima1DInterpolator(freq_x, sp_render, axis=1)

            # stretch spectral envelope depending on gender
            freq_x = clip(np.linspace(0, gender, fft_size // 2 + 1), 0, 1) # clip axis because Akima1DInterpolator doesn't extrapolate (or even just extend)
            sp_render = sp_render_interp(freq_x).copy(order='C')

        # map unvoicedness (kinda like voisona huskiness)
        husk = np.mean(ap_render, axis=1)
        
        # Breathiness flag
        if 'B' in self.flags.keys():
            breath = self.flags['B']
            if breath <= 50: # Raise power to flatten smaller areas and keep max aperiodicity
                logging.info('Lowering breathiness.')
                breath = breath / 100
                ap_render = bias(ap_render, breath)
                ap_render[np.isclose(husk, 1),:] = 1 # make sure unvoiced areas stay unvoiced... only happens if breathiness is 0 but too much if statements
        else:
            breath = 0
            
        #Peak compressor flag
        peak = self.flags.get('P', 86) / 100

        rms = np.sqrt(2 * np.sum(sp_render, axis=1) / fft_size ** 2) # get RMS.. i'm not sure if this is right but i think it's fine
        rms_peak = np.max(rms)
        rms_norm = rms / (peak * rms_peak)

        comp = np.zeros(rms_norm.shape)
        comp[rms_norm >= 1] = rms_norm[rms_norm >= 1] - 1
        comp = (1 - peak) * comp / np.max(comp)
        comp = 1 - comp
        env = np.exp(np.linspace(0, -5, 10))
        env /= np.sum(env)

        comp = 1 - signal.convolve(1 - comp, env, mode='same')        

        comp = np.vstack([np.square(comp)] * sp_render.shape[1]).transpose()
        sp_render *= comp
        ap_render *= comp

        # remove pitch in areas with max aperiodicity
        f0_render[np.isclose(husk, 1)] = 0
        render = world.synthesize(f0_render, sp_render, ap_render, default_fs)
        
        ### AFTER RENDER FLAGS ###
        # Max aperiodicity flag
        if 'S' in self.flags.keys():
            amt = clip(self.flags['S'] / 100, 0, 1)
            render_ap = world.synthesize(f0_render, sp_render, np.ones(ap_render.shape), default_fs)
            render = render * (1 - amt) + render_ap * amt
        
        if breath > 50: # mix max breathiness signal
            logging.info('Raising breathiness.')
            breath = clip((breath - 50) / 50, 0, 1)
            render_breath = world.synthesize(f0_render, sp_render * np.square(ap_render), np.ones(ap_render.shape), default_fs) # apply band AP on regular specgram, max out ap            
            
            render = render * (1 - breath) + render_breath * breath # Mix signals
            
        t_sample = np.arange(len(render)) / default_fs # temporal position per sample
        if 'fe' in self.flags.keys():
            fry = self.flags['fe'] / 1000
            fry_len = 0.05
            fry_offset = 0
            fry_vol = 0.1
            if 'fl' in self.flags.keys(): # check length flag
                fry_len = max(self.flags['fl'] / 1000, 0.001)

            if 'fo' in self.flags.keys():
                fry_offset = self.flags['fo'] / 1000

            if 'fv' in self.flags.keys():
                fry_vol = clip(self.flags['fv'] / 100, 0, 1)
            
            # Prepare envelope
            t_fry = t_sample - t[con] - fry_offset # temporal positions centered around the consonant shifted by offset
            amt = smoothstep(-fry - fry_len / 2, -fry + fry_len / 2, t_fry) * smoothstep(fry_len / 2, -fry_len / 2, t_fry) #fry envelope
            env = 1 - amt + fry_vol * amt

            render_hp = highpass(render, cutoff=300) # add a highpass through the fry area
            render = render * (1 - amt) + render_hp * amt
            render *= env
        
        # Fix voicing flag
        if 've' in self.flags.keys():
            logging.info('Fixing voicing.')
            end_breath = self.flags['ve'] / 1000
            render_breath = world.synthesize(f0_render, sp_render * np.square(ap_render), np.ones(ap_render.shape), default_fs) # apply band AP on regular specgram, max out ap  

            offset = 0
            if 'vo' in self.flags.keys(): # check offset flag
                offset = self.flags['vo'] / 1000
                logging.info(offset)
            
            amt = smoothstep(-end_breath / 2, end_breath / 2, t_sample - t[con] - offset) # smoothstep with consonant at 0.5
            render = render * (1 - amt) + render_breath * amt # mix sample based on envelope
            
        normalize = max(self.flags.get('p', 6), 0)

        normal = render / np.max(render)
        render = normal * (10 ** (-normalize / 20))

        ### AFTER PEAK NORMALIZATION ###
        # Tremolo flag
        if 'A' in self.flags.keys():
            logging.info('Adding tremolo.')
            tremolo = self.flags['A'] / 100
            
            pitch_sample = pitch_interp(clip(t_sample, 0, t_pitch[-1])) # probably bad because of how low the sampling rate is for the pitch
            pitch_smooth = lowpass(pitch_sample, cutoff=8, order=16)
            vibrato = highpass(pitch_smooth, cutoff=4, order=16)

            amt = np.maximum(tremolo * vibrato + 1, 0)
            render = render * amt
        
        render *= vol # volume
        save_wav(self.out_file, render)

if __name__ == '__main__':
    logging.info(f'straycat {version}')
    try:
        Resampler(*sys.argv[1:])
    except Exception as e:
        name = e.__class__.__name__
        if name == 'TypeError':
            logging.info(help_string)
        else:
            raise e
