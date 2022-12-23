import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import sys
import os
import pyworld as world # Vocoder
import numpy as np # Numpy <3
import scipy.io.wavfile as wav # WAV read + write
import scipy.signal as signal # for filtering
import scipy.interpolate as interp # Interpolator for feats
import resampy # Resampler (as in sampling rate stuff)
import re

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
flags = ['fe', 'fl', 'fo', 'fv', 've', 'vo', 't', 'A', 'B', 'G', 'P']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

# Utility functions
def smoothstep(edge0, edge1, x):
    """Smoothstep function from GLSL that works with numpy arrays."""
    x = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return 3*x*x - 2*x*x*x

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
    """Read WAV file that remaps unsigned integer WAV files to [-1, 1] range, resamples to 44.1kHz if needed, and mixes down to mono if needed. May fail on unsigned integer types.

    Parameters
    ----------
    loc : str or file
        Input WAV file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    fs, x = wav.read(loc)
    # Check integer typing
    xtype = x.dtype
    int_type = np.issubdtype(xtype, np.integer)

    if int_type:
        # Divide by max integer
        info = np.iinfo(xtype)
        x = x / info.max

    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than 2.0
        x = np.mean(x, axis=1)

    if fs != default_fs:
        x = resampy.resample(x, fs, default_fs)

    return x

def save_wav(loc, x):
    """Save data into a WAV file. Assumes data is in 44.1kHz and in [-1, 1] range.

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
    info = np.iinfo(np.int16)
    x = np.clip(x * info.max, info.min, info.max).astype(np.int16)
    wav.write(loc, default_fs, x)

# Processing WORLD things
def base_frq(f0, f0_min=None, f0_max=None):
    """Get average F0 with a stronger bias on flatter areas. Port from https://github.com/titinko/frq0003gen

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
    value = 0
    r = 1
    p = [0, 0, 0, 0, 0, 0]
    q = 0
    avg_frq = 0
    base_value = 0

    if not f0_min:
        f0_min = f0_floor

    if not f0_max:
        f0_max = f0_ceil
    
    for i in range(0, len(f0)):
        value = f0[i]
        if value <= f0_max and value >= f0_min:
            r = 1

            for j in range(0, 6):
                if i > j:
                    q = f0[i - j - 1] - value
                    p[j] = value / (value + q * q)
                else:
                    p[j] = 1 / (1 + value)
                    
                r *= p[j]

            avg_frq += value * r
            base_value += r

    if base_value > 0:
        avg_frq /= base_value
    return avg_frq

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
        # Convert cut times to units of 5 ms
        logging.info('Calculating timing.')
        start = int(np.floor(self.offset / 5))
        if self.cutoff >= 0:
            end = -int(np.floor(self.cutoff / 5)) - 1
        else:
            end = int(np.ceil((self.offset - self.cutoff) / 5))
        con = int(self.consonant / 5)

        # Convert percentages to decimal
        vel = np.exp2(1 - self.velocity / 100)
        vol = self.volume / 100
        mod = self.modulation / 100

        logging.info('Decoding WORLD features.')
        # Recalculate spectral envelope and aperiodicity and trim to length
        sp = world.decode_spectral_envelope(features['mgc'], default_fs, fft_size)[start:end]
        ap = world.decode_aperiodicity(features['bap'], default_fs, fft_size)[start:end]

        # Turn F0 to offset map for modulation
        base_f0 = features['base']
        f0 = features['f0'][start:end]
        f0[f0 == 0] = base_f0
        f0_off = f0 - base_f0
        
        # Calculate temporal positions
        t_area = np.arange(len(f0)) * 0.005

        logging.info('Preparing interpolators.')
        # Make interpolators to render new areas
        f0_off_interp = interp.UnivariateSpline(t_area, f0_off, s=0, ext='const')
        sp_interp = interp.Akima1DInterpolator(t_area, sp)
        ap_interp = interp.Akima1DInterpolator(t_area, ap)

        # Make new temporal positions array for stretching
        t_pivot = t_area[con] # temporal position pivot for stretched/unstretched area
        t_consonant = np.linspace(0, t_pivot, num=int(vel * self.consonant / 5), endpoint=False) # temporal positions of the unstretched area. can be stretched because of velocity
        # stretched area only needs to stretch if the length required is longer than the stretch area
        length_req = int(self.length / 5)
        stretch_length = len(f0) - con
        if stretch_length > length_req:
            t_stretch = t_area[con:con+length_req]
        else:
            t_stretch = np.linspace(t_pivot, t_area[-1], num=length_req)
        
        t_render = np.concatenate([t_consonant, t_stretch]) # concatenate for interpolation
        con = len(t_consonant)
        
        logging.info('Interpolating WORLD features.')
        # Interpolate render area
        f0_off_render = f0_off_interp(t_render)
        sp_render = sp_interp(t_render)
        ap_render = np.clip(ap_interp(t_render), 0, 1) # aperiodicity freaks out if not within [0, 1] range

        # Calculate new temporal positions for tuning
        t = np.arange(len(sp_render)) * 0.005

        logging.info('Calculating pitch.')
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96)
        pitch_interp = interp.UnivariateSpline(t_pitch, pitch, s=0, ext='const')
        pitch_render = pitch_interp(t)
        
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
            if 'fl' in self.flags.keys(): # check length flag
                fry_len = max(self.flags['fl'] / 1000, 0.001)

            if 'fo' in self.flags.keys():
                fry_offset = self.flags['fo'] / 1000
            
            # Prepare envelope
            t_fry = t - t[con] - fry_offset # temporal positions centered around the consonant shifted by offset
            amt = smoothstep(-fry - fry_len / 2, -fry + fry_len / 2, t_fry) * smoothstep(fry_len / 2, -fry_len / 2, t_fry) #fry envelope

            f0_render = f0_render * (1 - amt) + f0_floor * amt # mix low F0 for fry
                
        # Breathiness flag
        if 'B' in self.flags.keys():
            breath = self.flags['B']
            if breath <= 50: # Raise power to flatten smaller areas and keep max aperiodicity
                logging.info('Lowering breathiness.')
                breath = np.clip(2 * breath / 50, 0, 1)
                ap_render = np.power(ap_render, 5 * (1 - breath) + 1)
        else:
            breath = 0
            
        # remove pitch in areas with max aperiodicity
        husk = np.mean(ap_render, axis=1)
        f0_render[np.isclose(husk, 1)] = 0
        render = world.synthesize(f0_render, sp_render, ap_render, default_fs)
        
        ### AFTER RENDER FLAGS ###
        if breath > 50: # mix max breathiness signal
            logging.info('Raising breathiness.')
            breath = np.clip((breath - 50) / 50, 0, 1)
            render_breath = world.synthesize(f0_render, sp_render, np.ones(ap_render.shape), default_fs) # render with all max aperiodicity
            render_breath = highpass(render_breath)
            
            
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
                fry_vol = np.clip(self.flags['fv'] / 100, 0, 1)
            
            # Prepare envelope
            t_fry = t_sample - t[con] - fry_offset # temporal positions centered around the consonant shifted by offset
            amt = smoothstep(-fry - fry_len / 2, -fry + fry_len / 2, t_fry) * smoothstep(fry_len / 2, -fry_len / 2, t_fry) #fry envelope
            env = 1 - amt + fry_vol * amt

            render *= env
        
        # Fix voicing flag
        if 've' in self.flags.keys():
            logging.info('Fixing voicing.')
            end_breath = self.flags['ve'] / 1000
            render_breath = world.synthesize(f0_render, sp_render, np.ones(ap_render.shape), default_fs) # render with all max aperiodicity
            render_breath = highpass(render_breath)

            offset = 0
            if 'vo' in self.flags.keys(): # check offset flag
                offset = self.flags['vo'] / 1000
                logging.info(offset)
            
            amt = smoothstep(-end_breath / 2, end_breath / 2, t_sample - t[con] - offset) # smoothstep with consonant at 0.5
            render = render * (1 - amt) + render_breath * amt # mix sample based on envelope

        # Tremolo flag
        if 'A' in self.flags.keys():
            logging.info('Adding tremolo.')
            tremolo = self.flags['A'] / 100
            
            pitch_sample = pitch_interp(t_sample) # probably bad because of how low the sampling rate is for the pitch
            pitch_smooth = lowpass(pitch_sample, cutoff=8, order=4)
            vibrato = highpass(pitch_smooth, cutoff=4, order=4)

            amt = np.maximum(tremolo * vibrato + 1, 0)
            render = render * amt
            
        peak = 1 # Peak "compression" but it's actually just normalization LOL
        if 'P' in self.flags.keys():
            peak = np.clip(self.flags['P'] / 100, 0, 1)

        normal = 0.9 * render / np.max(np.abs(render))
        render = render * (1 - peak) + normal * peak
        
        render *= vol # volume
        if self.out_file != 'nul':
            save_wav(self.out_file, render)

if __name__ == '__main__':
    logging.info('straycat 0.1.0')
    try:
        Resampler(*sys.argv[1:])
    except Exception as e:
        name = e.__class__.__name__
        if name == 'TypeError':
            logging.info(help_string)
        else:
            raise e
