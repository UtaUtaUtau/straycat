# straycat
 Yet another WORLD-based UTAU resampler.

# How to use (development version)
 You need to have Python installed. This was made using Python 3.8.10.
 
 You must install the needed libraries first, which are numpy, scipy, resampy, and pyworld. To do that, you may run a regular pip installation:
 
```
pip install numpy scipy resampy pyworld
```
 
## Running in UTAU
 Download the `straycat.py` file and put it somewhere. Open your `.ust` file as a text file with whichever text editor. Change the `Tool2` resampler to the path of `straycat.py`. You can now open the `.ust` and use `straycat.py` as a resampler. You may need to press cancel for the project properties when UTAU shows the project properties panel.
 
## Running throught terminal
 Most resamplers can take arguments to render a sample. This resampler only reads the terminal arguments.
 
```
usage: straycat in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

Resamples using the WORLD Vocoder.

arguments:
	in_file		Path to input file.
	out_file	Path to output file.
	pitch		The pitch to render on.
	velocity	The consonant velocity of the render.

optional arguments:
	flags		The flags of the render.
	offset		The offset from the start of the render area of the sample. (default: 0)
	length		The length of the stretched area in milliseconds. (default: 1000)
	consonant	The unstretched area of the render in milliseconds. (default: 0)
	cutoff		The cutoff from the end or from the offset for the render area of the sample. (default: 0)
	volume		The volume of the render in percentage. (default: 100)
	modulation	The pitch modulation of the render in percentage. (default: 0)
	tempo		The tempo of the render. Needs to have a ! at the start. (default: !100)
	pitch_string	The UTAU pitchbend parameter written in Base64 with RLE encoding. (default: AA)
```

# Example Renders

 The renders use straycat 0.2.1. No flags are used in these renders.

**Voicebank**: 櫻花アリス -吾亦紅- / Ouka Alice -Waremokou- / VCV

https://user-images.githubusercontent.com/29729824/214388141-07bc30e3-5068-421f-a871-10bc02b7ef02.mp4

**Voicebank**: 紅 通常 / Kurenai Normal / VCV

https://user-images.githubusercontent.com/29729824/214388215-fb27cbb4-4242-4423-a928-194a37bcb710.mp4

**Voicebank**: 戯白メリー 太神楽 / Kohaku Merry Daikagura / VCV

https://user-images.githubusercontent.com/29729824/214388523-7586efb7-f8e3-406e-bf77-23ba84d179f4.mp4

**Voicebank**: 匿名：MERGE / Tokumei MERGE / VCV

https://user-images.githubusercontent.com/29729824/214388931-60b11373-8ad9-44b2-b47f-c422fe15281a.mp4

**Voicebank**: 吼音ブシ-武- / Quon Bushi -武- / VCV

https://user-images.githubusercontent.com/29729824/214389016-0413db90-fdfb-4929-a4a0-f41f7eebbe3d.mp4

**Voicebank**: 松木マックス SPRUCE(NEO)v2.0 / Matsuki Max SPRUCE (NEO) v2.0 / VCV

https://user-images.githubusercontent.com/29729824/214389127-42546acc-2c77-4096-b6cd-1685e72ef1ee.mp4

**Voicebank**: 紅 地球 2.0 / Kurenai Earth 2.0 / CVVC

https://user-images.githubusercontent.com/29729824/214389485-14c6677d-8329-4c77-bc9a-90fb26f19de0.mp4

**Voicebank**: 学人デシマル χ / Gakuto Deshimaru Chi / CVVC

https://user-images.githubusercontent.com/29729824/214389797-9a2cdbf9-960d-4f1a-8efd-8266dcd1cacc.mp4

# straycat flags

 This is the main documentation of the flags available in straycat.

## Valued flags

### fe, fo, fl, fv, fp
 Fake vocal fry flag (and glottal stop flag).

 **fe(-inf, +inf)** is the main flag to enable this feature. fe is the length of the vocal fry in milliseconds. It adds the vocal fry so that the end of it is at the consonant point of the oto.
 
 **fo(-inf, +inf)** is the offset of the vocal fry in milliseconds. Negative values move it earlier. Default is 0.
 
 **fl[1, +inf)** is the length of the transition to vocal fry in milliseconds. Default is 75.

 **fv[0, 100]** is the volume of the vocal fry in percentage. Default is 10. Turning it into 0 makes a glottal stop.
 
 **fp[0, +inf)** is the pitch of the vocal fry. This is written in Hz. Default is 71.

### ve, vo
 Flag for fixing voicing.

 **ve(-inf, +inf)** is the main flag to enable this feature. ve is the length of the transition from voiced to unvoiced centered at the consonant point. Positive values make the area after the consonant unvoiced and negative values make the area before the consonant unvoiced.

 **vo(-inf, +inf)** is the offset of the transition from the consonant in milliseconds. Negative values move it earlier. Default is 0.

### g(-inf, +inf)
 Gender/Formant shift flag. Shifts the formant of the render, more commonly known as adding gender. 10 units in this flag is equivalent to pitching the sample a semitone without formant preservation and pitching it back with formant preservation.

### B[0, 100]
 Breathiness flag. Values lower than 50 lowers breathiness, but it does not have much effect. Values higher than 50 mixes an unvoiced render in, with 100 as being only the unvoiced render. Default is 50.

### P[0, 100]
 Peak normalization flag. This flag normalizes the sample to have the same peak volume for each note. At 0, this flag does not touch the volume of the render at all. Default is 86.
 
### A(-inf, +inf)
 Tremolo flag. This flag tries to isolate the vibrato from the pitchbend and modulates the volume based on this isolated vibrato, which means it may also react on drawn vibrato and more. Default is 0. This flag is applied after the peak normalization flag, so it may cause clipping issues.

### t(-inf, +inf)
 Pitch offset flag. One unit in this flag is a cent offset. It offsets the whole note so it might result in bad crossfades for VCV voicebanks. Default is 0

### S[0, 100]
 Max aperiodicity flag. This mixes in a max aperiodicity which kind of sounds like a weird whisper scream at lower frequencies. This is only experimental.

## Option flags

### G
 Force feature rerendering. This rerenders the cached file straycat reads which is the `.sc.npz` file. It is a regular Numpy compressed array file.
 
# Remarks
 This resampler is very slow considering it's written in pure Python. It would actually be pretty fast if it wasn't for Python's packages basically having so much stuff that it makes load times way slower. Python's nature of being an interpreted language might also be a big bottleneck, but Python itself has been considerably fast for me. This is just one of those cases... I could technically speed it up by doing an UTAU specific hack, but it might not work with OpenUtau anymore after this.
 
 I don't want to beat myself down that much for this but the slow speed very much ensures complete compatibility to both OpenUtau and classic UTAU. I guess you could say this would be a resampler in the olden days of single-thread resampling. The new Ameya resamplers are fast because of multiprocessing, and the other WORLD-based resamplers are fast because of their compiled nature. This resampler will always underperform because of Python's interpreted nature. Python was made for scripting after all, as much as so many AI models rely heavily on it.
