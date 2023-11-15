# straycat
 Yet another WORLD-based UTAU resampler.

# How to use (development version)
 You need to have Python installed. This was made using Python 3.8.10.
 
 You must install the needed libraries first, which are numpy, scipy, resampy, and pyworld. To do that, you may run a regular pip installation:
 
```
pip install numpy scipy resampy pyworld
```
 
## Running in UTAU
 1. Download the `straycat.py` file and put it somewhere.
 2. Setup your `.ust` file to have the proper voicebank and wavtool selected.
 3. Open your `.ust` file as a text file with whichever text editor.
 4. Change the `Tool2` resampler to the path of `straycat.py`.

 You can now open the `.ust` and use `straycat.py` as a resampler. You need to press cancel in the project properties when UTAU shows the project properties panel.
 
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

# How to use (release version)
## Classic UTAU
 1. Download `straycat.7z` for the version you want to use.
 2. Extract the 7z archive anywhere.
 3. Set your resampler to `straycat.exe` in Project Properties.

 You can now use straycat in classic UTAU.

## OpenUtau
 1. Download `straycat.7z` for the version you want to use.
 2. Extract the 7z archive in the `Resamplers` folder of OpenUtau.
 3. Select the resampler in whichever way you prefer.

 You can now use straycat in OpenUtau.

# Addendum

Astel has made a server version that works with both UTAU and OpenUtau. It may render faster than regular straycat. You can get it [over here.](https://github.com/Astel123457/straycat)

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

See [Flags Documentation](flag_docs.md)

# Remarks
 This resampler is very slow considering it's written in pure Python. It would actually be pretty fast if it wasn't for Python's packages basically having so much stuff that it makes load times way slower. Python's nature of being an interpreted language might also be a big bottleneck, but Python itself has been considerably fast for me. This is just one of those cases... I could technically speed it up by doing an UTAU specific hack, but it might not work with OpenUtau anymore after this.
 
 I don't want to beat myself down that much for this but the slow speed very much ensures complete compatibility to both OpenUtau and classic UTAU. I guess you could say this would be a resampler in the olden days of single-thread resampling. The new Ameya resamplers are fast because of multiprocessing, and the other WORLD-based resamplers are fast because of their compiled nature. This resampler will always underperform because of Python's interpreted nature. Python was made for scripting after all, as much as so many AI models rely heavily on it.
