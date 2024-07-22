# straycat
 Yet another WORLD-based UTAU resampler.

### Other versions

Astel has made a server version that works with both UTAU and OpenUtau. It may render faster than regular straycat. You can get it [over here.](https://github.com/Astel123457/straycat-server)

I have made a new and improved version of [straycat implemented in Rust](https://github.com/UtaUtaUtau/straycat-rs/), which in nature should supersede straycat. This repository will be kept read-only for archiving purposes.

# How to use (development version)
 You need to have Python installed. This was made using Python 3.10.11.
 
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

## Running through terminal
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

## Running straycat server (UTAU and OpenUtau)
 1. To run straycat server, run the straycat server script `python straycat.py`.
 2. Put `StrayCatRunner.exe` in your Resampelers folder (OpenUtau only) or wherever else you keep your resamplers.
 3. Put `libcurl.dll` in the same folder as `StrayCatRunner.exe` otherwise it will error out and not render anything.
 4. Set `StrayCatRunner.exe` as your Resampeler or Tool2 and call to render.

# Example Renders

 The renders use straycat 0.3.1. No flags are used in these renders.

**Voicebank**: 櫻花アリス -吾亦紅- / Ouka Alice -Waremokou- / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/890c589e-5815-442e-bfaf-a9894cca6454

**Voicebank**: 紅 通常 / Kurenai Normal / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/308b9253-b69e-489b-81c7-ea614585be8a

**Voicebank**: 戯白メリー 太神楽 / Kohaku Merry Daikagura / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/85a4a7d4-1e2d-4946-b30c-32d259993246

**Voicebank**: 匿名：MERGE / Tokumei MERGE / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/6b78bc12-39d2-4e6a-a742-a9fb511d5853

**Voicebank**: 吼音ブシ-武- / Quon Bushi -武- / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/9e6164f1-8c93-49b1-9f33-7ea583ae6611

**Voicebank**: 松木マックス SPRUCE(NEO)v2.0 / Matsuki Max SPRUCE (NEO) v2.0 / VCV

https://github.com/UtaUtaUtau/straycat/assets/29729824/ba5f689d-29ba-4907-a54c-6583aa1d014c

**Voicebank**: 紅 地球 2.0 / Kurenai Earth 2.0 / CVVC

https://github.com/UtaUtaUtau/straycat/assets/29729824/8464ebd9-4349-43c3-ba81-6c8a64dcbaac

**Voicebank**: 学人デシマル χ / Gakuto Deshimaru Chi / CVVC

https://github.com/UtaUtaUtau/straycat/assets/29729824/e90a4159-77dc-4807-81a6-6c602143316c

**Voicebank**: CZloid / English VCCV (uses P0p-1 in CCs)

https://github.com/UtaUtaUtau/straycat/assets/29729824/ed9a1964-8dde-42c7-a4d8-596a9eb663bd

# straycat flags

See [Flags Documentation](flag_docs.md)

# Remarks
 This resampler is very slow considering it's written in pure Python. It would actually be pretty fast if it wasn't for Python's packages basically having so much stuff that it makes load times way slower. Python's nature of being an interpreted language might also be a big bottleneck, but Python itself has been considerably fast for me. This is just one of those cases... I could technically speed it up by doing an UTAU specific hack, but it might not work with OpenUtau anymore after this.
 
 I don't want to beat myself down that much for this but the slow speed very much ensures complete compatibility to both OpenUtau and classic UTAU. I guess you could say this would be a resampler in the olden days of single-thread resampling. The new Ameya resamplers are fast because of multiprocessing, and the other WORLD-based resamplers are fast because of their compiled nature. This resampler will always underperform because of Python's interpreted nature. Python was made for scripting after all, as much as so many AI models rely heavily on it.

 Running the server version is about 3.5x times faster than runnning the pure Python implementation. The server is Windows only until Aster can build for Linux and Mac.
