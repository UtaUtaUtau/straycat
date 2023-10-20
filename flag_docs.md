# straycat flags

 This is the main documentation of the flags available in straycat.

# Value Notation for valued flags
 The notation for the ranges of each flag follows interval notation from mathematics. Square brackets **[]** mean that the value is included within the range, while parentheses **()** mean that the value is not included within the range. **inf** means infinity.

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

### P[0, 100)
 Peak compressor flag. This flag compresses the render to have a more consistent volume throughout the render. At 0, this flag does not compress the render at all. Default is 86.

### p[0, +inf)
 Peak normalization flag. This flag normalizes the sample to make the peak of the sample -p dB, where p is the value set for the flag. Negative values skip this normalization step. Default is 6.
 
### A(-inf, +inf)
 Tremolo flag. This flag tries to isolate the vibrato from the pitchbend and modulates the volume based on this isolated vibrato, which means it may also react on drawn vibrato and more. Default is 0. This flag is applied after the peak normalization flag, so it may cause clipping issues.

### t(-inf, +inf)
 Pitch offset flag. One unit in this flag is a cent offset. It offsets the whole note so it might result in bad crossfades for VCV voicebanks. Default is 0

### S[0, 100]
 Max aperiodicity flag. This mixes in a max aperiodicity which kind of sounds like a weird whisper scream at lower frequencies. This is only experimental.

## Option flags

### G
 Force feature rerendering. This rerenders the cached file straycat reads which is the `.sc.npz` file. It is a regular Numpy compressed array file.
