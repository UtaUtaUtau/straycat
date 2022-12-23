# straycat flags

 This is the main documentation of the flags available in straycat.

## Valued flags

### fe, fl, fo, fv
 Fake vocal fry flag (and glottal stop flag).

 **fe(-inf, +inf)** is the main flag to enable this feature. fe is the length of the vocal fry in milliseconds. It adds the vocal fry so that the end of it is at the consonant point of the oto.

 **fl[1, +inf)** is the length of the transition to vocal fry in milliseconds. Default is 75.

 **fo(-inf, +inf)** is the offset of the vocal fry in milliseconds. Negative values move it earlier. Default is 0.

 **fv[0, 100]** is the volume of the vocal fry in percentage. Default is 10. Turning it into 0 makes a glottal stop.

### ve, vo
 Flag for fixing voicing.

 **ve(-inf, +inf)** is the main flag to enable this feature. ve is the length of the transition from voiced to unvoiced centered at the consonant point. Positive values make the area after the consonant unvoiced and negative values make the area before the consonant unvoiced.

 **vo(-inf, +inf)** is the offset of the transition from the consonant in milliseconds. Negative values move it earlier. Default is 0.

### B[0, 100]
 Breathiness flag. Values lower than 50 lowers breathiness, but it does not have much effect. Values higher than 50 mixes an unvoiced render in, with 100 as being only the unvoiced render. Default is 50.

### A(-inf, +inf)
 Tremolo flag. This flag tries to isolate the vibrato from the pitchbend and modulates the volume based on this isolated vibrato, which means it may also react on drawn vibrato and more. Default is 0.

### P[0, 100]
 Peak normalization flag. This flag normalizes the sample to have the same peak volume for each volume. At 0, this flag does not touch the volume of the render at all. Default is 100.

### t(-inf, +inf)
 Pitch offset flag. One unit in this flag is a cent offset. It offsets the whole note so it might result in bad crossfades for VCV voicebanks. Default is 0

## Option flags

### G
 Force feature rerendering. This rerenders the cached file straycat reads which is the `.sc.npz` file. It is a regular Numpy compressed array file.