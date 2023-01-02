# Changelog

## [dev] - 2023-01-03

## Fixed
 - Fixed A flag not outputting anything.

## [0.2.0] - 2023-01-02

## Added
 - Added the g flag.

## Changed
 - Changed the algorithm of `base_frq` to be more sensitive to flatter regions.
 - Changed peak normalization volume to make renders less loud and make early clipping for the A flag happen less.
 - Sped up functions using `numba`. Might've helped feature generation and some flags. The functions sped up and their usages are:
	- `smoothstep(edge0, edge1, x)`
		- Used for flags with "envelopes" (e.g. vocal fry flag, voicing flag)
	- `clip(x, x_min, x_max)`
		- Used generally for clipping things within range. `numpy.clip` exists but for some reason numba made it faster.
	- `base_frq(f0, f0_min=None, f0_max=None)`
		- Used to get the base frequency from the frequency information. Feature generation might be a little faster because of this.
 - Reworked timing system to be more robust.
 
## [0.1.3] - 2022-12-27

## Changed
 - Changed P flag default to 86. Makes it so that the A flag doesn't clip as much. Still does though.

## [0.1.2] - 2022-12-26

## Fixed
 - Skipping rendering if out file is `nul` to not pull up an error when generating `.sc.npz` files through the short frq generation commant UTAU uses. It still generates the `.sc.npz` but I just don't like how unclean it is...

## [0.1.1] - 2022-12-24

### Added
 - This changelog!
 - An icon for the executable version. Icon is [Cat icons created by Freepik - Flaticon](https://www.flaticon.com/free-icons/cat)

### Changed
 - Changed flag order for A and P flag. Peak normalization is applied before tremolo now. May cause clipping issues.

## [0.1.0] - 2022-12-23

### Added
 - First release.
