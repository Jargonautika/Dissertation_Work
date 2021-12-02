Masker (optional):

This option allows to calculate the AI under masking.

For the AI calculation to be valid, the masker spectrum must be
sufficiently smooth so that spread of masking can be neglected.
If spread of masking cannot be neglected, the "effective masker
spectrum", i.e., the spectrum of a smooth masker that would produce
a masking pattern identical to that expected by the sharply filtered
masker, must be submitted to the calculation instead.

To calculate the AI under masking, select "MASKER" and "Load MASKER"
from the "AI-Setup" menu to load the spectrum level of the masking
noise at the listener's ear.  If "Load MASKER" is selected again,
the current masker spectrum will be replaced by the new selection.

If no masker is selected, or if a previously loaded masker has been
deleted by selecting "MASKER" and "Delete MASKER" from the "AI-Setup"
menu, the spectrum level will be set to -500 dB SPL/Hz. Results obtained
with this masker are virtually indistinguishable from the results 
obtained with Fletcher's method "Case I: No Noise".

The spectrum level of the masker is to be provided in the same format as
the filter response, except that the response is replaced by the 
spectrum level in dB SPL/Hz.  MATLAB linearly interpolates the spectrum 
level on a logarithmic frequency scale to yield the masker spectrum 
level at the frequencies specified in vector f0.

The currently active masker spectrum can be displayed by making the
appropriate selection in "Review Parameters".

