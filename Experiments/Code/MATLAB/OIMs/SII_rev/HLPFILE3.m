Audiogram (optional):

Select "AUDIOGRAM" and "LOAD Audiogram" from the "AI-Setup" 
menu if the audiogram of the listener deviates from 0 dB HL.
(See Fletcher 1953 p. 411 and p. 427 for normal pure-tone
thresholds.) Remember that the procedure was developed for
normally hearing listeners. Hence the audiogram should deviate
only minimally from 0 dB HL. If "LOAD Audiogram" is selected
again, the audiogram will be replaced with that specified in
the new selection.  

If no selection is made or if a  previously loaded audiogram is
deleted by selecting "Audiogram" and "Assume 0 dB HL" from the
"AI-Setup" menu, a flat audiogram of 0 dB HL is assumed.

The audiogram is to be provided in the format described for the 
filter response but the response is replaced by the
threshold values in dB HL. The audiogram in dB will be linearly
interpolated on a logarithmic frequency scale to yield the
threshold elevations at the frequencies specified in vector f0.

The currently active audiogram can be displayed by
making the appropriate selection in "Review Parameters".

The audiogram is used to calculate the hearing loss for
speech, beta_H. It is calculated according to Eq.17-82 of Fletcher
1953 (p. 413).