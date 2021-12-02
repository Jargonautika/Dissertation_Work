%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         ARTICULATION INDEX                           %
%  Model by H. Fletcher (1953, Speech and Hearing in Communication)    %
%                   20-Band Method: Case II (p.378)                    %
%                                                                      %
%                        MATLAB 5.1 and higher                         %
%                  (Student or Professional Edition)                   %
%                   Copyright 1999 by Hannes Muesch                    %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This software implements the Articulation Index Model as described by
H. Fletcher (1953) "Speech and Hearing in Communication." on pp. 378
(Case II). Unless indicated otherwise, all references in the online help
are with respect to Fletcher, H. (1953). 

It should be noted that the AI model was developed and optimized on
the systems whose responses are depicted on pages 382 to 394 of
Fletcher (1953). When the procedure is supplied with extreme and
unreasonable input, the predictions are almost certainly invalid. For
example, when the IDEAL low-pass filter with a cutoff frequency of 
200 Hz and a stop band attenuation of 500 dB is subjected to an AI
calculation, perfect performance is predicted.  According to the dicrete
implementation of the model, speech is brought to detection threshold at
a gain of 444 dB, and the AI of the system raises to unity when the gain
is approximately 515 dB. This result is understandable once one recognizes
that the model does not take into consideration speech energy at
frequencies below 200 Hz. Only the flat stop band is visible to the
procedure and when the high stop-band attenuation is overcome by an
appropriate amount of gain, performance is perfect.

Although most parameters of the model are defined as functions
continuous in frequency, the model was implemented and tested as a 20
band discrete approximation. In particular, the model is calculated in
frequency bands whose bounds are placed so that the integral of any
weighting function over any band is 1/20th of the area under the entire
weighting function. The effect of the discrete approximation can be
observed when filters with unreasonably steep transition bands are
evaluated. For example, when the pass band of the IDEAL low pass discussed
above is extended to 400 Hz, the predicted maximum achievable gain is
reduced. This is because for the calculation of R1, the frequency range
from 510 Hz to 3960 Hz is used (vector f1) whereas the frequency range
between 320 Hz and 6600 Hz (vector f2) is used for the calculation of R4.
Consequently, R1 is calculated as though the filter were an ideal
all pass whereas R4 is calculated as if the filter were a low pass. In
this example, R1 is -500 dB and R4 is -52 dB. The result is a flat PI
function that reaches a maximum AI of 0.0486 at a gain of
approximately 520 dB.

While these effects do exist, the model does a remarkable job of
predicting intelligibility of systems with "well behaved"
transfer functions.
