================ README.TXT ================

ARTICLE INFORMATION:         E-ARLOFJ-2-005101

JOURNAL:
     Acoustic Research Letters Online (ARLO), January 2001, Vol. 2, No. 1, pages: 25-30

AUTHOR:  Hannes Muesch

TITLE:    
"Review and computer implementation of Fletcher and Galt's method of calculating the Articulation Index"

DEPOSIT INFORMATION:

DESCRIPTION:  
This directory contains 23 script files for a MATLAB implementation of the method of calculating the
Articulation Index as described by H. Fletcher (1953) in "Speech and Hearing in "Communication" on pp. 378
(Case II - noise - no special types of distortion). The book can be ordered from the Acoustical Society of
America: (http://asa.aip.org/publications.html#pub09A)
These scripts (ascii files) are identified by the extension "*.m".
Also provided are 9 ascii files with test data (extension "*.txt").
You need MATLAB Version 5.1 or higher to run these scripts.

Bugs and software problems should be reported to 
     HMuesch@GNReSound.com

Total No. of Files: 33
23  m-files (MATLAB code)
9   txt-files (Test data)
1   txt-file   (Read.me)

The filenames are as follows

m-files:
Aivsgain.m
Ap2s2.m
Ap2s23.m
Ap2s3.m
Ap2s3m.m
Delmsk.m
Delsnl.m
FAI.m
Hlpfile1.m
Hlpfile2.m
Hlpfile3.m
Hlpfile4.m
Hlpfile5.m
Hlpfile6.m
Ldnoise.m
Ldresp.m
Ldthrsn.m
Prmsetup.m
Rmsk.m
Rresp.m
Rsnl.m
Savedata.m
Startai.m

txt-files
H2.txt              (System H-2, see Fig. 215)
H3.txt              (System H-3, see Fig. 216)
IIIRL5.txt          (System III-RL-5, see Fig. 200)
IIILP4500.txt       (System III-LP-4500, see Fig. 196)
SII.txt             (System II, see Fig. 187)
SIII.txt       (System III, see Fig. 188)
NoiseA.txt          (Noise A, see Fig. 215)
NoiseB.txt          (Noise B, see Fig. 215)
ExpAudg.txt         (Example audiogram of normal-hearing listener)

It is essential that the names of the script files are exactly the same as those listed above. In particular, make
sure the extension "m" is lower case. 

INSTALLATION:
Copy all 23 .m files to a subdirectory within the search path of MATLAB and type "startAI" at the MATLAB
prompt. This will open a window with a menu bar. If you are running a version of MATLAB that
is older than 5.1, you will not be allowed to continue program execution because your MATLAB version does
not support the user interface. You can still use the script "FAI.m", which implements the core procedure,
but you will have to define parameters and load data by evoking the appropriate commands in MATLAB's
command window.

The scripts have been developed and tested in MATLAB 5.1 on a Macintosh computer and have been run
successfully in MATLAB 5.2 on the Windows 95 and 98 operating systems. 

CONTACT INFORMATION
Hannes Muesch
HMuesch@GNReSound.com
Tel.: 650 261 2259    FAX.: 650 261 2284

STEP-BY-STEP INSTRUCTIONS FOR PROGRAM EXECUTION
Provided together with the MATLAB scripts are data files that can be used to test if your installation was
successful. They also demonstrate the data format. We will use these files to demonstrate how to use the
software by making simple calculations.

After typing "startAI" on the MATLAB prompt, you will see an empty window
with a menu bar (on the MAC, the menu bar will be on top of the screen.)
Use that menu bar for program operation. 

The first step is always to load a System Response in MATLAB's workspace.
To do so, select "Load RESPONSE" from the "AI-Setup" menu. This will open
an interactive box. Browse until you find "H3.txt" and select "open". This
will load the orthotelephonic response of system H3 into the workspace.
The response should now be displayed on the screen. 

Orthotelephonic response:
"The performance of any telephone system is here expressed in terms of its orthotelephonic response, which at
each frequency is equal to the difference in dB between the transmission supplied by the telephone system and
that supplied by the orthotelephonic reference system. This reference system consists of the air path between
the talker and a listener, using one ear, who faces the talker in an otherwise free acoustic field at a distance of
one meter between the lips of the talker and the aural axis of the listener." (p. 303 of Fletcher, 1953)

File format:
The file H3.txt is a TAB-delimited text file that holds samples of the orthotelephonic response of system H3.
The response is organized in two columns. The first column specifies the frequencies [Hz] at which the
response is sampled. The second column, which is separated from the first by a "TAB", contains the
filter response [dB] at the corresponding frequencies. Such a file is created easily by saving a spreadsheet (e.g.
MS Excel) where frequencies and response data are arranged in two columns as TAB-delimited text. The
lowest frequency at which the response is specified must be between 100 and 200 Hz and the highest
frequency between 8000 and 10000 Hz. The software linearly interpolates the response in dB on a logarithmic
frequency scale to yield the response at the frequencies specified in the vectors f0, f1, f2, and f3, which are
defined in "FAI.m".

Now you are ready to calculate the AI for a number of system gains. Select "CALCULATE NOW" from the
"Fletcher AI" menu to initiate the AI calculation. The AI is plotted as a function of the gain alpha of an
amplifier that delivers the speech signal to the filter. The values for the talking level and the hearing loss for
speech that were used in the calculation are displayed in the upper-left corner of the plot. Since we did not
make any special selections, you will see default values. The filename of the response is displayed in the title
line of the plot. The title also indicates that we have not chosen any masking noise, or any particular
audiogram. 

If you want to see a percent score rather than an AI value, convert the AI-versus-gain plot into a
performance-versus-gain plot by making the appropriate selections in the "Fletcher AI" menu. The
transformation functions are taken from Fletcher (1953).
These displays are useful for comparison with the model predictions published in Fletcher's book (1953, Figs.
196 to 218)

S3 vs Gain shows the predicted CVC score as a function of gain. 
s3M vs Gain shows the predicted phoneme score as a function of gain.
S2 vs Gain shows the predicted VC/CV score as a function of gain.
S23 vs Gain shows the predicted score when 1/5 of the sounds are of the type CV
and 4/5 are of the CVC type.
See Fletcher, 1953, Chapter 15, particularly Fig. 179 on page 301.

The transformation depends on the test-taking proficiency of the talker-listener pair.
The proficiency factor used in the transformation is also displayed in the upper left corner of the plot (the
default is "p=1.0"). See page 282 of Fletcher (1953) for a discussion of the proficiency factor.

We assumed in our calculation that the talking level, beta_t, is 68 dB (default). 

Talking Level, beta_t:
"The talking level is an over-all measure of the acoustic level and is here defined
to be the long-time average intensity level of speech at a point one meter directly in front
of the talker's lips in a free coustic field and is designated beta_t. The long-time average
intensity is the average taken over a length of time sufficient to include the typical
pauses between syllables and words" (p. 313 of Fletcher, 1953).

Suppose the actual talking level is 70 dB. We would like to redo the calculation with
the talking level equal to 70 dB. Select "Set 'bt', 'bH', and 'p'. .." from the "AI-Setup"
menu. A dialog box will pop up in which you can adjust the talking level, among other
parameters. Type 70 dB in the appropriate box and click OK. Notice that the display
has not changed. For the new parameter to have effect, you must select "Calculate NOW"
from the "Fletcher AI" menu. After initiating the new calculation, the parameter beta_t is in effect (see the
inset in the plot). Also observe the shift of the AI-versus-gain curve towards lower gains. This shift can be
easily observed in alpha_0. Alpha_0 is the gain at which the AI becomes zero (i.e., the gain that brings speech
to detection threshold).
For each calculation, alpha_0 is displayed in the MATLAB command window together with several other
internal variables of the procedure. If you compare the last two printouts, you will notice that the alpha_0s
differ by 2dB. The lowering of alpha_0 mirrors the increase in talking level beta_t.  
So far our calculations have assumed that the speech was presented in quiet. We can also calculate the
AI-vs-gain function in the presence of a masking noise. Following Fletcher's example in Fig 216, we will
calculate the AI of System H3 in the presence of noise A. From the "AI-Setup" menu, select "MASKER" and
"LOAD Masker". Browse until you find "noiseA.txt", and click "open". The spectral density level of the noise
will be displayed. You can compare this plot with Fig. 215, where the spectrum level of noise A is also shown.

File format:
The spectrum level of the masker is to be provided in the same format as the filter
response, except that the spectrum level in dB SPL/Hz replaces the response. MATLAB linearly
interpolates the spectrum level on a logarithmic frequency scale to yield the masker spectrum
level at the frequencies specified in vector f0.

Before we recalculate the AI, we notice that Fletcher's listeners had a hearing loss for
speech that was beta_H = -4dB. To make our calculation comparable to Fletcher's prediction,
we too will set beta_H to -4dB. To do so, select "Set 'bt', 'bH', and 'p'. .." from the "AI-Setup"
menu and set p = 1, beta_t = 70, and beta_H = -4. Click OK and recalculate the AI-vs-Gain
curve by selecting "Calculate NOW" from the "Fletcher AI" menu. (NOTE: We entered the hearing loss for
speech, beta_H, directly into the dialog box. We will see later that beta_H can be estimated from the
audiogram and we could, alternatively, have loaded the average audiogram of Fletcher's listening crew into
the workspace and the software would have calculated the beta_H for us.) Fletcher plotted his results in terms
of the predicted articulation S2. To get a comparable plot, select "Display S2 Score vs Gain" from the
"Fletcher AI" menu. The resulting plot should match the curve that connects the open circles in the graph on
the lower left of Fig. 216. The open circles are measured data points and the solid line is the prediction that
Fletcher derived with pencil and paper. It should match the prediction on your screen.

To calculate the AI of System H3 in the presence of noise B, load the spectrum of noise B in the workspace
and recalculate the AI. By selecting "LOAD Masker", the newly loaded masker will replace the previous
masker. You can verify this any time by selecting "REVIEW Parameters" and "Display Masker Spectrum"
from the "AI Setup" menu.

Noises A and B are both very smooth and spread of masking is not a concern with these noises.
For noises with steeply sloping spectra, however, we must supply the procedure with the  spectrum level of a
broadband masker that produces the masked thresholds observed with the actual masker. The masker
spectrum is the masker spectrum at the listener's ear. The masker is NOT filtered by the active response.

Hearing loss for speech, beta_H:
In the context of this procedure, the hearing loss for speech, beta_H, is calculated as a
weighted average of the listener's audiogram (p. 413, Fletcher 1953). For listeners with a flat
audiogram of 0 dB HL, beta_H will be 0 dB. If the audiogram deviates from this baseline, beta_H
will assume values other than zero. It is important to remember that the procedure was developed
for normally hearing listeners and that the threshold correction should only be used to account for
small deviations from the 0-dB-HL norm. The hearing loss for speech, beta_H, was not intended to account
for hearing loss.
Fletcher suggested a strategy for incorporating hearing impairment into the AI calculation. This
strategy essentially assumes that the sensori-neural part of the loss can be modeled by a masking
noise that produces in normal-hearing listeners a masked threshold that equals the quiet threshold
of the hearing-impaired listener. The conductive component is modeled as an additional attenuation that is
incorporated into the response R(f). Fletcher (1953) describes this rationale in Chapter 19 of his book in the
context of a simplified version of the AI model. This rationale could easily be incorporated into the current
software.

In most cases we will not know beta_H, but we will know the listener's audiogram. To calculate
beta_H, load the listener's audiogram in the workspace by selecting "AUDIOGRAM" and "LOAD
Audiogram" from the "AI Setup" menu. When you recalculate the AI with the audiogram loaded, the software
will estimate beta_H and allpy it to the calculation. The estimate is also displayed in the inset of the graph and
in MATLAB's command window. 


Several small points:

(1)
If the orthotelephonic response of the transmission system has a very sharp peak,
(narrower than 180 Hz), then the response must be widened:

"Since masking occurs principally when voiced sounds are used, we can consider the components of the
voices of men and women as spaced 180 cycles apart as an average. Consequently, in a frequency region
where the Z versus log f curve has a peak, the curve is regarded as flat over a band 180 cycles wide and the
ordinate is taken to be the average ordinate over the 180 cycle band. The same also applies to filters having
band widths less than 180 cycles .. ." (Fletcher 1953, p. 329)

(2)
When the procedure is supplied with extreme and unreasonable input, the predictions are almost certainly
invalid. For example, when the IDEAL low-pass filter with a cutoff frequency of 200 Hz and a stop band
attenuation of 500 dB is subjected to an AI calculation, perfect performance is predicted.  According to the
discrete implementation of the model, speech is brought to detection threshold at a gain of 444 dB, and the AI
of the system raises to unity when the gain is approximately 515 dB. This result is understandable once one
recognizes that the model does not take intoconsideration speech energy at frequencies below 200 Hz. Only
the flat stop band is visible to the procedure and when the high stop-band attenuation is overcome by an
appropriate amount of gain, performance is perfect.

Although most parameters of the model are defined as functions continuous in frequency, the model was
implemented and tested as a 20 band discrete approximation. In particular, the model is calculated in
frequency bands whose bounds are placed so that the integral of any weighting function over any band is
1/20th of the area under the entire weighting function. The effect of the discrete approximation can be
observed when filters with unreasonably steep transition bands are evaluated. For example, when the pass
band of the IDEAL low pass discussed above is extended to 400 Hz, the predicted maximum achievable AI is
reduced. This is because for the calculation of R1, the frequency range from 510 Hz to 3960 Hz is used
(vector f1) whereas the frequency range between 320 Hz and 6600 Hz (vector f2) is used for the calculation
of R4. Consequently, R1 is calculated as though the filter was an ideal all pass whereas R4 is calculated as if
the filter was a low pass. In this example, R1 is -500 dB and R4 is -52 dB. The result is a flat PI function that
reaches a maximum AI of 0.0486 at a gain of approximately 520 dB.

While these effects do exist, the model does a remarkable job of predicting intelligibility of systems with "well
behaved" transfer functions. 

==============================================================================