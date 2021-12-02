Load Orthotelephonic Response:

Select "Load RESPONSE" from the "AI-Setup" menu to load an 
orthotelephonic response into the workspace. The orthotelephonic
response is, "at each frequency, ... equal to the difference in
dB between the transmission supplied by the … system and that
supplied by the orthotelephonic reference system. This reference
system consists of the air path between the talker and a listener,
using one ear, who faces the talker in an otherwise free acoustic
field at a distance of one meter between the lips of the talker
and the aural axis of the listener." (Fletcher 1953, p. 303) 

The response can be replaced by selecting "Load RESPONSE" again.
 
The response is to be provided in a text file of two columns: The
First column specifies the frequencies [Hz] at which the response
is sampled. The second column, which is separated from the first
by a "TAB", contains the filter response [dB] at the corresponding
frequencies. Such a file is created easily by saving a spreadsheet
(e.g. Excel) where frequencies and response data are arranged in
two columns as TAB-delimited text. The lowest frequency at which
the response is specified must be between 100 and 200 Hz and the
highest frequency between 8000 and 10000 Hz. The response in dB is
linearly interpolated on a logarithmic frequency scale to yield the
response at the frequencies specified in the vectors f0, f1, f2,
and f3, which are defined in "FAI.m".

The currently active system response can be displayed by making the
appropriate selection in "Review Parameters".

