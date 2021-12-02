function csv = dwgpCSV(inputFolder, destFolder, noiseFolder, SNR)
% inputFolder should point to waveforms sampled to 16khz
% destFolder should point to a landing space for the CSVs
% noiseFolder should contain speech modulated noise or speech shaped noise segments of the same shape as same-named files in inputFolder
% SNR is an integer value used for calculating the scaling factor during SNR leveling

inputFiles = dir(inputFolder);
noiseFiles = dir(noiseFolder);

for k = 3 : length(inputFiles)

    fullInputName = fullfile(inputFiles(k).folder, inputFiles(k).name);
    fullNoiseName = fullfile(noiseFiles(k).folder, noiseFiles(k).name);
    
    [sig,fs] = audioread(fullInputName);
    [noise,fs] = audioread(fullNoiseName);

    % Calculate a scaling factor
    s = sqrt((sum(sig.^2)./sum(noise.^2))./(10.^(SNR/10)));
    scaledNoise = noise.*s;

    objscore = DWGP(sig, scaledNoise, fs);

    base = strrep(inputFiles(k).name, '.wav', '.csv');
    dest = strcat(destFolder, '/', base);
    writetable(struct2table(objscore), dest);

end
