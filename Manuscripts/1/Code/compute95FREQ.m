function feature = compute95FREQ(COPTS,varargin)
    
    % feature = compute95FREQ(COPTS,varargin)
    % 95FREQ : 95% POWER FREQUENCY / Median Power Frequency
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : 95FREQ_RD, 95FREQ_AP, 95FREQ_ML
    % Function Required : interpPower.m, interpFreq.m
    
    defaultSampFreq = 100; % fs
    defaultFreqRange = [0.15,5];
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    addParameter(p,'SampFreq',defaultSampFreq);
    addParameter(p,'FreqRange',defaultFreqRange);
    parse(p,COPTS,varargin{:});
    
    m = 8;
    NFFT = 2^nextpow2(length(p.Results.COPTS));
    % Multitaper Power Spectral Density Estimate
    [Pxx,F] = pmtm(p.Results.COPTS,m,'Tapers','sine',NFFT,p.Results.SampFreq);
    % Return the frequency widths of each frequency bin
    width = signalwavelet.internal.specfreqwidth(F);
    % Multiply the PSD by the width to get the power within each bin
    P = bsxfun(@times,width,Pxx);
    % Cumulative rectangular integration
    cumPwr = [zeros(1,size(P,2),'like',P); cumsum(P,1)];
    % Place borders halfway between each estimate
    cumF = [F(1,1); (F(1:end-1,1)+F(2:end,1))/2; F(end,1)];
    % Find the integrated power for the low and high frequency range
    Plo = interpPower(cumPwr,cumF,p.Results.FreqRange(1)); 
    Phi = interpPower(cumPwr,cumF,p.Results.FreqRange(2));
    % Return the power between the frequency range % Unit : dB/Hz
    Pwr = Phi-Plo;
    % Return the frequency below which 95% of the total power is found
    feature = interpFreq(cumPwr,cumF,Plo+(Pwr*0.95));
    
end

