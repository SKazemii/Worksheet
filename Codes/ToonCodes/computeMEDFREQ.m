function feature = computeMEDFREQ(COPTS,varargin)
    
    % feature = computeMEDFREQ(COPTS,varargin)
    % MEDFREQ : 50% POWER FREQUENCY / Median Power Frequency
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : MEDFREQ_RD, MEDFREQ_AP, MEDFREQ_ML
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
    % Return the frequency that divides the power equally
    feature = interpFreq(cumPwr,cumF,(Plo+Phi)/2);
    
end

