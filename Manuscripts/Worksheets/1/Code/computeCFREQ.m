function feature = computeCFREQ(COPTS,varargin)
    
    % feature = computeCFREQ(COPTS,varargin)
    % CFREQ : Centroidal Frequency / Zero Crossing Frequency
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : CFREQ_RD, CFREQ_AP, CFREQ_ML
    
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
    % Find the index for the low and high frequency range
    idx_range = (find(F>p.Results.FreqRange(1),1):find(F<p.Results.FreqRange(2),1,'last'));
    % The second spectral moment
    m2 = sum(Pxx(idx_range,:).*(F(idx_range).^2));
    % The zeroth spectral moment
    m0 = sum(Pxx(idx_range,:));
    % The square root of the ratio of the second to the zeroth spectral moments
    feature = sqrt(m2./m0);
    
end

