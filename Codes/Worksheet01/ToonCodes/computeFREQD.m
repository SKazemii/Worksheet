function feature = computeFREQD(COPTS,varargin)
    
    % feature = computeFREQD(COPTS,varargin)
    % FREQD : Frequency Dispersion
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : FREQD_RD, FREQD_AP, FREQD_ML
    
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
    % The first spectral moment
    m1 = sum(Pxx(idx_range,:).*F(idx_range));
    % The zeroth spectral moment
    m0 = sum(Pxx(idx_range,:));
    % Zero for a pure sinusoid and increases with spectral bandwidth to a maximum to one
    feature = sqrt(1-((m1.^2)./(m0.*m2)));
    
end

