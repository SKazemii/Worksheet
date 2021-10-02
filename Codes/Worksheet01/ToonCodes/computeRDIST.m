function feature = computeRDIST(COPTS)
    
    % feature = computeRDIST(COPTS)
    % RDIST : RMS Distance
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : RDIST_RD, RDIST_AP, RDIST_ML
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    feature = rms(p.Results.COPTS);
    
end