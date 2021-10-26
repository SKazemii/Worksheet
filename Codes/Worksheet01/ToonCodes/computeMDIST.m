function feature = computeMDIST(COPTS)
    
    % feature = computeMDIST(COPTS)
    % MDIST : Mean Distance
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : MDIST_RD, MDIST_AP, MDIST_ML
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    feature = mean(abs(p.Results.COPTS));
    
end