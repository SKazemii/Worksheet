function feature = computeFDPD(COPTS)
    
    % feature = computeFDPD(COPTS)
    % FD-PD : Fractal Dimension based on the Plantar Diameter of the Curve
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : FD-PD_RD, FD-PD_AP, FD-PD_ML
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    N = size(p.Results.COPTS,1);
    TOTEX = computeTOTEX(p.Results.COPTS);
    d = computeRANGE(p.Results.COPTS);
    
    feature = log(N)./log((N.*d)./TOTEX);
    
end