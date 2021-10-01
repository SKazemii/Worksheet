function feature = computeRANGE(COPTS)
    
    % feature = computeRANGE(COPTS)
    % RANGE : Range
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : RANGE_RD, RANGE_AP, RANGE_ML
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    feature(:,1) = max(pdist(COPTS(:,2:3)));
    feature(:,2:3) = range(COPTS(:,2:3));
    
end