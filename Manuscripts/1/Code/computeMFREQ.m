function feature = computeMFREQ(COPTS,varargin)
    
    % feature = computeMFREQ(COPTS,varargin)
    % MFREQ : Mean Frequency (or Rotational Frequency)
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : MFREQ_RD, MFREQ_AP, MFREQ_ML
    
    defaultT = 1; % the period of time selected for analysis (CASIA-D = 1s)
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    addParameter(p,'T',defaultT);
    parse(p,COPTS,varargin{:});
    
    TOTEX = computeTOTEX(p.Results.COPTS);
    MDIST = computeMDIST(p.Results.COPTS);
    
    feature(:,1) = TOTEX(:,1)./(2*pi*p.Results.T*MDIST(:,1));
    feature(:,2) = TOTEX(:,2)./(4*sqrt(2)*p.Results.T*MDIST(:,2));
    feature(:,3) = TOTEX(:,3)./(4*sqrt(2)*p.Results.T*MDIST(:,3));
    
end