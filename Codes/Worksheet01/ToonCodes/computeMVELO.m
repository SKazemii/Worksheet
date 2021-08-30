function feature = computeMVELO(COPTS,varargin)
    
    % feature = computeMVELO(COPTS,varargin)
    % MVELO : Mean Velocity
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : MVELO_RD, MVELO_AP, MVELO_ML
    
    defaultT = 1; % the period of time selected for analysis (CASIA-D = 1s)
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    addParameter(p,'T',defaultT);
    parse(p,COPTS,varargin{:});
    
    feature(:,1) = sum(sqrt(diff(p.Results.COPTS(:,2)).^2+diff(p.Results.COPTS(:,3)).^2))./p.Results.T;    
    feature(:,2:3) = sum(abs(diff(p.Results.COPTS(:,2:3))))./p.Results.T;
    
end