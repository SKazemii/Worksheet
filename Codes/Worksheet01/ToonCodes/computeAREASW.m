function feature = computeAREASW(COPTS,varargin)
    
    % feature = computeAREASW(COPTS,varargin)
    % AREASW : Sway area
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [1] : AREA-SW

    defaultT = 1; % the period of time selected for analysis (CASIA-D = 1s)
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    addParameter(p,'T',defaultT);
    parse(p,COPTS,varargin{:});
    
    AP = p.Results.COPTS(:,2);
    ML = p.Results.COPTS(:,3);
    feature = sum(abs((AP(2:end).*ML(1:end-1))-(AP(1:end-1).*ML(2:end))))./(2*p.Results.T);
    
end