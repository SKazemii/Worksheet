function feature = computeFDCC(COPTS)
    
    % feature = computeFDCC(COPTS)
    % FD-CC : Fractal Dimension based on the 95% Confidence Circle
    % Only RD time series is used in the paper.
    % COPTS [t,3] : RD (AP, ML) COP time series
    % feature [1] : FD-CC
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    N = size(p.Results.COPTS,1);
    MDIST = mean(abs(p.Results.COPTS(:,1)));
    RDIST = rms(p.Results.COPTS(:,1));
    z05 = 1.645; % z.05 = the z statistic at the 95% confidence level
    SRD = sqrt(RDIST.^2-MDIST.^2); % the standard deviation of the RD time series
    d = 2*(MDIST+z05*SRD);
    TOTEX = computeTOTEX(p.Results.COPTS);
    
    feature = log(N)./log((N.*d)./TOTEX(1));
    
end