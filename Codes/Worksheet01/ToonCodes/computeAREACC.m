function feature = computeAREACC(COPTS)
    
    % feature = computeAREACC(COPTS)
    % AREA-CC : 95% Confidence Circle Area
    % Only RD time series is used in the paper.
    % COPTS [t,3] : RD (AP, ML) COP time series
    % feature [1] : AREA-CC
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    MDIST = mean(abs(p.Results.COPTS(:,1)));
    RDIST = rms(p.Results.COPTS(:,1));
    z05 = 1.645; % z.05 = the z statistic at the 95% confidence level
    SRD = sqrt(RDIST.^2-MDIST.^2); % the standard deviation of the RD time series
    
    feature = pi*(MDIST+(z05.*SRD)).^2;
    
end