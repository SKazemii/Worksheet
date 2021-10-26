function feature = computeFDCE(COPTS)
    
    % feature = computeFDCE(COPTS)
    % FD-CE : Fractal Dimension based on the 95% Confidence Ellipse
    % Only AP, ML time series are used in the paper.
    % COPTS [t,3] : (RD) AP, ML COP time series
    % feature [1] : FD-CE
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    N = size(p.Results.COPTS,1);
    F05 = 3; % F.05[2,n-2]; the F statistic at a 95% confidence level for a bivariate distribution with n data points (when n>120)
    SAP = rms(p.Results.COPTS(:,2));
    SML = rms(p.Results.COPTS(:,3));
    SAPML = mean(p.Results.COPTS(:,2).*p.Results.COPTS(:,3));
    d = sqrt(8*F05*sqrt(SAP.^2.*SML.^2-SAPML.^2));
    TOTEX = computeTOTEX(p.Results.COPTS);
    
    feature = log(N)./log((N.*d)./TOTEX(1));
    
end