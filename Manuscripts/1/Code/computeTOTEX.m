function feature = computeTOTEX(COPTS)
    
    % feature = computeTOTEX(COPTS)
    % TOTEX : Total Excursions
    % COPTS [t,3] : RD, AP, ML COP time series
    % feature [3] : TOTEX_RD, TOTEX_AP, TOTEX_ML
    
    p = inputParser;
    addRequired(p,'COPTS',@(x)validateattributes(x,{'numeric'},{'2d'}));
    parse(p,COPTS);
    
    feature(:,1) = sum(sqrt(diff(p.Results.COPTS(:,2)).^2+diff(p.Results.COPTS(:,3)).^2));
    feature(:,2:3) = sum(abs(diff(p.Results.COPTS(:,2:3))));
    
end