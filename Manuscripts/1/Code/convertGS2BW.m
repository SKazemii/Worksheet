function BW = convertGS2BW(GS,varargin)
    
    % BW = convertGS2BW(GS,varargin)
    % GS : 2-D or 3-D grayscale image
    % BW : 2-D or 3-D binary image
    
    defaultBinarize = 'simple'; % 'otsu', 'adaptive'
    p = inputParser;
    addRequired(p,'GS',@(x)validateattributes(x,{'numeric'},{'3d'}));
    addParameter(p,'Binarize',defaultBinarize);
    parse(p,GS,varargin{:});
    
    if strcmp(p.Results.Binarize,'simple')
        BW = imbinarize(p.Results.GS,eps);
    elseif strcmp(p.Results.Binarize,'otsu')
        BW = imbinarize(p.Results.GS,'global');
    elseif strcmp(p.Results.Binarize,'adaptive')
        BW = imbinarize(p.Results.GS,'adaptive');
    end
    
end

