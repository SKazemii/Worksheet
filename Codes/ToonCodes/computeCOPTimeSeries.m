function COPTS = computeCOPTimeSeries(Footprint3D,varargin)
    
    % COPTS = computeCOPTimeSeries(Footprint3D,varargin)
    % Footprint3D : [x,y] image x t frames
    % COPTS : RD, AP, ML COP time series
    
    defaultType = 'cop'; % 'coa'
    defaultBinarize = 'simple'; % 'otsu', 'adaptive'
    defaultReference = 'mean'; % 'local'
    % A 4th-order zero phase Butterworth low-pass digital filter with a 5-Hz cut-off frequency
    defaultFilter_Order = 4; % n
    defaultFilter_CutOffFreq = 5; % fc
    defaultFilter_SampFreq = 100; % fs
    p = inputParser;
    addRequired(p,'Footprint3D',@(x)validateattributes(x,{'numeric'},{'3d'}));
    addParameter(p,'Type',defaultType);
    addParameter(p,'Binarize',defaultBinarize);
    addParameter(p,'Reference',defaultReference);
    addParameter(p,'Filter_Order',defaultFilter_Order);
    addParameter(p,'Filter_CutOffFreq',defaultFilter_CutOffFreq);
    addParameter(p,'Filter_SampFreq',defaultFilter_SampFreq);
    parse(p,Footprint3D,varargin{:});
    
    % Convert 3D grayscale image to 3D binary image
    % Low-pass Butterworth filter
    [b,a] = butter(p.Results.Filter_Order,p.Results.Filter_CutOffFreq/(p.Results.Filter_SampFreq/2));
    t_max = size(Footprint3D,3); % t frames
    COPTrajectory = zeros(t_max,2); % 2 for x-and y-axis
    
    for t = 1:t_max
        Footprint2D = p.Results.Footprint3D(:,:,t);
        Footprint2D(isnan(Footprint2D)) = 0; % Replace NaNs with Zeros
        BW2D = convertGS2BW(Footprint2D,'Binarize',p.Results.Binarize);

        props = regionprops(double(BW2D),Footprint2D,'Centroid','WeightedCentroid');
        if(isempty(props))
            COPTrajectory(t,:) = NaN; % Replace empty with NaN
        else
            if strcmp(p.Results.Type,'cop')||strcmp(p.Results.Type,'COP')
                % Center of Pressure: 
                % Center of the region based on location and intensity value
                COPTrajectory(t,:) = props.WeightedCentroid;
            elseif strcmp(p.Results.Type,'coa')||strcmp(p.Results.Type,'COA')
                % Center of Area: 
                % Center of mass of the region with a uniform intensity
                COPTrajectory(t,:) = props.Centroid;
            end
        end
    end
    
    % Filtering on COP time series
    COPTrajectory = filtfilt(b,a,COPTrajectory);
    
    % Normalization : ML = x-axis, AP = y-axis
    if strcmp(p.Results.Reference,'local')
        % Reference to the local foot-centered frame
        ML = COPTrajectory(:,1); 
        AP = COPTrajectory(:,2); 
    elseif strcmp(p.Results.Reference,'mean')
        % Reference to the mean COP
        ML = COPTrajectory(:,1)-mean(COPTrajectory(:,1)); 
        AP = COPTrajectory(:,2)-mean(COPTrajectory(:,2)); 
    end
    RD = sqrt(ML.^2+AP.^2);
    COPTS = [RD,AP,ML];
    
end
