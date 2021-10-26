function COPTS = computeCOPTimeSeries1(Footprint3D)
    

    
    t_max = size(Footprint3D,3); % t frames
    COPTrajectory = zeros(t_max,2); % 2 for x-and y-axis
    for t = 1:t_max
        Footprint2D = Footprint3D(:,:,t);
        BW2D = imbinarize(Footprint3D(:,:,t),eps); 
        Footprint2D(isnan(Footprint2D)) = 0; % Replace NaNs with Zeros
        props = regionprops(double(BW2D),Footprint2D,'Centroid','WeightedCentroid');
        
        COPTrajectory(t,:) = props.WeightedCentroid;
    end

    ML = COPTrajectory(:,1)-mean(COPTrajectory(:,1)); 
    AP = COPTrajectory(:,2)-mean(COPTrajectory(:,2)); 
    
    RD = sqrt(ML.^2+AP.^2);
    COPTS = [RD,AP,ML];
    
end
