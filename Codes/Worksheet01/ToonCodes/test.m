ACC_L = readNPY('ACC_L.npy');

a = [round(ACC_L(1:79,3),2);round(ACC_L(81:end,3),2)];
mean(a)
min(a)
max(a)

ACC_R = readNPY('ACC_R.npy');

a = [round(ACC_R(1:79,3),2);round(ACC_R(81:end,3),2)];
mean(a)
min(a)
max(a)



distModel2 = readNPY('distModel2.npy');





%COPTS_P = readNPY('COPTS.npy')';
close all
subject = 1
left = 3


dist = distance_Genuine{subject, left};
Anan = dist;
Anan(dist == 0) = NaN;
Model_client = median(dist, 2,'omitnan');

dist = distance_Imposter{subject, left};
Model_imposter = median(dist, 2);

% dist = distance_Genuine{subject, left};
% Model_client = sum(dist, 2)/(size(dist,2)-1);
% 
% dist = distance_Imposter{subject, left};
% Model_imposter = mean(dist, 2);


% dist = distance_Genuine{subject, left};
% Anan = dist;
% Anan(dist == 0) = NaN;
% Model_client = min(Anan, [], 2);
% 
% dist = distance_Imposter{subject, left};
% Model_imposter = min(dist, [], 2);


k=1;
for i= 0:0.04:2
    E1 = zeros(size(Model_client));
    E1(Model_client > i) = 1; 
    FRR(k) = sum(E1)/size(Model_client,1);
    
    E2 = zeros(size(Model_imposter));
    E2(Model_imposter < i) = 1; 
    FAR(k) = sum(E2)/size(Model_imposter,1);
    
    k = k+1;
    
end
figure()
plot(0:0.04:2,FRR,0:0.04:2,FRR, 'bo')
hold on
plot(0:0.04:2,FAR,0:0.04:2,FAR, 'r+')
saveas(gcf,'ACC.png')

figure()
plot(FAR,FRR)
saveas(gcf,'ROC.png')

