clc
clear all
close all
path(path,genpath('/Users/saeedkazemi/Desktop'))
cd '/Users/saeedkazemi/Documents/Python/Worksheet/results/corr_min_1.0_0.5/NPY/'
% jump(path)
FRR_R = readNPY('FRR_R.npy');
FRR_L = readNPY('FRR_L.npy');
FRR_R(80,:) = [];
FRR_L(80,:) = [];
FRR_R = mean(FRR_R);
FRR_L = mean(FRR_L);


FAR_R = readNPY('FAR_R.npy');
FAR_L = readNPY('FAR_L.npy');
FAR_R(80,:) = [];
FAR_L(80,:) = [];
FAR_R = mean(FAR_R);
FAR_L = mean(FAR_L);



EER_R = readNPY('EER_R.npy');
EER_L = readNPY('EER_L.npy');
EER_R(80,:) = [];
EER_L(80,:) = [];
table(1,1) = mean(EER_L(:,1));
table(1,2) = mean(EER_R(:,1));
table(2,1) = min(EER_L(:,1));
table(2,2) = min(EER_R(:,1));
table(3,1) = max(EER_L(:,1));
table(3,2) = max(EER_R(:,1));
table(4,1) = median(EER_L(:,1));
table(4,2) = median(EER_R(:,1));

ACC_L = readNPY('ACC_L.npy');
ACC_R = readNPY('ACC_R.npy');
ACC_L(80,:) = [];
ACC_R(80,:) = [];
table(1,3) = mean(ACC_L(:,3));
table(1,4) = mean(ACC_R(:,3));
table(2,3) = min(ACC_L(:,3));
table(2,4) = min(ACC_R(:,3));
table(3,3) = max(ACC_L(:,3));
table(3,4) = max(ACC_R(:,3));
table(4,3) = median(ACC_L(:,3));
table(4,4) = median(ACC_R(:,3));

table = round(table,2);

th = 0.5005:0.0005:1;
figure % new figure
ax1 = subplot(2,2,[1,2]); % top subplot
ax2 = subplot(2,2,3); % bottom subplot
ax3 = subplot(2,2,4); % bottom subplot
set(gcf,'position',[500,400,1200,1200])

% saveas(gcf,'ACC.png')


hold(ax1,'on')
plot(ax1,FAR_R,FRR_R,'r--', 'LineWidth',2)
plot(ax1,FAR_L,FRR_L,'b--', 'LineWidth',2)
plot(ax1,[0, 1], [0, 1],'k--', 'LineWidth',1)

% xticks(ax1,[0 .25 .50 .75 1])
% xticklabels(ax1,{'0', '0.25', '0.5', '0.75','1'})
% yticks(ax1,[0 .25 .50 .75 1])
% yticklabels(ax1,{'0', '0.25', '0.5', '0.75','1'})
legend(ax1,'right side', 'left side')
pbaspect(ax1,[1 1 1])
xlabel(ax1,'FAR')
ylabel(ax1,'FRR')
title(ax1,'ROC curve')


plot(ax2, th,FRR_R,'--', 'LineWidth',2, 'Color',[0 0.4470 0.7410])
hold(ax2,'on')
grid(ax2,'on')
pbaspect(ax2,[1 1 1])
plot(ax2, th,FAR_R,'k--', 'LineWidth',2, 'Color',[0.3010 0.7450 0.9330])
xlabel(ax2, 'Threshold')
abs_diffs = abs(FAR_R - FRR_R);    
min_index = find(abs_diffs==min(abs_diffs));
eer = mean([FAR_R(min_index), FRR_R(min_index)]);
str1 = ['EER = ',num2str(round(eer,2))];
title(ax2,['The accuracy graph, right side, ',str1])
legend(ax2, 'FAR','FRR')



plot(ax3, th,FRR_L,'k--', 'LineWidth',2, 'Color',[0 0.4470 0.7410])
hold(ax3,'on')
grid(ax3,'on')
pbaspect(ax3,[1 1 1])
plot(ax3, th,FAR_L,'k--', 'LineWidth',2, 'Color',[0.3010 0.7450 0.9330])
xlabel(ax3, 'Threshold')
abs_diffs = abs(FAR_L - FRR_L);    
min_index = find(abs_diffs==min(abs_diffs));
eer = mean([FAR_L(min_index), FRR_L(min_index)]);

str1 = ['EER = ',num2str(round(eer,2))];
title(ax3,['The accuracy graph, left side, ',str1])
legend(ax3, 'FAR','FRR')


tightfig;

saveas(gcf,'/Users/saeedkazemi/Desktop/ACC.png')

%%

distModel1 = readNPY('distModel1.npy');





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






clc; clear all;
FAR=[0.027,0.111,0.027,0.027,0,0,0,0.027,0,0]; 
FRR=[0,0,6,0,3,0,1.333333333,3,0,0]; 
plot(FAR)
figure, plot(FRR)
idx = find(FAR - FRR < eps, 1); %// Index of coordinate in array 
px = FAR(idx); py = FRR(idx); figure, plot(px,py,'or','MarkerSize',18);