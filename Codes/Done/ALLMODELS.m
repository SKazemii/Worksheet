clc
clear all
close all
path(path,genpath('/Users/saeedkazemi/Desktop'))

test_ratio = 0.2;
persentage = '0.95';
mode = 'corr';%#corr dist
model_type = 'min';% #min median average
folder = [mode,'_',model_type,'_',persentage,'_0.2']
cd(['/Users/saeedkazemi/Documents/Python/Worksheet/results/',folder,'/NPY/'])

FRR_R = readNPY('FRR_R.npy');
FRR_L = readNPY('FRR_L.npy');
FRR_R(80,:) = [];
FRR_L(80,:) = [];
FRR_R_av = mean(FRR_R);
FRR_L_av = mean(FRR_L);


FAR_R = readNPY('FAR_R.npy');
FAR_L = readNPY('FAR_L.npy');
FAR_R(80,:) = [];
FAR_L(80,:) = [];
FAR_R_av = mean(FAR_R);
FAR_L_av = mean(FAR_L);
%%%%%%%%%%%%% min
folder = [mode,'_',model_type,'_',persentage,'_0.35']
cd(['/Users/saeedkazemi/Documents/Python/Worksheet/results/',folder,'/NPY/'])

FRR_R = readNPY('FRR_R.npy');
FRR_L = readNPY('FRR_L.npy');
FRR_R(80,:) = [];
FRR_L(80,:) = [];
FRR_R_mi = mean(FRR_R);
FRR_L_mi = mean(FRR_L);


FAR_R = readNPY('FAR_R.npy');
FAR_L = readNPY('FAR_L.npy');
FAR_R(80,:) = [];
FAR_L(80,:) = [];
FAR_R_mi = mean(FAR_R);
FAR_L_mi = mean(FAR_L);
%%%%%%%%%%%%% median
folder = [mode,'_',model_type,'_',persentage,'_0.5']
cd(['/Users/saeedkazemi/Documents/Python/Worksheet/results/',folder,'/NPY/'])

FRR_R = readNPY('FRR_R.npy');
FRR_L = readNPY('FRR_L.npy');
FRR_R(80,:) = [];
FRR_L(80,:) = [];
FRR_R_me = mean(FRR_R);
FRR_L_me = mean(FRR_L);


FAR_R = readNPY('FAR_R.npy');
FAR_L = readNPY('FAR_L.npy');
FAR_R(80,:) = [];
FAR_L(80,:) = [];
FAR_R_me = mean(FAR_R);
FAR_L_me = mean(FAR_L);

%%%%%%%%%%%%%

figure
ax1 = subplot(1,2,1); % top subplot
ax2 = subplot(1,2,2); % bottom subplot
set(gcf,'position',[500,400,1200,1200])

hold(ax1,'on')
plot(ax1,FAR_L_av,FRR_L_av,'r--', 'LineWidth',2)
plot(ax1,FAR_L_mi,FRR_L_mi,'b--', 'LineWidth',2)
plot(ax1,FAR_L_me,FRR_L_me,'g--', 'LineWidth',2)
plot(ax1,[0, 1], [0, 1],'k--', 'LineWidth',1)

legend(ax1,'20%', '35%', '50%')
pbaspect(ax1,[1 1 1])
xlabel(ax1,'FAR')
ylabel(ax1,'FRR')
title(ax1,'ROC curve, left side')


hold(ax2,'on')
plot(ax2,FAR_R_av,FRR_R_av,'r--', 'LineWidth',2)
plot(ax2,FAR_R_mi,FRR_R_mi,'b--', 'LineWidth',2)
plot(ax2,FAR_R_me,FRR_R_me,'g--', 'LineWidth',2)
plot(ax2,[0, 1], [0, 1],'k--', 'LineWidth',1)

legend(ax2,'20%', '35%', '50%')
pbaspect(ax2,[1 1 1])
xlabel(ax2,'FAR')
ylabel(ax2,'FRR')
title(ax2,'ROC curve, right side')

tightfig;

saveas(gcf,'/Users/saeedkazemi/Desktop/ALL.png')
%%

th = 0.001:0.001:1;
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

xticks(ax1,[0 .25 .50 .75 1])
xticklabels(ax1,{'0', '0.25', '0.5', '0.75','1'})
yticks(ax1,[0 .25 .50 .75 1])
yticklabels(ax1,{'0', '0.25', '0.5', '0.75','1'})
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
