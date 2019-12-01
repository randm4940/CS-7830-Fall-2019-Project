%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wright State University                                Machine Learning %
% Written By: Alec Petrack                 Instructor: Dr. Tanvi Banerjee %
% Modified: 11/24/2019                                                    %
%                                                                         %
% Code Description: Feature analysis code for machine learning project.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%% Load Pre-Procssed Data
%x = xlsread('../data/processedHeart.xlsx'); % read data
X = xlsread('../data/heart.xlsx'); % read data
numFolds = 5;
indices = crossvalind('Kfold',X(:,end),numFolds);

% show distribution of labels
figure('name','Label Distribution')
histogram(X(:,end))
xticks([0,1])
xticklabels({'False','True'})
xlabel('Has Heart Disease?')
ylabel('Frequency')
title({'Distribution of Labels'})

%% Analyze Features in Dataset
% principle component analysis
featureNames = {'age','sex','cp','trestbps','chol','fbs','restecg',...
    'thalach','exang','oldpeak','slope','ca','thal'};
[coeff,score,latent,tsquared,explained,mu] = pca(X(:,1:end-1));
figure('name','Relationship Between Features')
imagesc(coeff)
colorbar
xticks(1:1:size(X,2))
yticks(1:1:size(X,2))
ytickangle(45)
yticklabels(featureNames)
xlabel('Principal Components')
ylabel('Features')
title('Principal Component Analysis')
% plot first two principal components
figure('name','Plot of First Two Principal Components')
scatter(X(:,5),X(:,4))
% pearson's R
[rho,pval] = corr(X(:,1:end-1));
figure('name','Pearson Correlation')
imagesc(rho)
colorbar
xticks(1:1:size(X,2))
xtickangle(45)
xticklabels(featureNames)
yticks(1:1:size(X,2))
ytickangle(45)
yticklabels(featureNames)
xlabel('Feature')
ylabel('Feature')
title('Pearson R Values')
% % look for relationships between data points
% figure('name','Scatter Plots')
% plotmatrix(X(:,1:end-1))
% title('Scatter Matrix')