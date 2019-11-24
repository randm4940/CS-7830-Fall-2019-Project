%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wright State University                                Machine Learning %
% Written By: Alec Petrack                 Instructor: Dr. Tanvi Banerjee %
% Modified: 11/24/2019                                                     %
%                                                                         %
% Code Description: Final project code for machine learning.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%% Load Pre-Procssed Data
X = xlsread('../data/processedHeart.xlsx'); % read data
numFolds = 5;
indices = crossvalind('Kfold',X,numFolds);

%% Analyze Features in Dataset
% principle component analysis
[coeff,score,latent,tsquared,explained,mu] = pca(X(:,1:end-1));
figure('name','Relationship Between Features')
imagesc(coeff)
colorbar
% pearson's R
[rho,pval] = corr(X(:,1:end-1));

