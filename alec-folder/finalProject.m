%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wright State University                                Machine Learning %
% Written By: Alec Petrack                 Instructor: Dr. Tanvi Banerjee %
% Modified: 11/7/2019                                                     %
%                                                                         %
% Code Description: Final project code for machine learning.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%% Load Data
X = xlsread('../processedHeart.xlsx'); % read data

%% Perform PCA
[coeff,score,latent,tsquared,explained,mu] = pca(X);
figure('name','Relationship Between Features')
imagesc(coeff)
colorbar

%% Pearson's R


%% Split Data Into Training and Testing Sets
numFolds = 5;
indices = crossvalind('Kfold',X,numFolds);

%% Train Logistic Regression Model to Classify Patients w/ Heart Disease
for ii = 1:numFolds
test = (indices == ii); 
train = ~test;
[B,dev,stats] = mnrfit(zscore(X(train,:)),categorical(Y(train,:)));
pValue(:,ii) = stats.p;
yHat(:,ii) = sigmoid([ones(size(X,1),1),zscore(X)]',B);
end

%% Evaluate Performance of the Model
% p = stats.p;
% cp = classperf(species);

%% Algorithms/Equations

function hypothesis = sigmoid(X,theta)
hypothesis = round(1./(1+exp(-theta'*X)));
end

% extra trees classifier


% rest ecg and "slope" sort of the same (one is at rest, the other not
% rest)