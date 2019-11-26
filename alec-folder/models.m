%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wright State University                                Machine Learning %
% Written By: Alec Petrack                 Instructor: Dr. Tanvi Banerjee %
% Modified: 11/25/2019                                                    %
%                                                                         %
% Code Description: Fit heart disease data to predict whether patient has %
% heart disease or not.                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc

%% Load Pre-Procssed Data
X = xlsread('../data/processedHeart.xlsx'); % read data
numFolds = 5;
indices = crossvalind('Kfold',X(:,end),numFolds);

%% Model Heart Data
logConfMat = [105,33;21,144];
svmConfMat = [105,33;17,148];

tpSvm = logConfMat(1);
tnSvm = logConfMat(end);
fpSvm = logConfMat(2);
fnSvm = logConfMat(3);
[accuracySvm,sensitivitySvm,specificitySvm,precisionSvm,f1Svm] = ...
    calcConfStats(tpSvm,tnSvm,fpSvm,fnSvm);

tpSvm = svmConfMat(1);
tnSvm = svmConfMat(end);
fpSvm = svmConfMat(2);
fnSvm = svmConfMat(3);
[accuracyLog,sensitivityLog,specificityLog,precisionLog,f1Log] = ...
    calcConfStats(tpSvm,tnSvm,fpSvm,fnSvm);

function [accuracy,sensitivity,specificity,precision,f1] = calcConfStats(TP,TN,FP,FN)
totalSamples = TP+TN+FP+FN;
accuracy = (TP + TN) / totalSamples;
sensitivity = TP / (TP + FN);
specificity = TN / (FP + TN);
precision = TP / (TP + FP);
f1 = 2*TP/(2*TP + FP + FN);
end

