clear;
clc;
data = load('G:/MATLAB/ML/hw1data.mat');
X = data.X;
Y = data.Y;
%figure;
%imagesc(1-reshape(X(23,:),[28,28])');
%colormap gray;
training_X = X(1:4000,:);
testing_X = X(4001:10000,:);
training_Y = Y(1:4000,:);
testing_Y = Y(4001:10000,:);
%PCA
%[~,training_X,~] = pca(training_X);
%[~,testing,~] = pca(testing_X);
%pre-handle the raw data only select 32 features
[~,index] = sort(var(training_X),'descend');
training_X = training_X(:,index(1,1:50));
testing_X = testing_X(:,index(1,1:50));
%training_X = training_X(:,1:32);
%testing_X = testing_X(:,1:32);
%z-score both data
training_X = zscore(training_X);
testing_X = zscore(testing_X);
%specify the depth of decision tree
%B = sort(unique(training_X(:,4)));
%t = fitctree(training_X,training_Y).;
K = 12;
t.nodeNum = 1;
t.splitTimes = 0;
myTree = makeDecisionTree_2(training_X,training_Y,K,t);
accurancy_train = computeAccurancy(myTree, training_X, training_Y);
accurancy_test = computeAccurancy(myTree, testing_X, testing_Y);