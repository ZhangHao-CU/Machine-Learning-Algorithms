clear;
data = load('G:/MATLAB/ML/hw1data.mat');
X = data.X;
Y = data.Y;
%figure;
%imagesc(1-reshape(X(23,:),[28,28])');
%colormap gray;
training_X = X(1:6000,:);
testing_X = X(6001:10000,:);
training_Y = Y(1:6000,:);
testing_Y = Y(6001:10000,:);
%pre-handle the raw data only select 200 features
[~,index] = sort(var(training_X),'descend');
training_X = training_X(:,index(1,1:200));
testing_X = testing_X(:,index(1,1:200));
%z-score both data
training_X = zscore(training_X);
testing_X = zscore(testing_X);
%perform KNN, particularly K = 10
dist_t = zeros(1,size(training_X,1));
testing_results = zeros(size(testing_Y,1), 1);
for i = 1:size(testing_X,1)
    for j = 1:size(training_X,1)
%        dist_t(1,j) = norm(training_X(j,:)-testing_X(i,:));
%        dist_t(1,j) = norm(training_X(j,:)-testing_X(i,:),1);
        dist_t(1,j) = norm(training_X(j,:)-testing_X(i,:),inf);
    end
    [~,K_index] = sort(dist_t);
    testing_results(i,1) =mode(training_Y(K_index(1:10),:));
end
%calculate the accurancy
KNN = sum(testing_results == testing_Y)/size(testing_Y,1);