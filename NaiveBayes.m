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
%pre-handle the raw data
[~,index] = sort(var(training_X),'descend');
training_X = training_X(:,index(1,1:200));
testing_X = testing_X(:,index(1,1:200));
%z-score both data
training_X = zscore(training_X);
testing_X = zscore(testing_X);
%calculate the class priors
class_p = zeros(1,10);
for i = 1:10
    class_p(1,i) = size(find(training_Y == i-1),1)/size(training_Y,1);
end
%training_Y = mat2cell(training_Y,divide);
%Mdl = fitcnb(training_X, training_Y,'Prior', class_p);
%define the class conditionals, using multivariate gaussian MLE
Gau_mu = zeros(10,size(training_X,2));
Gau_sigma = cell(1,10);
for i = 1:10
    Gau_mu(i,:) = mean(training_X(training_Y == i-1,:));
    Gau_sigma{1,i} = cov(training_X(training_Y == i-1,:));
end
testing_results = zeros(size(testing_Y,1), 1);
for i = 1:size(testing_X,1)
    each_p = zeros(1,10);
    for j = 1:10
        each_p(1,j) = class_p(1,j)*mvnpdf(testing_X(i,:),Gau_mu(j,:),Gau_sigma{1,j});
    end
    [~,I] = max(each_p);
    testing_results(i,1) = I-1;
end
%calculate the accurancy
naive_bayes = sum(testing_results == testing_Y)/size(testing_Y,1);