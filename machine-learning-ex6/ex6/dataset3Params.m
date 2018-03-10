function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Set the C's and sigma's
C_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
errors = zeros(size(C_set, 1), size(sigma_set, 1));
% Train, predict and compute the error for each C and sigma
for i=1:length(C_set),
	for j=1:length(sigma_set),
		c = C_set(i);
		s = sigma_set(j);
		% Train the SVM using parameters c and s
		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
		% Predict the values for the cross validation set
		predictions = svmPredict(model, Xval);
		% Compute the error
		errors(i,j) = mean(double(predictions ~= yval));
	end
end

% Choose the best C and sigma pair (lowest prediction error on val set)
min_err = min(min(errors));
[i,j] = find(errors==min_err);
% i,j might have more than one element, return the first
% and inform the user that there are multiple minimums,
% so he can choose manually to trade off bias and variance
if length(i)>1,
	fprintf('Found multiple minimum error (%f) values for C and sigma:\n', min_err);
	for k=1:length(i),
		fprintf('C: %f, sigma: %f\n', C(k), sigma(k));
	end
	i = i(1);
	j = j(1);
end
C = C_set(i);
sigma = sigma_set(j);

% =========================================================================

end
