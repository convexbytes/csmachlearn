function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% % First approach: compute manually each layer
% % Input layer
% a_1 = [ones(m, 1) X];
% % Compute second layer
% a_2 = a_1 * Theta1'; 
% z_2 = [ones(m,1) sigmoid(a_2)];
% % Compute output layer
% a_3 = z_2 * Theta2';
% z_3 = sigmoid(a_3);
% % Compute predictions
% [prob, p] = max(z_3, [], 2);

% Second approach: Automatically for each layer
all_theta = {Theta1';Theta2'};
% Set the input for the initial layer
layer_input = [ones(size(X,1),1) X];
for i = 1:size(all_theta, 1),	
	% Compute activations for current layer
	theta_i = all_theta{i};
	a_i = layer_input * theta_i;
	z_i = sigmoid(a_i);
	% Set the input for the next iteration
	layer_input = [ones(size(z_i,1),1) z_i];
end
% The last layer_input is the final h
% Remove the ones-feature from h
h = layer_input(:,2:end);
% Remove the ones-feature from h
[prob,p] = max(h, [], 2);

% =========================================================================


end
