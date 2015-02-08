function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% Compute every h(xi) - y(i) --> return vector diff of length m
% Compute every h(xi) --> vector H lenght m
H = X * theta;
Diff = H - y;

% Compute the square of diff
Diff = Diff.^2;

% Sum the vector
SDiff = sum(Diff);

% Divide by 2m
J = SDiff / (2*m);


% =========================================================================

end
