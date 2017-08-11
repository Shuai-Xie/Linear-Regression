function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

E = X * theta - y; % Æ«²î¾ØÕó(m*1)
J = (E' * E) / (2 * m); % (E'*E) ·½²îºÍ

end
