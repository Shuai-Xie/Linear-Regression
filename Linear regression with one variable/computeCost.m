function J = computeCost(X, y, theta)

% ��ʼ��һЩֵ
m = length(y); % ѵ��������

E = X * theta - y; % ƫ�����
J = (E' * E) / (2 * m); % E' * E ƽ����

end