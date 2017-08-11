function plotData(x, y) 
% PLOTDATA Plots the data points x and y into a new figure ����ֵΪ�յĺ���
%   1. plots the data points. 
%   2. gives the figure axes labels of population and profit.

figure; % ��һ���յ�ͼƬ����

% ���� matlab ��㺯�� �� figure �򿪵Ĵ��������
plot(x, y, 'rx', 'MarkerSize', 10); % rx ��ʮ�֣�10 ���� MarkerSize

% ���� x, y ���ǩ
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

end