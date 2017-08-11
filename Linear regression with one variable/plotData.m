function plotData(x, y) 
% PLOTDATA Plots the data points x and y into a new figure 返回值为空的函数
%   1. plots the data points. 
%   2. gives the figure axes labels of population and profit.

figure; % 打开一个空的图片窗口

% 调用 matlab 描点函数 在 figure 打开的窗口上描点
plot(x, y, 'rx', 'MarkerSize', 10); % rx 红十字，10 设置 MarkerSize

% 设置 x, y 轴标签
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

end