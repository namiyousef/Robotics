%% Introduction
% Author: Yousef Nami
% Date: 25.10.2020
% Purpose: create a function that plots the positions of the vectors

% note that the function has been implemented within the
% transformation_operator file, this file is just for testing

AB_R = [cosd(30), -sind(30); sind(30), cosd(30)];
AB_0 = 2*rand(1,2);
B_p = 3*rand(1,2);

obj = transformation_operator(AB_R, AB_0, B_p); % instantiate class
A_p = obj.HT();
obj.plot_vectors()
