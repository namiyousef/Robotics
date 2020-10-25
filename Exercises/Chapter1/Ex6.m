%% Introduction
% Author: Yousef Nami
% Date: 24.10.2020
% Purpose: create a function HT that implements the homogeneous transform
% operator from frame {B} to frame {A}

% the function has been created within the file transformation_operator you
% this file just tests it
obj = transformation_operator([cosd(30), -sind(30), 0; sind(30), cosd(30), 0; 0, 0, 1], [1,2,3], [2,3,3]);

% run function
A_p = obj.HT();

%% for testing purpose
hold on
plot([0,1], [0,2]);
plot([1,3],[2,5]);
plot([0,3],[0,5]);
plot([0,A_p(1)],[0,A_p(2)]);
Rot = [cosd(30), -sind(30); sind(30), cosd(30)];
v = [2,3]';
New_vec = Rot * v;
plot([1,1+New_vec(1)],[2, 2+New_vec(2)]);


