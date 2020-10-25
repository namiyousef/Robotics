%% Introduction
% Author: Yousef Nami
% Date: 24.10.2020
% Purpose: create a function HT that implements the homogeneous transform
% operator from frame {B} to frame {A}

A_p = HT([cosd(30), -sind(30), 0; sind(30), cosd(30), 0; 0, 0, 1], [1,2,3], [2,3,3]);
disp(A_p);
hold on
plot([0,1], [0,2])
plot([1,3],[2,5])
plot([0,3],[0,5])
plot([0,A_p(1)],[0,A_p(2)])
%% Function definition

function A_p = HT(AB_R, AB_0, B_p)
AB_T = eye(4);
AB_T(1:3, 1:3) = AB_R;
AB_T(4, 1:3) = AB_0;
P = [B_p(1), B_p(2), B_p(3), 1]';
A_p = linsolve(AB_T, P);
end

