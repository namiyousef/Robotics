%% Introduction
% Author: Yousef Nami
% Date: 24.10.2020
% Purpose: create a function HT that implements the homogeneous transform
% operator from frame {B} to frame {A}

A_p = HT([cosd(30), -sind(30), 0; sind(30), cosd(30), 0; 0, 0, 1], [1,2,3], [2,3,3]);
hold on
plot([0,1], [0,2]);
plot([1,3],[2,5]);
plot([0,3],[0,5]);
plot([0,A_p(1)],[0,A_p(2)]);
Rot = [cosd(30), -sind(30); sind(30), cosd(30)];
v = [2,3]';
New_vec = Rot * v;
plot([1,1+New_vec(1)],[2, 2+New_vec(2)]);
%% Function definition

function A_p = HT(AB_R, AB_0, B_p)
dof = length(AB_R);
AB_T = eye(dof + 1);
AB_T(1:dof, 1:dof) = AB_R;
AB_T(1:dof,dof + 1) = AB_0;
P = [];
for i = 1:dof
    P(i) = B_p(i);
end
P(dof + 1) = 1;
P = P';
A_p = AB_T * P;
disp(AB_T);
disp(P);
disp(A_p);
end

