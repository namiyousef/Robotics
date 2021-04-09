%% trajectory planning 

clear all
close all

% define points
P0 = [1,1,2]; gamma0 = 0;
Pf = [3,4,5]; gammaf = pi/2;

% define time lapse
tf = 3;


% points of trajectory per segment (maximum points that you descritize)
Np = 10;

% links
L1 = sqrt(2);
phi = pi/4; 
Le = 1;
L3 = 1;
% inverse kinematics

% this is the joint parameters! theta1, d2, theta3, from what we calculated
% before

% define joint params at point 0
d2 = find_d2(P0(3), L1, L3, phi);
theta1 = find_theta1(P0(1), P0(2), L1, Le, phi, d2);
theta3 = find_theta3(gamma0, theta1);

J0 = [theta1, d2, theta3];

% define joint params at point f
d2 = find_d2(Pf(3), L1, L3, phi);
theta1 = find_theta1(Pf(1), Pf(2), L1, Le, phi, d2);
theta3 = find_theta3(gammaf, theta1);

Jf = [theta1, d2, theta3];

% build time model
t = linspace(0, tf, Np);

for i = 1:3
    a0 = J0(i);
    a1 = 0;
    a2 = 3/tf^2*(Jf(i) - J0(i));
    a3 = -2/tf^3*(Jf(i) - J0(i));
    JT(i, :) = a0 + a1*t + a2*t.^2 + a3*t.^3; 
end

% save it in degrees
JTd = JT * 180/pi;

HT = HTR3plan();

% position of the end effector is always 0 0 0 with respect to itself!
EF = [0;0;0;1];

for i = 1:Np
    for k = 1:3
        HTk = squeeze(HT(k,:,:));
        
        HTp = double(subs(HTk,...
            {'theta1', 'd2', 'theta3', 'L1', 'L3', 'Le', 'phi'}, ...
            {JT(1,i), JT(2,i), JT(3,i), L1, L3, Le, phi}...
            ));
        P(k, :, i) = HTp * EF;
    end
end

figure
hold on

plot(t, JTd(1,:), 'ro-', 'Linewidth', 3)
plot(t, JTd(2,:), 'bo-', 'Linewidth', 3)
plot(t, JTd(3,:), 'mo-', 'Linewidth', 3)
legend('T1', 'T2', 'T3')

figure
grid on
hold on
for i = 1: Np
    plot([P(1,1,i), P(2,1,i)], [P(1,2,i), P(2,2,i)], 'r')
    plot([P(2,1,i), P(3,1,i)], [P(2,2,i), P(3,2,i)], 'b')
    plot(P(3,1, i), P(3,2,i), 'mo', 'Linewidth', 5)
end


