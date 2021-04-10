%% trajectory planning 

clear all
close all

% define points
P0 = [1,-1,2]; gamma0 = 0;
Pf = [0,-2,2]; gammaf = -pi/2;

% define time lapse
tf = 3;


% points of trajectory per segment (maximum points that you descritize)
Np = 10;

% links
L1 = 1;
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
disp(J0)
% define joint params at point f
d2 = find_d2(Pf(3), L1, L3, phi);
theta1 = find_theta1(Pf(1), Pf(2), L1, Le, phi, d2);
theta3 = find_theta3(gammaf, theta1);

Jf = [theta1, d2, theta3];
disp(Jf)
% build time model
t = linspace(0, tf, Np);

for i = 1:3
    a0 = J0(i);
    a1 = 0;
    a2 = 3/tf^2*(Jf(i) - J0(i));
    a3 = -2/tf^3*(Jf(i) - J0(i));
    JT(i, :) = a0 + a1*t + a2*t.^2 + a3*t.^3; 
    Jvel(i, :) = a1 + 2*a2*t + 3*a3*t.^2;
    Jacc(i, :) = 2*a2 + 6*a3*t;
end

% save it in degrees
JTd = JT;

HT = HTR3plan();
disp(HT)
% position of the end effector is always 0 0 0 with respect to itself!
EF = [0;0;0;1];

for i = 1:Np
    for k = 1:3
        HTk = squeeze(HT(k,:,:));
        
        HTp = double(subs(HTk,...
            {'theta1', 'd2', 'theta3', 'L1', 'L3', 'Le', 'phi'}, ...
            {JT(1,i), JT(2,i), JT(3,i), L1, L3, Le, phi}...
            ));
        
        HTvel = double(subs(HTk,...
            {'theta1', 'd2', 'theta3', 'L1', 'L3', 'Le', 'phi'}, ...
            {Jvel(1,i), Jvel(2,i), Jvel(3,i), L1, L3, Le, phi}...
            ));
        
        HTacc = double(subs(HTk,...
            {'theta1', 'd2', 'theta3', 'L1', 'L3', 'Le', 'phi'}, ...
            {Jacc(1,i), Jacc(2,i), Jacc(3,i), L1, L3, Le, phi}...
            )); 
        P(k, :, i) = HTp * EF;
        vel(k, :, i) = HTvel * EF;
        acc(k, :, i) = HTacc * EF;
    end
end

figure
hold on

plot(t, JTd(1,:), 'ro-', 'Linewidth', 3)
plot(t, JTd(2,:), 'bo-', 'Linewidth', 3)
plot(t, JTd(3,:), 'mo-', 'Linewidth', 3)
legend('$\theta_1$', '$d_2$', '$\theta_3$', 'Interpreter', 'latex')

figure
grid on
hold on
for i = 1: Np
    plot([P(1,1,i), P(2,1,i)], [P(1,2,i), P(2,2,i)], 'r')
    plot([P(2,1,i), P(3,1,i)], [P(2,2,i), P(3,2,i)], 'b')
    plot(P(3,1, i), P(3,2,i), 'mo', 'Linewidth', 5)
end

figure
grid on
hold on
for i = 1: Np
    plot([vel(1,1,i), vel(2,1,i)], [vel(1,2,i), vel(2,2,i)], 'r')
    plot([vel(2,1,i), vel(3,1,i)], [vel(2,2,i), vel(3,2,i)], 'b')
    plot(vel(3,1, i), vel(3,2,i), 'mo', 'Linewidth', 5)
end

figure
grid on
hold on
for i = 1: Np
    plot([acc(1,1,i), acc(2,1,i)], [acc(1,2,i), acc(2,2,i)], 'r')
    plot([acc(2,1,i), acc(3,1,i)], [acc(2,2,i), acc(3,2,i)], 'b')
    plot(acc(3,1, i), acc(3,2,i), 'mo', 'Linewidth', 5)
end


