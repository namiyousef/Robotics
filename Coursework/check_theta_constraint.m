function theta1 = check_theta_constraint(x,y,theta1, gamma, d2, L1,Le, phi, L3)
error = 0.01;
theta3 = gamma - theta1;

HTs = HTR3plan;
HT = squeeze(HTs(3,:,:));
EF = [0;0;0;1];

HT = double(subs(HT,{'theta3','theta1', 'd2', 'L1', 'Le', 'phi', 'L3'}, ...
            {theta3, theta1, d2, L1, Le, phi, L3}));
        
p = HT*EF;
x_ = p(1); y_ = p(2);
while abs(x_ - x) > error || abs(y_ - y) > error
%     if theta1 >= 2*pi
%         theta1 = theta1 - 2*pi;
%     end
    theta1 = theta1 + 0.1;
theta3 = gamma - theta1;

HTs = HTR3plan;
HT = squeeze(HTs(3,:,:));
EF = [0;0;0;1];

HT = double(subs(HT,{'theta3','theta1', 'd2', 'L1', 'Le', 'phi', 'L3'}, ...
            {theta3, theta1, d2, L1, Le, phi, L3}));

x;
y;
p = HT*EF;
x_ = p(1);
y_ = p(2);

end

if theta1 >= 2*pi
theta1 = theta1 - 2*pi;
end
end