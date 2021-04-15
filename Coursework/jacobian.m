function j = jacobian()
syms theta1 d2 theta3 phi Le L1
gamma = theta1+theta3;
x = Le*cos(gamma) +cos(phi)*sin(theta1)*(L1+d2);
y = Le*sin(gamma) -cos(phi)*cos(theta1)*(L1+d2);

j(1,1) = -y; j(1,2) = cos(phi)*sin(theta1); j(1,3) = -Le*sin(gamma);
j(2,1) = x; j(2,2) = -cos(phi)*cos(theta1); j(2,3) = Le*cos(gamma);
j(3,1) = 0; j(3,2) = 0; j(3,3) = sin(phi);
j(4,1) = 0; j(4,2) = 0; j(4,3) = 0;
j(5,1) = 0; j(5,2) = 0; j(5,3) = 0;
j(6,1) = 1; j(6,2) = 0; j(6,3) = 1;

end