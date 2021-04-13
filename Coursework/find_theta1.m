function theta1 = find_theta1(x, y, L1, Le, phi, d2, gamma)
l = (L1+d2)*cos(phi);
%theta1 = atan2(Le*y - l*x, Le*x + l*y);
%theta1 = atan2((Le*y - l*x),(Le*x + l*y));
theta1 = asin((x - Le*cos(gamma))/l);

end