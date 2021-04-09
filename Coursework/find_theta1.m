function theta1 = find_theta1(x, y, L1, Le, phi, d2)
lambda = (L1+d2)*cos(phi);
theta1 = atan2(Le*y - lambda*x, Le*x + lambda*x);
end