function T = LT(DH)
    a = DH(2); alpha = DH(1); d = DH(3); theta = DH(4);
    T = sym('a%d%d', [4,4]);
    
    T(1,1) = cos(theta); T(1,2) = -sin(theta); T(1,3) = 0; T(1,4) = a;
    T(2,1) = sin(theta)*cos(alpha); T(2,2) = cos(theta)*cos(alpha);
    T(2,3) = -sin(alpha); T(2,4) = -sin(alpha)*d;
    T(3,1) = sin(theta)*sin(alpha); T(3,2) = cos(theta)*sin(alpha);
    T(3,3) = cos(alpha); T(3,4) = cos(alpha) * d;
    T(4,1) = 0; T(4,2) = 0; T(4,3) = 0; T(4,4) = 1;
end