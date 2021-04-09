function HT = HTR3plan()
    syms theta1 theta3 L1 d2 L3 Le phi
    DH1 = [sym(0); sym(0); sym(0); theta1];
    T01 = LT(DH1);
    
    DH2 = [pi/2 - phi; sym(0); L1 + d2; sym(0)];
    T12 = LT(DH2);
    
    DH3 = [-pi/2 + phi, Le, L3, theta3];
    T23 = LT(DH3);
    
    HT(1,:,:) = T01;
    HT(2, :, :) = T01 * T12;
    HT(3,:,:) = T01 * T12 * T23;
end