%% Function definition
classdef transformation_operator
    properties
        AB_R
        AB_0
        B_p
    end
    methods
        % constructor method
        function obj = transformation_operator(AB_R, AB_0, B_p)
            if nargin == 3 % this is required for some reason...
                obj.AB_R = AB_R;
                obj.AB_0 = AB_0;
                obj.B_p = B_p;
            else
                error('Please input AB_R, AB_0 and B_p');
            end
        end
        
        function A_p = HT(obj)
            AB_R = obj.AB_R;
            AB_0 = obj.AB_0;
            B_p = obj.B_p;
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
        
    end
end