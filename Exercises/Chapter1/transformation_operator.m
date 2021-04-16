%% Function definition
classdef transformation_operator < dynamicprops
    properties
        AB_R
        AB_0
        B_p
    end
    methods (Static)
        function quiver_plot_deets(q,type)
            if type == 'minor'
                q.Color = 'red';
                q.LineStyle = '--';
                q.ShowArrowHead = 'off';
            else if type == 'major'
                q.Color = 'black';
                q.MaxHeadSize = 0.1;
                end
            end
        end
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
        
        function obj = add_A_p(A_p)
            if nargin == 1
                obj.A_p = A_p;
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
            obj.addprop('A_p');
            obj.A_p = A_p;
        end
        
        function plot_vectors(obj)
            AB_R = obj.AB_R;
            AB_0 = obj.AB_0;
            B_p = obj.B_p;
            A_p = obj.A_p;
            hold on
            q = quiver(0,0,AB_0(1),AB_0(2),0);
            obj.quiver_plot_deets(q,'minor');
            q = quiver(AB_0(1),AB_0(2),B_p(1),B_p(2),0);
            obj.quiver_plot_deets(q,'minor');
            q = quiver(0,0,AB_0(1)+B_p(1),AB_0(2)+B_p(2),0);
            obj.quiver_plot_deets(q,'major');

            q = quiver(0,0,AB_0(1),AB_0(2),0);
            obj.quiver_plot_deets(q,'minor');
            q = quiver(AB_0(1),AB_0(2),A_p(1) - AB_0(1),A_p(2) - AB_0(2),0);
            obj.quiver_plot_deets(q,'minor');
            q = quiver(0,0,A_p(1),A_p(2),0);
            obj.quiver_plot_deets(q,'major');

            hold off

        end
        
        
    end
end