%% Introduction

function proj_vec = return_vector(A_R_B, output_vector)
if nargin < 2
    A_R_B = [1,2,3; 4,5,6; 7,8,9]
    disp(A_R_B)
    output_vector = 'Ai_B'    
end

if output_vector == 'Ai_B'
    proj_vec = A_R_B(:,1)
elseif output_vector == 'Aj_B'
    proj_vec = A_R_B(:,2)
    elseif output_vector == 'Ak_B'
    proj_vec = A_R_B(:,3)
    elseif output_vector == 'Bi_A'
    proj_vec = A_R_B(1,:)
    elseif output_vector == 'Bj_A'
    proj_vec = A_R_B(2,:)

elseif output_vector == 'Bk_A'
    proj_vec = A_R_B(3,:)
end
end