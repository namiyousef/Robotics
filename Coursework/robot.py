from transformations import get_dh_params, dh_transformer, multiply_matrices, find_rotation_matrix
import sympy as sym
from parse_tools import define_sympy_vars, convert_to_sympy, remove_whitespace
from operator import mul
class manipulator:
    def __init__(self, base_loc = [0,0,0]):
        self.locs = [self.generate_point(base_loc)]
        self.joints = [] # are we actually interested in the joints?
        self.T_forwards = []
        self.T_FK = None
        self.joint_param_relations = []

        # TODO add a feature using __call__ similar to keras
    def add_joints(self, joints):
        for joint in joints:

            self.joints.append(joint)
            self.T_forwards.append(joint.T_forward)
            self.locs.append(
                self.T_forwards[-1] * self.locs[-1]
            )

    def calculate_forward_kinematics(self):
        self.T_FK = multiply_matrices(self.T_forwards)

    def calculate_inverse_kinematics(self, p, **rotations):

        p = [define_sympy_vars(comp) if isinstance(comp, str) else comp for comp in p]
        # TODO need to clean up what is and what isn't in the matrix form, not convinced? Either go all in or only evaluate at the end!
        p = sym.Matrix(p)
        R = find_rotation_matrix(**rotations)
        print(R)
        T = sym.Matrix(R)
        T = T.col_insert(3, p)
        p = sym.Matrix([[0, 0, 0, 1]])
        T = T.row_insert(3, p)
        return T

    def generate_point(self, vector):
        return sym.Matrix(vector + [1])


    def add_joint_param_relations(self, *relations):
        for relation in relations:
            LHS, RHS = remove_whitespace(relation).split('=')
            self.joint_param_relations.append(
                sym.Eq(
                convert_to_sympy(RHS),
                convert_to_sympy(LHS)
                )
            )

            # now need to create expressions, so need to pass through the routine
    def plot_motion(self, p_0, p_f, via = None):
        pass
        # very smart! How can I take advantage of the subs??


# TODO not sure if it needs to be a subclass!

class joint: #TODO add parent manipulator class?
    def __init__(self, dh_params):
        self.dh_params = get_dh_params(dh_params)
        for item in self.dh_params:
            print(type(item))
        self.T_forward = dh_transformer(self.dh_params)

class revolute(joint):
    """
    Note that T_forward is effectively showing m in coordinates of m-1!

    """
    # TODO took out **rotations, not necess?
    def __init__(self, dh_params):
        super().__init__(dh_params)


class prismatic(joint):
    def __init__(self, dh_params):
        super().__init__(dh_params)

if __name__ == '__main__':
    m = manipulator()
    r1 = revolute(['0', '0', '0', 'theta1'])
    p1 = prismatic(['sym.pi/2 - phi', '0', 'L1+d2', '0'])
    r2 = revolute(['-sym.pi/2 + phi', 'Le', 'L3', 'theta3'])
    m.add_joints([r1, p1, r2])
    m.calculate_forward_kinematics()
    m.calculate_inverse_kinematics(['x','y','z'], yaw = '-psi')
    m.add_joint_param_relations("d2 = -L1 + (z-L3)/sym.sin(phi)")
    print(m.joint_param_relations[-1].subs([('L1', 1), ('z', 1), ('L3', 0), ('phi', 'pi/2')]))
    # TODO need to think of a smarter way!


