from transformations import get_dh_params, dh_transformer, multiply_matrices, find_rotation_matrix
import sympy as sym
from parse_tools import define_sympy_vars, convert_to_sympy, remove_whitespace
import math
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
# TODO add code that makes it user friendly, i.e. user can specify x,y,z and angles!
class manipulator:
    def __init__(self, base_loc = [0,0,0]):
        self.locs = [self.generate_point(base_loc)]
        self.joints = [] # are we actually interested in the joints?
        self.T_forwards = []
        self.T_FK = None
        self.joint_param_relations = {}

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

    def cubic_interpolator(self, q0, qf, t, td):
        a0 = q0
        a1 = 0
        a2 = (3 / (td ** 2)) * (qf - q0)
        a3 = -(2 / (td ** 3)) * (qf - q0)
        qi = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
        # TODO qdoti and qddoti are incorrecti
        qdoti = a1 + 2 * a2 * t + 3 * a3 * t ** 2
        qddoti = 2 * a2 + 6 * a3 * t
        return qi, qdoti, qddoti


    def equate_fk_ik(self, ik):
        eqns = [sym.simplify(fk_item - ik_item) for fk_item, ik_item in zip(self.T_FK, ik) if sym.simplify(fk_item - ik_item) != 0]
        pprint(sym.solve([eqns[0], eqns[2]], ('theta1', 'theta3')))
        # TODO this part struggles a lot, need to implement rules manually

    def generate_point(self, vector):
        return sym.Matrix(vector + [1])


    def add_joint_param_relations(self, *relations):
        for relation in relations:
            LHS, RHS = remove_whitespace(relation).split('=')
            self.joint_param_relations[LHS] = sym.Eq(
                                                convert_to_sympy(RHS),
                                                convert_to_sympy(LHS)
                                            )


            # now need to create expressions, so need to pass through the routine
    def plot_motion(self, x_0, x_f, timedelta, n_points, manipulator_params,via = None, parametrizer = 'cubic'):
        """ x are cartesian parameters, they contain angles too!"""
        if len(x_0) !=6 or len(x_f) !=6:
            raise TypeError('Your Cartesian parameters must be [x,y,z, phi, theta, psi]')

        # TODO add part for via point
        X = [self.define_cartesian_params_dict(x) for x in [x_0, x_f]]
        Q = []
        for x in X:
            Q.append(self.define_joint_params_dict(x, manipulator_params))

        t = np.linspace(0, timedelta, n_points)

        Q_labels = list(Q[-1].keys())
        print(Q_labels)
        Q_vals = [np.asarray(list(Q_val.values())).reshape(-1,1) for Q_val in Q]


        if parametrizer == 'cubic':
            qi, qdoti, qddoti = self.cubic_interpolator(*Q_vals, t, timedelta)


        axis_loc = sym.Matrix([0,0,0,1])

        axis_locs = []
        all_points = np.zeros(
            [ 4, len(self.T_forwards)]
        )
        fig, ax = plt.subplots()

        for i in range(n_points):
            for j in range(len(self.T_forwards)):
                q_temp = {label : val for label, val in zip(Q_labels, qi[:,i])}
                T = self.T_forwards[j].subs(q_temp).subs(manipulator_params)
                # TODO need to improve this portion of the code, make more consistent and nice?
                P = T * axis_loc
                axis_locs.append(
                    tuple(p for p in P[:-1])
                )
                plt.plot(*axis_locs[-1][:-1], 'bo')

                all_points[:,j] = np.asarray(P).reshape(-1)


        # TODO fix this, currently not looking very great!
        plt.show()
            # TODO find the joint parameters
            # parametrize them using the relevant function
            # save that
        # needs to give orientation, how do you generalise this?
        pass
        # very smart! How can I take advantage of the subs??

    def check_in_workspace(self, point):
        pass

    def define_cartesian_params_dict(self, x):
        return {
            name : val for name, val in zip(
                # TODO replace this when these values are actually input by the user
                ['x','y','z','phi', 'theta', 'psi'], x
            )
        }
    def define_joint_params_dict(self, x, manipulator_params):
        joint_params = {}
        for joint_param, joint_relation in self.joint_param_relations.items():
            sol = sym.solve(joint_relation.subs(x).subs(manipulator_params).subs(joint_params))[0]
            joint_params[joint_param] = sol
        return joint_params





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
    p1 = prismatic(['sym.pi/2 - alpha', '0', 'L1+d2', '0'])
    r2 = revolute(['-sym.pi/2 + alpha', '0', 'L3', 'theta3'])
    end_effector = joint(['0', 'Le', '0', '0'])
    m.add_joints([r1, p1, r2, end_effector])
    m.calculate_forward_kinematics()
    ik = m.calculate_inverse_kinematics(['x','y','z'], yaw = '-psi')
    m.add_joint_param_relations(
        "d2 = -L1 + (z-L3)/sym.sin(alpha)",
        "theta1 = sym.asin((x - Le*sym.cos(psi))/((L1+d2)*sym.cos(alpha)))",
        "theta3 = psi - theta1"
    )
    pprint(a:= m.joint_param_relations)
    """pprint(sym.solve(a[1].subs({
        'gamma':0,
        'Le':1,
        'x':1,
        'd2':math.sqrt(2) -1,
        'L1':1,
        'phi':sym.pi/4
    })))"""

    m.plot_motion([1,-1,2,0,0,0], [-2,0,2, 0, 0, -sym.pi], 3, 10,
                  {
                      'Le': 1,
                      'L1': 1,
                      'alpha': sym.pi / 4,
                      'L3':1
                  }
                  )
    #print(m.joint_param_relations[-1].subs([('L1', 1), ('z', 1), ('L3', 0), ('phi', 'pi/2')]))
    # TODO need to think of a smarter way!


