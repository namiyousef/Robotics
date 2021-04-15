from transformations import get_dh_params, dh_transformer, multiply_matrices, find_rotation_matrix
import sympy as sym
from parse_tools import define_sympy_vars, convert_to_sympy, remove_whitespace
import math
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotting_tools import visualiser
# TODO add plotting of the individual components of q
# TODO add plotting of the acceleration and velcoity of the end effector
# TODO add plotting for qdot, qddot
# TODO add prismatic and revolute expressions for intermediate jacobians and do the repsective plots
# TODO make a plotting class, ensure that things are inherited correctly
# TODO add main function that does the correct things when given expressions and dh parameters as files
# TODO make default calculations symbolic, but add wrappers to use numpy calculations instead?
# TODO params contains parameters and their values

# TODO add code that makes it user friendly, i.e. user can specify x,y,z and angles!
class manipulator(visualiser):
    """
    Variable names:
    You have 3 diff types of vars: joint params (q), cartesian values (x), manipulator params (m)
    x : [x,y,z,phi,theta,psi]
        These refer to Cartesian values required to define a points with respect to the base
        Note that the labels can be specified when instantiating the manipulator

    q : [theta1, d2, theta3]
        These refer to joint parameters, of course they change depending on the manipulator used

    m : [L1, L2, Le, alpha]
        These refer to the parameters that define the dimensions of the manipulator

    In general, capital letters of these refer to lists of variables
    So X would have x1, x2...

    {}_labels define the labels that the parameters take
    {}_params refer to dictionaries that contain labels of the varialbes, as well as their values
    """
    def __init__(self,
                 m_params,
                 base_loc = [0,0,0],
                 x_labels = ['x', 'y', 'z', 'phi', 'theta', 'psi'],
                 ):


        # TODO need a method to convert 3x3 matrices to transform matrix
        # also to convert vectors to 4x1 vectors!

        self.m_params = m_params
        self.x_labels = x_labels

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

    def linear_parabolic(self, ):
        pass


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



    def check_in_workspace(self, x):
        x_params = self.define_cartesian_params_dict(x)
        q_params = self.solve_for_q(x_params, self.m_params)

        # TODO need to make all of this generalisable, rn really bad
        q_constraints = {
            joint.q_param : joint.constraint for joint in self.joints[:-1]
        }
        for q_param, q_ in q_params.items():
            q_min, q_max = q_constraints[q_param]
            if q_ < q_min or q_ > q_max:
                raise ValueError('The point you have entered is outside the workspace.')





    def generate_workspace(self, constraints):
        # TODO generate a meshgrid from constraints
        # TODO use inverse kinematics to generate the workspace
        pass

    def define_parameter_dict(self, x, x_labels):
        return {
            x_label: x_ for x_label, x_ in zip(x_labels, x)
        }

    def define_cartesian_params_dict(self, x):
        # TODO need to change this to make syntax consistnet
        return {
            name : val for name, val in zip(
                # TODO replace this when these values are actually input by the user
                self.x_labels, x
            )
        }

    # TODO should add a general method that defines dicts

    def solve_for_q(self, x, m_params):
        q_params = {}
        for q_, q_rel in self.joint_param_relations.items():
            sol = sym.solve(q_rel.subs(x).subs(m_params).subs(q_params))[0]
            q_params[q_] = sol
        return q_params


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
        self.T_forward = dh_transformer(self.dh_params)

class revolute(joint):
    # TODO need to add physical constraints to the angles that these can take!
    # instantiated by defualt, but user also given an option
    """
    Note that T_forward is effectively showing m in coordinates of m-1!

    """
    # TODO took out **rotations, not necess?
    def __init__(self, dh_params, q_param, constraint = [0, 2*sym.pi]):
        super().__init__(dh_params)
        # TODO need to change the name of this to q_label
        self.q_param = q_param
        self.constraint = constraint


class prismatic(joint):
    def __init__(self, dh_params, q_param, constraint = [0, 1]):
        super().__init__(dh_params)
        self.q_param = q_param
        self.constraint = constraint











class visualiser(manipulator):

    def plot_motion(self, x0, xf, timedelta, n_points, manipulator_params, via = [], parametrizer = 'cubic'):
        """ x are cartesian parameters, they contain angles too!"""
        # TODO add code for via points
        X = [x0, via, xf] if via else [x0, xf]
        for x in X:
            if len(x) != 6:
                raise TypeError(f'Your Cartesian parameters must be {self.x_labels}')

        X_params = [self.define_parameter_dict(x, self.x_labels) for x in X]

        Q_params = [self.solve_for_q(x, self.m_params) for x in X_params]
        Q_labels = list(Q_params[-1].keys())

        t = np.linspace(0, timedelta, n_points)

        Q = [np.asarray(list(Q_param.values())).reshape(-1,1) for Q_param in Q_params]

        if parametrizer == 'cubic':
            qi, qdoti, qddoti = self.cubic_interpolator(*Q, t, timedelta)

        # TODO what does this location refer to?
        axis_loc = sym.Matrix([0,0,0,1])

        axis_locs = []
        all_points = np.zeros(
            [4, len(self.T_forwards)]
        )
        fig, ax = plt.subplots()

        """        fig = plt.figure()
        ax = fig.gca(projection='3d')"""

        for i in range(n_points):
            T = 1
            for j, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
                q_temp = {label : val for label, val in zip(Q_labels, qi[:,i])}
                T *= self.T_forwards[j].subs(q_temp).subs(manipulator_params)
                # TODO need to improve this portion of the code, make more consistent and nice?
                P = T * axis_loc
                axis_locs.append(
                    tuple(p for p in P[:-1])
                )
                points = np.asarray(axis_locs[j-1:]).T
                ax.plot(*axis_locs[-1][:-1], '.', c = color)

                all_points[:,j] = np.asarray(P).reshape(-1)
                # plot axes
                print(all_points[:-2, j-1:j+1])
                ax.plot(*all_points[:-2, j-1:j+1], 'black', linewidth = 0.2)



        # TODO fix this, currently not looking very great!
        plt.show()
            # TODO find the joint parameters
            # parametrize them using the relevant function
            # save that
        # needs to give orientation, how do you generalise this?
        pass
        # very smart! How can I take advantage of the subs??

if __name__ == '__main__':
    m = manipulator({
                      'Le': 1,
                      'L1': 1,
                      'alpha': sym.pi / 4,
                      'L3':1
                  })

    r1 = revolute(['0', '0', '0', 'theta1'], 'theta1', constraint = [0, sym.pi/2])
    p1 = prismatic(['sym.pi/2 - alpha', '0', 'L1+d2', '0'], 'd2')
    r2 = revolute(['-sym.pi/2 + alpha', '0', 'L3', 'theta3'], 'theta3', constraint = [-sym.pi/2, sym.pi/2])
    end_effector = joint(['0', 'Le', '0', '0'])
    m.add_joints([r1, p1, r2, end_effector])
    m.calculate_forward_kinematics()
    ik = m.calculate_inverse_kinematics(['x','y','z'], yaw = '-psi')
    m.add_joint_param_relations(
        "d2 = -L1 + (z-L3)/sym.sin(alpha)",
        "theta1 = sym.asin((x - Le*sym.cos(psi))/((L1+d2)*sym.cos(alpha)))",
        "theta3 = psi - theta1"
    )

    m.plot_motion([1,-1,2,0,0,0], [-2,0,2, 0, 0, -sym.pi], 3, 10,
                  {
                      'Le': 1,
                      'L1': 1,
                      'alpha': sym.pi / 4,
                      'L3':1
                  }
                  )

    print(m.check_in_workspace([1,-1,2,0,0,0]))

    m.plot_workspace('points', 10)
    # TODO if I enter an angle that doesn't exist, it will not ask me to check that!
    # TODO maybe that is good?
    # TODO need to think of a smarter way!


