from transformations import get_dh_params, dh_transformer, multiply_matrices, find_rotation_matrix
import sympy as sym
from parse_tools import define_sympy_vars, convert_to_sympy, remove_whitespace
from pprint import pprint
from plotting_tools import visualiser
import plac
# TODO add plotting of the individual components of q
# TODO add plotting of the acceleration and velcoity of the end effector
# TODO add plotting for qdot, qddot
# TODO add prismatic and revolute expressions for intermediate jacobians and do the repsective plots
# TODO make a plotting class, ensure that things are inherited correctly
# TODO add main function that does the correct things when given expressions and dh parameters as files
# TODO make default calculations symbolic, but add wrappers to use numpy calculations instead?
# TODO params contains parameters and their values

# TODO add code that makes it user friendly, i.e. user can specify x,y,z and angles!
# TODO need a method to convert 3x3 matrices to transform matrix
# also to convert vectors to 4x1 vectors!

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
        self.m_params = {
            m_label : eval(m_) if isinstance(m_, str) else m_ for m_label, m_ in m_params.items()}
        self.x_labels = x_labels
        self.locs = [self.generate_point(base_loc)]
        self.joints = []
        self.T_forwards = []
        self.T_FK = None
        self.joint_param_relations = {}
        self.joint_param_relations_from_code = {}


    # TODO add a feature using __call__ similar to keras
    def add_joints(self, joints):
        for joint in joints:
            self.joints.append(joint)
            self.T_forwards.append(joint.T_forward)
            # TODO consider removing location saving? You are caclulating this for the plot anyways
            self.locs.append(
                self.T_forwards[-1] * self.locs[-1]
            )

    def add_joint_param_relations(self, *relations):
        for relation in relations:
            LHS, RHS = remove_whitespace(relation).split('=')
            self.joint_param_relations[LHS] = sym.Eq(
                convert_to_sympy(RHS),
                convert_to_sympy(LHS)
            )

    def calculate_forward_kinematics(self):
        self.T_FK = multiply_matrices(self.T_forwards)

    def generate_point(self, vector):
        return sym.Matrix(vector + [1])

    def check_in_workspace(self, x):
        x_params = self.define_parameter_dict(x, self.x_labels)
        q_params = self.solve_for_q(x_params, self.m_params)

        # TODO need to make all of this generalisable, rn really bad
        q_constraints = {
            joint.q_param : joint.constraint for joint in self.joints[:-1]
        }
        for q_param, q_ in q_params.items():
            q_min, q_max = q_constraints[q_param]
            if q_ < q_min or q_ > q_max:
                raise ValueError('The point you have entered is outside the workspace.')

    def define_parameter_dict(self, x, x_labels):
        return {
            x_label: x_ for x_label, x_ in zip(x_labels, x)
        }

    def solve_for_q(self, x, m_params):
        q_params = {}
        for q_, q_rel in self.joint_param_relations.items():
            sol = sym.solve(q_rel.subs(x).subs(m_params).subs(q_params))[0]
            q_params[q_] = sol
        return q_params

    # TODO this stuff is incomplete
    # TODO need to add a numpy wrapper, what if we wanted to evaluate the stuff?
    def calculate_inverse_kinematics(self, p, **rotations):
        # TODO how can you check this against the xlabels?

        p = [define_sympy_vars(comp) if isinstance(comp, str) else comp for comp in p]
        p = sym.Matrix(p)
        R = find_rotation_matrix(**rotations)
        T = sym.Matrix(R)
        T = T.col_insert(3, p)
        p = sym.Matrix([[0, 0, 0, 1]])
        T = T.row_insert(3, p)
        return T

    def equate_fk_ik(self, ik):
        eqns = [sym.simplify(fk_item - ik_item) for fk_item, ik_item in zip(self.T_FK, ik) if sym.simplify(fk_item - ik_item) != 0]
        eqns = [eqn for eqn in eqns if -eqn not in eqns]
        eqns = list(set(eqns))
        q_labels = [joint.q_param for joint in self.joints[:-1]]

        labels = [[q_label for q_label in q_labels if q_label in str(eqn)] for eqn in eqns]
        counts, labels, index = zip(*sorted(zip([len(c) for c in labels], labels, range(len(labels)))))
        eqns = [eqns[i] for i in index]
        sols = [sym.solve(eqn, label[0])[0] for eqn, label in zip(eqns, labels) if len(label) == 1]

        print(sols)
        print(self.recursive_solver(q_labels, eqns))

        raise Exception
        # TODO this part struggles a lot, need to implement rules manually

    def recursive_solver(self, q_labels, eqns, sols = {}):
        print('start', q_labels)
        labels = [[q_label for q_label in q_labels if q_label in str(eqn)] for eqn in eqns]
        counts, labels, index = zip(*sorted(zip([len(c) for c in labels], labels, range(len(labels)))))
        eqns = [eqns[i] for i in index]


        if len(labels[0]) == 1:
            sols[labels[0][0]] = sym.solve(eqns[0], labels[0][0])[0]

            index = [labels.index([label]) for label in list(sols.keys())]
            q_labels = [q_label for q_label in q_labels if q_label not in list(sols.keys())]
            print(q_labels)
            for i in index:
                del eqns[i]
            return self.recursive_solver(q_labels, eqns, sols) if q_labels else sols
        else:
            # looks if any variables can be combined to form new vars
            # TODO unsure how to solve this part!
            print('TEST')
            pprint(eqns)
            pprint(sym.solve(eqns, tuple(q_labels)))


        #return self.recursive_solver(q_labels, eqns, sols) if q_labels else sols



        # first checks if there are any 1d things that can be found
        # then looks for simultanous eqns
        # then looks for custom solitions


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

@plac.annotations(
    path_to_config=('Path to configuration file', 'positional', None, str),
    x0=('initial position', 'positional', None, list),
    xf=('final position', 'positional', None, list),
    timedelta=('Time for movement', 'positional', None, int),
    n_points=('Number of points', 'positional',None, int)
)
def main(path_to_config, x0, xf, timedelta, n_points):
    import json
    print(path_to_config, x0, xf, timedelta, n_points)
    config = json.load(open(path_to_config, 'r'))
    m = manipulator(config['manipulator_params'])
    joints = []
    for joint_ in config['joints']:
        if joint_['type'] == 'r':
            joints.append(revolute(joint_['dh'], joint_['q'], constraint = joint_['constraint']))
        elif joint_['type'] =='p':
            joints.append(prismatic(joint_['dh'], joint_['q'], constraint = joint_['constraint']))
        else:
            joints.append(joint(joint_['dh']))



if __name__ == '__main__':
    plac.call(main)
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
    #m.equate_fk_ik(ik)
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


