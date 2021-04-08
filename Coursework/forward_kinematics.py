import plac
import numpy as np
import sympy as sym
import re
from sympy.parsing.sympy_parser import stringify_expr

def remove_whitespace(string):
    return re.sub(r"\s+", "", string)

def remove_sympy_expression(string):
    return re.sub(r'sym\..+?\(', '', string)
def remove_sympy(string):
    return re.sub(r'sym\..+?[\\\/\+\(\-\*\)]', '', string)

def remove_punctuation(string):
    return re.sub(r'[\\\/\+\(\-\*\)]', ' ', string)

def remove_numericals(string):
    return re.sub(r"(?<!\_)[0-9]", '', string)

def remove_numericals(string):
    return re.sub(r"(?<![\_a-zA-Z])[0-9]", '', string)


@plac.annotations(
    dh_params=('Numpy array of dh parameters', 'option', 'dh', np.array)
)
def main(dh_params = None):
    if not dh_params:
        # TODO connect this with the GUI
        dh_params = input('Please input all the DH parameters separated by commas and ; to separate rows:\n')
        rows = dh_params.split(';')

        dh_params = np.zeros([len(rows), 4])
        dh_params = [[0 for j in range(4)] for i in range(len(rows))]

        #dh_params = sym.Matrix(dh_params)
        for i, row in enumerate(rows):
            cols = row.split(',')
            cols = [remove_whitespace(item) for item in cols]
            for j, col in enumerate(cols):
                try:
                    col = float(col)
                    dh_params[i][j] = col
                    #dh_params[i,j] = col

                except:
                    # recognizes that the cell isn't numerical
                    #print(col)
                    non_sympy = remove_sympy(col)
                    # need to trim whitespace, NOT remove
                    non_sympy = remove_punctuation(remove_numericals(non_sympy)).strip()
                    non_sympy = non_sympy.split()
                    #lcl = locals() # TODO maybe need to define this as global to preventb
                    #lcl = globals()
                    # TODO python from confusing two 'locally' named same name variables?
                    for var in non_sympy:
                        #if not lcl.__contains__(var):
                        #    lcl[var] = sym.Symbol(var)
                        define_sympy_vars(var)
                    # at this point, all non-sympy vars have been defined, can now directly evaluate!
                    dh_params[i][j] = eval(col)
                    #dh_params[i, j] = eval(col)

        #print(dh_params)
        T_matrices = [dh_transformer(joint) for joint in dh_params]
        T_forward = 1
        for T in T_matrices:
            T_forward *= T

        #print(T_forward)
        T_inverse = create_transformation_matrix(find_rotation_matrix(yaw = 'theta3'), ['x', 'y', 'z'])
        eqns = [sym.simplify(exp1 - exp2) for exp1, exp2 in zip(T_forward, T_inverse) if sym.simplify(exp1 - exp2) != 0]
        eqn_str = [str(eqn) for eqn in eqns]
        counts = [sum([int(var in eqn) for var in ['theta1', 'd2', 'theta3']]) for eqn in eqn_str]
        # TODO not very smart!
        # TODO need to think of a smart method to do this!

        for eqn, count in zip(eqn_str, counts):
            if count == 1:
                for var in ['theta1', 'd2', 'theta3']:
                    print(sym.solve(eqn, var))
        # TODO need to save var that is being saved!
        # THEN use simultaneous! SHOULD not be TOO hard!
        #print(sym.solve(eqns, (theta1, d2, theta3)))
        # rule based solver? maybe you need simultaneous equations??
        # given joint variables, needs to solve the equations?
        # maybe can isolate joint variables to find the simplest ones? starting

        # TODO need to add an input for this too, currently has no input feature!



        # TODO do this again, this is not correct! We want sympy for the definitions of the transformation matrices!
        # TODO needs to recognize: symbols, numbers, and sympy sumbols (i.e. cos or pi)
    """ Given dh parameters for a robot, calculates the forward kinematics """

def dh_transformer(dh_params):
    """
    dh_params : list
        4 items, contains the link length, link offset, joint offset, etc..
        convention is that alpha comes first, then a, then d, then theta!
    """
    alpha, a, d, theta = dh_params
    return sym.Matrix([
        [sym.cos(theta), -sym.sin(theta), 0, a],
        [sym.sin(theta)*sym.cos(alpha), sym.cos(theta)*sym.cos(alpha), -sym.sin(alpha), -sym.sin(alpha) * d],
        [sym.sin(theta)*sym.sin(alpha), sym.cos(theta)*sym.sin(alpha), sym.cos(alpha), sym.cos(alpha) * d],
        [0, 0, 0, 1]
    ])

def find_rotation_matrix(**rotations):
    R = 1
    for rotation, angle in rotations.items():
        if isinstance(angle, str):
            angle = define_sympy_vars(angle)
        # TODO need to verify angle digit or not, then create local variable if necess
        R *= create_rot_matrix(rotation, angle)
    return R

def define_sympy_vars(var):
    """
    Gets a variable and checks if it exists in the globa namespace
    If yes, then a warning is triggered, and the namespace is not re-written!
    If no, then the namespace is re-written
    """
    gbl = globals()
    if not gbl.__contains__(var):
        gbl[var] = sym.Symbol(var)
    else:
        print(f"""
        =====================================================================
        WARNING: variable {var} previously created. Will not be created again
        =====================================================================
        """)
    return sym.Symbol(var)
    # needs to return the variables as namespaces?
def create_rot_matrix(rotation, angle):
    """ takes an angle as a sumpy symbol, OR integer! DOES NOT ACCEPT STR"""
    if rotation not in ['yaw', 'pitch', 'roll']:
        raise ValueError('You must have either yaw pitch or roll. Other rotatinos are not accepted')
    elif rotation == 'yaw':
        return sym.Matrix([
            [sym.cos(angle), sym.sin(angle), 0],
            [-sym.sin(angle), sym.cos(angle), 0],
            [0, 0, 1]
        ])
    elif rotation == 'pitch':
        return sym.Matrix([
            [sym.cos(angle), 0, -sym.sin(angle)],
            [0, 1, 0],
            [sym.sin(angle), 0, sym.cos(angle)]
        ])
    elif rotation == 'roll':
        return sym.Matrix([
            [1, 0, 0],
            [0, sym.cos(angle), sym.sin(angle)],
            [0, -sym.sin(angle), sym.cos(angle)]
        ])
def create_transformation_matrix(rotation_matrix, position_vector):
    # TODO needs to define variables for position vector!
    position_vector = [define_sympy_vars(p) if isinstance(p, str) else p for p in position_vector]
    # TODO need to clean up what is and what isn't in the matrix form, not convinced? Either go all in or only evaluate at the end!
    p = sym.Matrix(position_vector)
    T = sym.Matrix(rotation_matrix)
    T = T.col_insert(3, p)
    p = sym.Matrix([[0,0,0,1]])
    T = T.row_insert(3, p)
    return T

def find_inverse(forward_kinematics, T_matrix):
    pass

if __name__ == '__main__':
    #R = find_rotation_matrix(yaw = 'theta3')
    #print(create_transformation_matrix(R, [1,2,3]))
    plac.call(main)
    #sprint(R)

