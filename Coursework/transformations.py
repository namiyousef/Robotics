from parse_tools import remove_whitespace, is_number, convert_to_sympy, define_sympy_vars
import sympy as sym

def get_dh_params(dh_params_list):
    # TODO add functionality to this, allowing it to recognize if Sympy vars are being given to it!
    dh_params_list = [remove_whitespace(param) for param in dh_params_list]
    dh_params_list = [num if ((num := is_number(param)) is not False) else convert_to_sympy(param) for param in dh_params_list]
    return dh_params_list

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

def multiply_matrices(matrix_list, M = 1):
    return multiply_matrices(matrix_list[1:], M *matrix_list[0] ) if matrix_list else M

def create_rot_matrix(rotation, angle):
    """ takes an angle as a sumpy symbol, OR integer! DOES NOT ACCEPT STR

    NOte:
        these are rotation matrices designed to shift a coordinate axis, NOT a point
        when using them for points, make sure you apply the transpose operator
        """
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


def find_rotation_matrix(**rotations):
    rot_matrices = [
        create_rot_matrix(rot, angle_var := convert_to_sympy(angle))\
            if isinstance(angle, str) else\
            create_rot_matrix(rot,angle) for rot, angle in rotations.items()
    ]
    # TODO need to add an option for the transpose, i.e. going from 1 -> 2 as opposed to 2 -> 1
    return multiply_matrices(rot_matrices)
