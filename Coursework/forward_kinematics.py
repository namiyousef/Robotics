import plac
import numpy as np
import sympy as sym
import re

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
        dh_params = sym.Matrix(dh_params)
        for i, row in enumerate(rows):
            cols = row.split(',')
            cols = [remove_whitespace(item) for item in cols]
            for j, col in enumerate(cols):
                try:
                    col = float(col)
                    dh_params[i,j] = col

                except:
                    # recognizes that the cell isn't numerical
                    print(col)
                    non_sympy = remove_sympy(col)
                    # need to trim whitespace, NOT remove
                    non_sympy = remove_punctuation(remove_numericals(non_sympy)).strip()
                    non_sympy = non_sympy.split()
                    lcl = locals()
                    for var in non_sympy:
                        if not lcl.__contains__(var):
                            lcl[var] = sym.Symbol(var)

                    # at this point, all non-sympy vars have been defined, can now directly evaluate!
                    dh_params[i, j] = eval(col)

            # TODO do this again, this is not correct! We want sympy for the definitions of the transformation matrices!
        # TODO needs to recognize: symbols, numbers, and sympy sumbols (i.e. cos or pi)
    """ Given dh parameters for a robot, calculates the forward kinematics """

def dh_transformer():
    pass
if __name__ == '__main__':
    plac.call(main)
