import re
import sympy as sym

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

def is_number(string):
    try:
        return float(string)
    except:
        return False
def define_sympy_vars(var):
    """
    Gets a variable and checks if it exists in the globa namespace
    If yes, then a warning is triggered, and the namespace is not re-written!
    If no, then the namespace is re-written
    """
    gbl = globals()
    if not gbl.__contains__(var):
        print(f'Does not contain {var}, making it now')
        gbl[var] = sym.Symbol(var)
    else:
        print(f"""
        =====================================================================
        WARNING: variable {var} previously created. Will not be created again
        =====================================================================
        """)
    return sym.Symbol(var)

def convert_to_sympy(string):
    non_sympy = remove_sympy(string)
    # need to trim whitespace, NOT
    non_sympy = remove_punctuation(remove_numericals(non_sympy)).strip()
    non_sympy = non_sympy.split()
    for var in non_sympy:
        print(var)
        define_sympy_vars(var)

    return eval(string)

if __name__ =='__main__':
    print(remove_punctuation('-psi'))