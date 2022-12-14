from dsl.function import Function
from dsl.types import INT, BOOL, LIST, FunctionType

# first-order functions
# HEAD = Function('HEAD', lambda xs: xs[0] if xs else None, LIST, INT)
# TAIL = Function('TAIL', lambda xs: xs[-1] if xs else None, LIST, INT)
# MINIMUM = Function('MINIMUM', lambda xs: min(xs) if xs else None, LIST, INT)
# MAXIMUM = Function('MAXIMUM', lambda xs: max(xs) if xs else None, LIST, INT)
# REVERSE = Function('REVERSE', lambda xs: xs[::-1], LIST, LIST)
# SORT = Function('SORT', sorted, LIST, LIST)
# SUM = Function('SUM', sum, LIST, INT)
#
# TAKE = Function('TAKE', lambda n, xs: xs[:n], (INT, LIST), LIST)
# DROP = Function('DROP', lambda n, xs: xs[n:], (INT, LIST), LIST)
# ACCESS = Function('ACCESS', lambda n, xs: xs[n] if n >= 0 and len(xs) > n else None, (INT, LIST), INT)

Empty = Function('Empty',sorted, LIST, LIST)
inner_join = Function('inner_join',sorted, LIST, LIST)
inner_join3 = Function('inner_join3',sorted, LIST, LIST)
inner_join4 = Function('inner_join4',sorted, LIST, LIST)
anti_join = Function('anti_join',sorted, LIST, LIST)
left_join = Function('left_join',sorted, LIST, LIST)
bind_rows = Function('binf_rows',sorted, LIST, LIST)
intersect = Function('intersect',sorted, LIST, LIST)
select = Function('select',sorted, LIST, LIST)



# lambda functions
# PLUS1 = Function('+1', lambda x: x + 1, INT, INT)
# MINUS1 = Function('-1', lambda x: x - 1, INT, INT)
# TIMES2 = Function('*2', lambda x: x * 2, INT, INT)
# DIV2 = Function('/2', lambda x: int(x / 2), INT, INT)
# TIMESNEG1 = Function('*-1', lambda x: -x, INT, INT)
# POW2 = Function('**2', lambda x: x ** 2, INT, INT)
# TIMES3 = Function('*3', lambda x: x * 3, INT, INT)
# DIV3 = Function('/3', lambda x: int(x / 3), INT, INT)
# TIMES4 = Function('*4', lambda x: x * 4, INT, INT)
# DIV4 = Function('/4', lambda x: int(x / 4), INT, INT)

# GT0 = Function('>0', lambda x: x > 0, INT, BOOL)
# LT0 = Function('<0', lambda x: x < 0, INT, BOOL)
# EVEN = Function('EVEN', lambda x: x % 2 == 0, INT, BOOL)
# ODD = Function('ODD', lambda x: x % 2 == 1, INT, BOOL)
#
# LPLUS = Function('+', lambda x, y: x + y, (INT, INT), INT)
# LMINUS = Function('-', lambda x, y: x - y, (INT, INT), INT)
# LTIMES = Function('*', lambda x, y: x * y, (INT, INT), INT)
# LMIN = Function('min', min, (INT, INT), INT)
# LMAX = Function('max', max, (INT, INT), INT)

# higher-order functions
def _scan1l(f, xs):
    ys = [0] * len(xs)
    for i, x in enumerate(xs):
        if i:
            ys[i] = f(ys[i - 1], x)
        else:
            ys[i] = x
    return ys

# MAP = Function('MAP', lambda f, xs: [f(x) for x in xs], (FunctionType(INT, INT), LIST), LIST)
# FILTER = Function('FILTER', lambda f, xs: [x for x in xs if f(x)], (FunctionType(INT, BOOL), LIST), LIST)
# COUNT = Function('COUNT', lambda f, xs: len([x for x in xs if f(x)]), (FunctionType(INT, BOOL), LIST), INT)
# SCAN1L = Function('SCAN1L', _scan1l, (FunctionType((INT, INT), INT), LIST), LIST)
# ZIPWITH = Function('ZIPWITH', lambda f, xs, ys: [f(x, y) for x, y in zip(xs, ys)], (FunctionType((INT, INT), INT), LIST, LIST), LIST)

LAMBDAS = [
    # PLUS1,
    # MINUS1,
    # TIMES2,
    # DIV2,
    # TIMESNEG1,
    # POW2,
    # TIMES3,
    # DIV3,
    # TIMES4,
    # DIV4,
    # GT0,
    # LT0,
    # EVEN,
    # ODD,
    # LPLUS,
    # LMINUS,
    # LTIMES,
    # LMIN,
    # LMAX,
]

HIGHER_ORDER_FUNCTIONS = [
    # MAP,
    # FILTER,
    # COUNT,
    # SCAN1L,
    # ZIPWITH,
]

FIRST_ORDER_FUNCTIONS = [
    # HEAD,
    # TAIL,
    # MINIMUM,
    # MAXIMUM,
    # REVERSE,
    # SORT,
    # SUM,
    # TAKE,
    # DROP,
    # ACCESS,
    Empty,
    inner_join,
    inner_join3,
    inner_join4,
    anti_join,
    left_join,
    bind_rows,
    intersect,
    select,
]

ALL_FUNCTIONS = FIRST_ORDER_FUNCTIONS + HIGHER_ORDER_FUNCTIONS
FUNCTIONS_AND_LAMBDAS = ALL_FUNCTIONS + LAMBDAS

NAME2FUNC = {x.name : x for x in FUNCTIONS_AND_LAMBDAS}