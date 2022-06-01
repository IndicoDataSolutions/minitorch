from minitorch import MathTestVariable
from minitorch import central_difference, operators, derivative_check, Scalar


one_arg, two_arg, _ = MathTestVariable._tests()

for name, _, scalar_fn in one_arg:
    print("====")
    print(scalar_fn)
    t1 = Scalar(0.0)
    derivative_check(scalar_fn, t1)
