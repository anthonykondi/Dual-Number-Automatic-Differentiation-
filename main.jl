import Pkg
using BenchmarkTools
import PyPlot as plt
import Base: +, -, *, /, ^, exp, log, sin, cos


#### Hyper-Dual Numbers struct ####

struct HDual{T<:Real, U<:Real, V<:Real, W<:Real}
    real::T
    dual1::U
    dual2::V
    dual12::W
end


#### HDual to HDual operations ####


+(x::HDual, y::HDual) = HDual(
    x.real + y.real, 
    x.dual1 + y.dual1, 
    x.dual2 + y.dual2,
    x.dual12 + y.dual12
)

-(x::HDual, y::HDual) = HDual(
    x.real - y.real, 
    x.dual1 - y.dual1,
    x.dual2 - y.dual2,
    x.dual12 - y.dual12
)

*(x::HDual, y::HDual) = HDual(
    x.real * y.real, 
    x.real * y.dual1 + x.dual1 * y.real,
    x.real * y.dual2 + x.dual2 * y.real,
    x.real * y.dual12 + x.dual1 * y.dual2 + x.dual2 * y.dual1 + x.dual12 * y.real 
)

inv(x::HDual) = HDual(
    1 / x.real,
    -x.dual1 / (x.real^2),
    -x.dual2 / (x.real^2),
    (2 * x.dual1 * x.dual2 / (x.real^3)) - (x.dual12 / (x.real^2)) 
)

/(x::HDual, y::HDual) = x * inv(y)

# also need to define negative (-)
-(x::HDual) = HDual(-x.real, -x.dual1, -x.dual2, -x.dual12)


#### Dual to Other operations ####
# could also use promotion rules to avoid having to implement every possible combination

+(x::HDual, y::Real) = HDual(x.real + y, x.dual1, x.dual2, x.dual12)

+(x::Real, y::HDual) = y + x    # if you do x + y you create ∞ recursion (stack overflow)

-(x::HDual, y::Real) = HDual(x.real - y, x.dual1, x.dual2, x.dual12)

-(x::Real, y::HDual) = HDual(x - y.real, -y.dual1, -y.dual2, -y.dual12)

*(x::HDual, y::Real) = HDual(x.real * y, x.dual1 * y, x.dual2 * y, x.dual12 * y)

*(x::Real, y::HDual) = y * x

/(x::Real, y::HDual) = x * inv(y)

/(x::HDual, y::Real) = HDual(x.real / y, x.dual1 / y, x.dual2 / y, x.dual12 / y)


#### other functions (using their Taylor series) ####

^(x::HDual, n::Real) = HDual(
    x.real^n,
    x.dual1 * n * x.real^(n - 1),
    x.dual2 * n * x.real^(n - 1),
    x.dual12 * n * x.real^(n - 1) + x.dual1 * x.dual2 * n * (n - 1) * x.real^(n - 2)
)

exp(x::HDual) = HDual(
    exp(x.real),
    x.dual1 * exp(x.real),
    x.dual2 * exp(x.real),
    x.dual12 * exp(x.real) + x.dual1 * x.dual2 * exp(x.real)
)

log(x::HDual) = HDual(
    log(x.real),
    x.dual1 / x.real,
    x.dual2 / x.real,
    x.dual12 / x.real - x.dual1 * x.dual2 / (x.real^2)
)

sin(x::HDual) = HDual(
    sin(x.real),
    x.dual1 * cos(x.real),
    x.dual2 * cos(x.real),
    x.dual12 * cos(x.real) - x.dual1 * x.dual2 * sin(x.real)
)

cos(x::HDual) = HDual(
    cos(x.real),
    -x.dual1 * sin(x.real),
    -x.dual2 * sin(x.real),
    -x.dual12 * sin(x.real) - x.dual1 * x.dual2 * cos(x.real)
)

tan(x::HDual) = sin(x) / cos(x)

sinh(x::HDual) = (exp(x) - exp(-x)) / 2

cosh(x::HDual) = (exp(x) + exp(-x)) / 2

tanh(x::HDual) = sinh(x) / cosh(x)


#### 1st and 2nd derivative ####

d1(f, x) = f(HDual(x, 1, 0, 0)).dual1
d2(f, x) = f(HDual(x, 1, 1, 0)).dual12


#### testing the derivatives ####

test_f1(x) = -4 + 1.2x - 3x^2 + 4x^3
test_f2(x) = x ^ (0.5)
test_f3(x) = exp(sin(x))
test_f4(x) = exp(-x^2) * sin(2π*x)
test_f5(x) = 1 / (1 + sin(x)^2)
test_f6(x) = test_f5(test_f1(test_f3(x)))

# f = test_f6

# x = -10:0.001:10
# y = f.(x)
# y_prime = d1.(f, x)
# y_dprime = d2.(f, x)

# plt.plot(x, y, color="b")
# plt.plot(x, y_prime, color="r")
# plt.plot(x, y_dprime, color="g")
# plt.show()





