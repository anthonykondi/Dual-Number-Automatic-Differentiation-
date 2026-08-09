import Pkg
using BenchmarkTools
import PyPlot as plt
import Base: +, -, *, /, ^, exp, log, sin, cos


#### first order derivatives ####

struct Dual{T<:Real, F<:Real}
    real::T
    dual::F
end


#### Dual to Dual operations ####


+(x::Dual, y::Dual) = Dual(
    x.real + y.real, x.dual + y.dual
)

-(x::Dual, y::Dual) = Dual(
    x.real - y.real, x.dual - y.dual
)

*(x::Dual, y::Dual) = Dual(
    x.real * y.real, x.dual * y.real + x.real * y.dual
)

/(x::Dual, y::Dual) = Dual(
    x.real / y.real, (x.dual * y.real - x.real * y.dual) / (y.real ^ 2)
)

# also need to define negative (-)
-(x::Dual) = Dual(-x.real, -x.dual)


#### Dual to Other operations ####
# could also use promotion rules to avoid having to implement every possible combination

+(x::Dual, y::Real) = Dual(x.real + y, x.dual)

+(x::Real, y::Dual) = y + x    # if you do x + y you create ∞ recursion (stack overflow)

-(x::Dual, y::Real) = Dual(x.real - y, x.dual)

-(x::Real, y::Dual) = Dual(x - y.real, -y.dual)

*(x::Dual, y::Real) = Dual(x.real * y, x.dual * y)

*(x::Real, y::Dual) = y * x

/(x::Real, y::Dual) = Dual(x / y.real, x * y.dual / (y.real ^ 2))

/(x::Dual, y::Real) = Dual(x.real / y, x.dual / y)


#### other functions (using their Taylor series) ####

function ^(x::Dual, n::Real)
    Dual(
        x.real^n,
        n * x.real ^ (n - 1) * x.dual
    )
end

exp(x::Dual) = Dual(
    exp(x.real),
    exp(x.real) * x.dual
)

log(x::Dual) = Dual(
    log(x.real),
    x.dual / x.real
)

sin(x::Dual) = Dual(
    sin(x.real),
    cos(x.real) * x.dual
)

cos(x::Dual) = Dual(
    cos(x.real),
    -sin(x.real) * x.dual
)

tan(x::Dual) = sin(x) / cos(x)

sinh(x::Dual) = (exp(x) - exp(-x)) / 2

cosh(x::Dual) = (exp(x) + exp(-x)) / 2

tanh(x::Dual) = sinh(x) / cosh(x)


#### 1st derivative ####

d1(f, x) = f(Dual(x, 1)).dual


#### second order derivatives ####

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


#### 2nd derivative ####

d2(f, x) = f(HDual(x, 1, 1, 0)).dual12


#### testing ####

test_f1(x) = -4 + 1.2x - 3x^2 + 4x^3
test_f2(x) = x ^ (0.5)
test_f3(x) = exp(sin(x))
test_f4(x) = exp(-x^2) * sin(2π*x)
test_f5(x) = 1 / (1 + sin(x)^2)
test_f6(x) = test_f5(test_f1(test_f3(x)))

f = test_f6

x = -10:0.001:10
y = f.(x)
y_prime = d1.(f, x)

plt.plot(x, y, color="b")
plt.plot(x, y_prime, color="r")
plt.show()
