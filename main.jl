import Pkg
using BenchmarkTools
import PyPlot as plt
import Base: +, -, *, /, ^, exp, log, sin, cos
using Random


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

#### minimization (Newton's method on derivatives) ####


newton_step(f, x) = x - (d1(f, x) / d2(f, x))    # this doesn't handle infinity well (zero curvature regions)

"""
Extremizes a function f on an interval [a, b].

a < b defines the region where the initial guesses will be.
f is the function you wish to extremize.
n_guesses is how many initial guesses will be made and iterated over by newton's method.
epsilon is the error at which |x_n+1 - x_n| is small enough that you can take x_n+1 as a candidate extreme value. 

-> returns a 2 element array with 1st value being the minimum x coord, and 2nd value the maximum x coord.
"""
function extremize_interval(f, a::Float64, b::Float64, n_guesses::Int, epsilon::Float64=0.01)
    spacing = (b - a) / n_guesses
    range = a:spacing:b
    extreme_x_coords = f(a) < f(b) ? [a, b] : [b, a]    # setting the initial extrema values

    for x₀ in range
        n_iter = 1              # going to cut off the loop if too many iterations occur without convergence
        x_old = x₀              # 1 iteration out of the while loop to initialize the x variables
        x_new = newton_step(f, x_old)

        if x_new == Inf || x_new == -Inf
            continue
        end

        while abs(x_new - x_old) > epsilon && n_iter < 20
            x_old = x_new
            x_new = newton_step(f, x_old)

            if x_new < a    # value falls out of range
                x_new = a
                x_old = a
            elseif x_new > b
                x_new = b
                x_old = b 
            end
        end

        if f(x_new) < f(extreme_x_coords[1])    # comparing new value to the old ones
            extreme_x_coords[1] = x_new
        elseif f(x_new) > f(extreme_x_coords[2])
            extreme_x_coords[2] = x_new
        end
    end

    return extreme_x_coords
end


#### random order n polynomial function ####

function rand_poly_coeffs(n::Int64, scale::Real, factorial_supression_on::Bool)
    if factorial_supression_on
        factorial_supression = [1 / (factorial(k)) for k in 0:1:n]
    else 
        factorial_supression = ones((n + 1,))
    end
    unscaled_coeffs = scale .* (2 .* rand(Float64, (n + 1,)) .- 1)    # ∈ [-scale, +scale]
    coeffs = unscaled_coeffs .* factorial_supression
    return coeffs
end


polynomial(x, coeffs::Vector) = sum([coeffs[k] * x^(k - 1) for k in 1:1:length(coeffs)]) 


#### testing the derivatives ####

# test functions 
test_f1(x) = -4 + 1.2x - 3x^2 + 4x^3
test_f2(x) = x ^ (0.5)
test_f3(x) = exp(sin(x))
test_f4(x) = exp(-x^2) * sin(2π*x)
test_f5(x) = exp(1 / (1 + sin(x)^2)) * sin(1 / (1 + x^2))
test_f6(x) = test_f5(test_f1(test_f3(x)))
test_f7(x) = (x - 3.5)^2 + 10
test_f8(x) = 5*(2(x/8) + (x/8)^2 - (x/8)^3)


# obtain random polynomial coefficients
# coeffs = rand_poly_coeffs(10, 100, true)

# f(x) = polynomial(x, coeffs)

f(x) = test_f5(x)


extrema = extremize_interval(f, -10., 10., 10, 0.001)

# evaluating and plotting
x = -10:0.001:10
y = f.(x)
y_prime = d1.(f, x)
y_dprime = d2.(f, x)

plt.plot(x, y, color="b", label="Function f(x)")
plt.plot(x, y_prime, color="r", label="1st derivative")
plt.plot(x, y_dprime, color="g", label="2nd derivative")
plt.scatter(extrema, f.(extrema), marker="o", color="r", label="Extrema")
plt.legend()

plt.savefig("example(2).png")

plt.show() 




