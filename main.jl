import Pkg
using BenchmarkTools
import PyPlot as plt
import Base: +,-,*,/,^


#### first order derivatives ####


struct Dual{T<:Real}
    real::T
    dual::T
end


#### Dual to Dual operations ####


function Base.:+(x::Dual, y::Dual)
    x₀, x₁ = x.real, x.dual
    y₀, y₁ = y.real, y.dual
    xy = Dual(x₀ + y₀, x₁ + y₁)
    return xy
end


function Base.:-(x::Dual, y::Dual)
    x₀, x₁ = x.real, x.dual
    y₀, y₁ = y.real, y.dual
    xy = Dual(x₀ - y₀, x₁ - y₁)
    return xy
end


function Base.:*(x::Dual, y::Dual)
    x₀, x₁ = x.real, x.dual
    y₀, y₁ = y.real, y.dual
    xy = Dual(x₀ * y₀, x₁ * y₀ + x₀ * y₁)
    return xy
end


function Base.:/(x::Dual, y::Dual)
    x₀, x₁ = x.real, x.dual
    y₀, y₁ = y.real, y.dual
    xy = Dual(x₀ / y₀, (x₁ * y₀ - x₀ * y₁) / (y₀^2))
    return xy
end


#### Dual to Other operations ####

function Base.:^(x::Dual, n::Int64)
    x_out = Dual(1, 0)
    if n == 0
        return x_out
    else
        for _ in 1:n
            x_out *= x
        end
        return x_out
    end
end


function Base.:+(x::Union{Float64, Int64}, y::Dual)
    return Dual(y.real + x, y.dual)
end


function Base.:+(x::Dual, y::Union{Float64, Int64})
    return Dual(x.real + y, x.dual)
end


function Base.:-(x::Union{Float64, Int64}, y::Dual)
    return Dual(x - y.real, -y.dual)
end


function Base.:-(x::Dual, y::Union{Float64, Int64})
    return Dual(x.real - y, x.dual)
end


function Base.:*(x::Union{Float64, Int64}, y::Dual)
    return Dual(y.real * x, y.dual * x)
end


function Base.:*(x::Dual, y::Union{Float64, Int64})
    return Dual(x.real * y, x.dual * y)
end


function Base.:/(x::Union{Float64, Int64}, y::Dual)
    return Dual(x / y.real, x * y.dual / (y.real ^ 2))
end


function Base.:/(x::Dual, y::Union{Float64, Int64})
    return Dual(x.real / y, x.dual / y)
end


#### 1st derivative ####


function d1(f, x::Float64)   # f must be a combination of basic operations
    x_prime = Dual(x, 1)
    return f(x_prime).dual
end


#### testing ####

test_f(x) = -4 + 1.2x - 3x^2 + 4x^3 + x^4 - 0.1x^6

x = -3:0.01:3
y = test_f.(x)
y_prime = d1.(test_f, x)

plt.plot(x, y, color="b")
plt.plot(x, y_prime, color="r")
plt.show()
