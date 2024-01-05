/**
 * \file real_type.h
 * \brief Define macros to switch single/double precision.
 *
 */
#ifndef REAL_TYPE_H_
#define REAL_TYPE_H_

#include <math.h>

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif // USE_DOUBLE


// math function
#if KOKKOS_VERSION_MAJOR > 3
using Kokkos::exp;
using Kokkos::fmax;
using Kokkos::fmin;
using Kokkos::sqrt;
using Kokkos::fabs;
using Kokkos::fmod;
using Kokkos::isnan;
using Kokkos::fmod;
using Kokkos::sin;
using Kokkos::cos;
using Kokkos::tan;
using Kokkos::asin;
using Kokkos::acos;
using Kokkos::atan;
using Kokkos::atan2;
#else
using Kokkos::Experimental::exp;
using Kokkos::Experimental::fmax;
using Kokkos::Experimental::fmin;
using Kokkos::Experimental::sqrt;
using Kokkos::Experimental::fabs;
using Kokkos::Experimental::fmod;
using Kokkos::Experimental::isnan;
using Kokkos::Experimental::fmod;
using Kokkos::Experimental::sin;
using Kokkos::Experimental::cos;
using Kokkos::Experimental::tan;
using Kokkos::Experimental::asin;
using Kokkos::Experimental::acos;
using Kokkos::Experimental::atan;
using Kokkos::Experimental::atan2;
#endif

#if defined(KOKKOS_ENABLE_CXX17)
#  define KOKKOS_IMPL_MATH_CONSTANT(TRAIT, VALUE) \
    template <class T>                            \
    inline constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point_v<T>, T>(VALUE)
#else
#  define KOKKOS_IMPL_MATH_CONSTANT(TRAIT, VALUE) \
    template <class T>                            \
    constexpr auto TRAIT##_v = std::enable_if_t<std::is_floating_point<T>::value, T>(VALUE)
#endif

KOKKOS_IMPL_MATH_CONSTANT(ZERO, 0.000000000000000000000000000000000000L);
KOKKOS_IMPL_MATH_CONSTANT(HALF, 0.500000000000000000000000000000000000L);
KOKKOS_IMPL_MATH_CONSTANT(ONE, 1.000000000000000000000000000000000000L);
KOKKOS_IMPL_MATH_CONSTANT(TWO, 2.000000000000000000000000000000000000L);
KOKKOS_IMPL_MATH_CONSTANT(ONE_FOURTH, 0.250000000000000000000000000000000000L);
KOKKOS_IMPL_MATH_CONSTANT(PI, 3.141592653589793238462643383279502884L);

#undef KOKKOS_IMPL_MATH_CONSTANT

constexpr auto ZERO_F = ZERO_v<real_t>;
constexpr auto HALF_F = HALF_v<real_t>;
constexpr auto ONE_F = ONE_v<real_t>;
constexpr auto TWO_F = TWO_v<real_t>;
constexpr auto ONE_FOURTH_F = ONE_FOURTH_v<real_t>;
constexpr auto PI_F = PI_v<real_t>;

// math function
#if defined(USE_DOUBLE)
#  define COPYSIGN(x, y) copysign(x, y)
#else
#  define COPYSIGN(x, y) copysignf(x, y)
#endif // USE_DOUBLE

// other usefull macros
#define SQR(x) ((x)*(x))

#endif // REAL_TYPE_H_
