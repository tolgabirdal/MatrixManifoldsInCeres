#pragma once

#ifndef __T_COMMONS_HPP_
#define __T_COMMONS_HPP_

#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include "ceres/ceres.h"

using namespace std;
using namespace Eigen;

const double EPSILON = 0.0001;

// define TAPI_EXPORTS to export from dll
#if (defined WIN32 || defined _WIN32 || defined WINCE) && defined TAPI_EXPORTS
# define T_EXPORTS __declspec(dllexport)
#else
# define T_EXPORTS
#endif

static bool equals(double a, double b) {
	return std::fabs(a - b)<EPSILON;
}

//supply tolerance that is meaningful in your context
//for example, default tolerance may not work if you are comparing double with float
template<typename T>
static bool isApproximatelyZero(T a, T tolerance = EPSILON)
{
	return (std::fabs(a) <= tolerance);
}


#endif