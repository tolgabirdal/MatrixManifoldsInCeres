#pragma once

#include "TCommons.hpp"

using namespace Eigen;

template<typename T>
static void Stiefel_projection_SVD(MatrixXd X, T& U)
{
	const auto &svd = X.jacobiSvd(ComputeThinU | ComputeThinV);
	U = svd.matrixU()*svd.matrixV().transpose();
}

static MatrixXd Stiefel_projection_SVD(MatrixXd &X)
{
	const auto &svd = X.jacobiSvd(ComputeThinU | ComputeThinV);
	return svd.matrixU()*svd.matrixV().transpose();
}

// many projections (retractions) are possible - I use the polar.
#define Stiefel_projection Stiefel_projection_SVD

template <typename T>
static void Stiefel_retract(const MatrixXd X, const MatrixXd U, T &Y, double tau = 1.0)
{
	Stiefel_projection((X + tau * U).eval(), Y);
}

// projection onto tangent space
template<typename T>
static void Stiefel_projection_TxM(MatrixXd X, MatrixXd U, T& XU)
{
	XU = X.transpose() * U;
	XU = U - X*(.5*(XU + XU.transpose()));
}

static MatrixXd Stiefel_rand(size_t N, size_t K)
{
	MatrixXd result = MatrixXd::Random(N, K) + MatrixXd::Ones(N, K);
	return Stiefel_projection_SVD(result);
}

static double Stiefel_distance(MatrixXd &X1, MatrixXd &X2)
{
	return X1.rows() - (X2.transpose()*X1).trace();
}

static bool Stiefel_check(MatrixXd &X, double tol = 0.00001)
{
	return (std::abs(Stiefel_distance(X, X)) < tol);
}


// manifold of orthonormal NxK matrices
class StiefelParameterization : public ceres::LocalParameterization {
public:
	StiefelParameterization(int _N, int _K) : K(_K), N(_N) {}
	virtual ~StiefelParameterization() {}
	virtual bool Plus(const double* x,
		const double* delta,
		double* x_plus_delta) const
	{
		const Map<const MatrixXd> P0(x, N, K);
		const Map<const MatrixXd> U(delta, N, K);

		// retract back on the manifold
		Map<MatrixXd> xPlusDelta(x_plus_delta, N, K);
		Stiefel_retract(P0, U, xPlusDelta);

		return true;
	};

	// if MultiplyByJacobian is implemented ComputeJacobian is not called.
	virtual bool ComputeJacobian(const double* x, double* jacobian) const
	{
		return false;
	}

	virtual bool MultiplyByJacobian(const double *x, const int num_rows, const double *global_matrix, double *local_matrix) const
	{
		const Map<const MatrixXd> P0(x, N, K);
		const Map<const MatrixXd> U(global_matrix, N, K);
		Map<MatrixXd> UProj(local_matrix, N, K);
		Stiefel_projection_TxM(P0, U, UProj);
		return true;
	}

	virtual int GlobalSize() const { return K * N; }
	virtual int LocalSize() const { return K * N; }

	int N, K;
};

