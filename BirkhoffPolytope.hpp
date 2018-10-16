#pragma once

/*
Riemannian structure of the doubly stochastic matrices:
 @Techreport{Douik2018Manifold,
   Title   = {Manifold Optimization Over the Set of Doubly Stochastic
              Matrices: {A} Second-Order Geometry},
   Author  = {Douik, A. and Hassibi, B.},
   Journal = {Arxiv preprint ArXiv:1802.02628},
   Year    = {2018}
}
*/

#include "TCommons.hpp"
#include "ceres/ceres.h"

// default values for Sinkhorn are taken from:
// THE SINKHORN-KNOPP ALGORITHM: CONVERGENCE AND APPLICATIONS, PHILIP A. KNIGHT
// Fig. 6.1 at 1e-4 #iterations~=40
// this effects the speed a lot, so I will not go for a lot of iterations
const double SINKHORN_EPSILON = 0.0001;
const double SINKHORN_MAXITER = 40;

// projects onto the Birkhoff Polytope
template<typename T>
static void sinkhorn(T& A, int maxIterations = SINKHORN_MAXITER, double tol = SINKHORN_EPSILON)
{
	// first iteration - no test
	int iter = 1;
	MatrixXd c = A.colwise().sum().cwiseInverse();
	VectorXd r = (A * c.transpose()).cwiseInverse();

	// subsequent iterations include test
	while (iter < maxIterations) {
		iter++;

		MatrixXd cinv = r.transpose() * A;

		// test whether the tolerance was achieved on the last iteration
		if (cinv.cwiseProduct(c).cwiseAbs().maxCoeff() < tol)
			break;

		c = cinv.cwiseInverse();
		r = (A * c.transpose()).cwiseInverse();
	}

	A = A.cwiseProduct((r * c));
}

// project eta onto the tangent space of X
static MatrixXd DSn_project_TxM(const MatrixXd &X, const MatrixXd &eta)
{
	size_t N = X.rows();
	MatrixXd I = MatrixXd::Identity(N, N);
	MatrixXd A(2 * N, 2 * N); // , B(2 * N, 2 * N - 1);
	A << I, X,
		X.transpose(), I;
	MatrixXd B = A.block(0, 1, 2 * N, 2 * N - 1);

	VectorXd b(2 * N);
	b << eta.rowwise().sum(),
		eta.colwise().sum().transpose();

	VectorXd zeta = B.colPivHouseholderQr().solve((b - A.col(0)));

	VectorXd alpha(N);
	alpha << 1,
		zeta.block(0, 0, N - 1, 1);

	// beta = zeta(n:2 * n - 1);
	VectorXd beta = zeta.block(N - 1, 0, N, 1);
	VectorXd e = VectorXd::Ones(N);
	MatrixXd etaproj = eta - (alpha*e.transpose() + e * beta.transpose()).cwiseProduct(X);
	return etaproj;
}

static void DSn_project_TxM(const MatrixXd &X, const MatrixXd &eta, Map<MatrixXd>& etaproj)
{
	size_t N = X.rows();
	MatrixXd I = MatrixXd::Identity(N, N);
	MatrixXd A(2 * N, 2 * N); // , B(2 * N, 2 * N - 1);
	A << I, X,
		X.transpose(), I;
	MatrixXd B = A.block(0, 1, 2 * N, 2 * N - 1);

	VectorXd b(2 * N);
	b << eta.rowwise().sum(),
		eta.colwise().sum().transpose();

	VectorXd zeta = B.colPivHouseholderQr().solve((b - A.col(0)));

	VectorXd alpha(N);
	alpha << 1,
		zeta.block(0, 0, N - 1, 1);

	// beta = zeta(n:2 * n - 1);
	VectorXd beta = zeta.block(N - 1, 0, N, 1);
	VectorXd e = VectorXd::Ones(N);
	etaproj = eta - (alpha*e.transpose() + e * beta.transpose()).cwiseProduct(X);
}

static void DSn_project_TxM(const vector<MatrixXd> &X, const vector<MatrixXd> &eta, vector<Map<MatrixXd>>& etaproj)
{
	vector<MatrixXd> Ps;
	Ps.resize(X.size());
	for (size_t i = 0; i < X.size(); i++)
		DSn_project_TxM(X[i], eta[i], etaproj[i]);
}

static double Exp(double x) // the functor we want to apply
{
	if (_isnan(x) || isinf(x) || x == -INFINITY)
		return 0;
	else
		return std::exp(x);
}

template<typename T>
static MatrixXd DSn_retract(const MatrixXd &X, const MatrixXd &eta, T& Y, double tau = 1.0)
{
	Y = X.cwiseProduct((tau * (eta.cwiseQuotient(X))).unaryExpr(&Exp));
	sinkhorn<T>(Y);
	return Y.cwiseMax(FLT_EPSILON);
}

template<typename T>
static bool DSn_check(T &X, double tol = 0.000001)
{
	bool isBirkhoff = (!(X.array() < -tol).any()); // check if any negative exists
	for (size_t i = 0; i < X.rows() && isBirkhoff; i++) // check all rows sum to 1
		isBirkhoff = (isApproximatelyZero(1.0 - X.row(i).sum(), tol));
	for (size_t i = 0; i < X.cols() && isBirkhoff; i++) // check all cols sum to 1
		isBirkhoff = (isApproximatelyZero(1.0 - X.col(i).sum(), tol));
	return isBirkhoff;
}

static bool DSn_check(vector<MatrixXd> &X)
{
	for (size_t i = 0; i < X.size(); i++)
		if (!DSn_check(X[i]))
			return false;
	return true;
}

static bool DSn_check_tangent(MatrixXd &Z)
{
	const double tau = 0.000001;
	VectorXd one = VectorXd::Ones(Z.rows());
	return ((Z*one).norm()<tau && (Z.transpose()*one).norm()<tau);
}

static MatrixXd DSn_rand(size_t N)
{
	MatrixXd result = MatrixXd::Random(N, N) + MatrixXd::Ones(N, N);
	sinkhorn<MatrixXd>(result, 100, 0.000001);
	return result;
}

static vector<MatrixXd> DSn_rand(size_t N, size_t numMatrices)
{
	vector<MatrixXd> result;
	result.resize(numMatrices);
	for (size_t i = 0; i < numMatrices; i++)
		result[i] = DSn_rand(N);
	return result;
}

// perturb the DS. this function is especially useful for birkhoff because the manifold is not
// well defined towards the vertices
static void DSn_via_perturb(MatrixXd &X, double perturbation = 0.001)
{
	X += (perturbation*(MatrixXd::Random(X.rows(), X.cols()) + MatrixXd::Ones(X.rows(), X.cols()))/2);
	sinkhorn(X);
}

static void DSn_via_perturb(std::vector<MatrixXd> &X, double perturbation = 0.001)
{
	for (size_t i = 0; i < X.size(); i++)
		DSn_via_perturb(X[i]);
}

class BirkhoffParameterization : public ceres::LocalParameterization {
public:
	BirkhoffParameterization(int _N) : N(_N) {}
	virtual ~BirkhoffParameterization() {}
	virtual bool Plus(const double* x,
		const double* delta,
		double* x_plus_delta) const
	{
		const Map<const MatrixXd> P0(x, N, N);
		const Map<const MatrixXd> U(delta, N, N);

		// retract back on the manifold
		Map<MatrixXd> xPlusDelta(x_plus_delta, N, N);
		DSn_retract(P0, U, xPlusDelta);

		return true;
	};

	// if MultiplyByJacobian is implemented ComputeJacobian is not called.
	virtual bool ComputeJacobian(const double* x, double* jacobian) const
	{
		return false;
	}

	virtual bool MultiplyByJacobian(const double *x, const int num_rows, const double *global_matrix, double *local_matrix) const
	{
		return Egrad2Rgrad(x, num_rows, global_matrix, local_matrix);
	}

	virtual bool Egrad2Rgrad(const double *x, const int num_rows, const double *egrad, double *rgrad) const
	{
		const Map<const MatrixXd> P0(x, N, N);
		const Map<const MatrixXd> U(egrad, N, N);
		Map<MatrixXd> UProj(rgrad, N, N);
		MatrixXd UP0 = U.cwiseProduct(P0);
		DSn_project_TxM(P0, UP0, UProj);
		return true;
	}

	virtual int GlobalSize() const { return N * N; }
	virtual int LocalSize() const { return N * N; }

	int N;
};

// update N DS matrices of size MxM on parallel Birkhoff polytopes (cartesian product)
class BirkhoffProductParameterization : public ceres::LocalParameterization {
public:
	BirkhoffProductParameterization(int _N, int _M) : N(_N), M(_M) {}
	virtual ~BirkhoffProductParameterization() {}
	virtual bool Plus(const double* x,
		const double* delta,
		double* x_plus_delta) const
	{
		for (int i = 0; i < N; i++) {
			const Map<const MatrixXd> P0(&x[i*M*M], M, M);
			const Map<const MatrixXd> U(&delta[i*M*M], M, M);
			Map<MatrixXd> xPlusDelta(&x_plus_delta[i*M*M], M, M);
			DSn_retract(P0, U, xPlusDelta);
		}
		return true;
	};

	// if MultiplyByJacobian is implemented ComputeJacobian is not called.
	virtual bool ComputeJacobian(const double* x, double* jacobian) const
	{
		return false;
	}

	virtual bool MultiplyByJacobian(const double *x, const int num_rows, const double *global_matrix, double *local_matrix) const
	{
		return Egrad2Rgrad(x, num_rows, global_matrix, local_matrix);
	}

	virtual bool Projection(const double *x, const int num_rows, const double *global_matrix, double *local_matrix) const
	{
		for (int i = 0; i < N; i++) {
			const Map<const MatrixXd> P0(&x[i*M*M], M, M);
			const Map<const MatrixXd> U(&global_matrix[i*M*M], M, M);
			Map<MatrixXd> UProj(&local_matrix[i*M*M], M, M);
			DSn_project_TxM(P0, U, UProj);
		}

		return true;
	}

	virtual bool Egrad2Rgrad(const double *x, const int num_rows, const double *egrad, double *rgrad) const
	{
		for (int i = 0; i < N; i++) {
			const Map<const MatrixXd> P0(&x[i*M*M], M, M);
			const Map<const MatrixXd> U(&egrad[i*M*M], M, M);
			Map<MatrixXd> UProj(&rgrad[i*M*M], M, M);
			MatrixXd UP0 = U.cwiseProduct(P0);
			DSn_project_TxM(P0, UP0, UProj);
		}

		return true;
	}

	virtual int GlobalSize() const {
		return M * M*N;
	}
	virtual int LocalSize() const {
		return M * M*N;
	}

	int N, M;
};
