#pragma once

#include "BirkhoffPolytope.hpp"
#include "StiefelManifold.hpp"

// update multiple matrices of varying sizes on the cartesian product of localparameterization
class ProductParameterization : public ceres::LocalParameterization {
public:
	ProductParameterization(std::vector<ceres::LocalParameterization*> &_parameterizations, bool _takeOwnership = false)
	: takeOwnership(_takeOwnership) 
	{
		parameterizations = _parameterizations;
		numManifolds = _parameterizations.size();
		localSize = 0;
		globalSize = 0;
		cumGlobalSizes.resize(numManifolds + 1);
		cumLocalSizes.resize(numManifolds + 1);
		for (size_t i = 0; i < numManifolds; i++)
		{
			localSizes[i] = parameterizations[i]->LocalSize();
			globalSizes[i] = parameterizations[i]->GlobalSize();
			globalSize += globalSizes[i];
			localSize += localSizes[i];
			cumGlobalSizes[i + 1] += cumGlobalSizes[i] + globalSizes[i];
			cumLocalSizes[i + 1] += cumLocalSizes[i] + localSizes[i];
		}
	}
	virtual ~ProductParameterization() {
		if (takeOwnership) {
			for (size_t i = 0; i < numManifolds; i++) {
				delete parameterizations[i];
				parameterizations[i] = NULL;
			}
		}
	}
	virtual bool Plus(const double* x,
		const double* delta,
		double* x_plus_delta) const
	{
		for (size_t i = 0; i < numManifolds; i++) {
			parameterizations[i]->Plus(&x[cumGlobalSizes[i]], &delta[cumLocalSizes[i]], &x_plus_delta[cumGlobalSizes[i]]);
		}
		return true;
	};

	// if MultiplyByJacobian is implemented ComputeJacobian is not called.
	virtual bool ComputeJacobian(const double* x, double* jacobian) const
	{
		return false;
	}

	// It is always the case that num_rows=1
	virtual bool MultiplyByJacobian(const double *x, const int num_rows, const double *global_matrix, double *local_matrix) const
	{
		for (size_t i = 0; i < numManifolds; i++) {
			parameterizations[i]->MultiplyByJacobian(&x[cumGlobalSizes[i]], num_rows, &global_matrix[cumGlobalSizes[i]], &local_matrix[cumLocalSizes[i]]);
		}
		return true;
	}

	virtual int GlobalSize() const {
		return globalSize;
	}
	virtual int LocalSize() const {
		return localSize;
	}

	bool takeOwnership; // erase local parameterizations at destruction? 
	size_t numManifolds;
	int M, N;
	int localSize, globalSize;
	std::vector<int> localSizes, globalSizes; /* local and global sizes for manifolds can be different */
	std::vector<int> cumLocalSizes, cumGlobalSizes; /* cumulative arrays for indirect indexing */
	std::vector<ceres::LocalParameterization*> parameterizations;
};
