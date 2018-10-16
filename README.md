Matrix Manifold Local Parameterizations for Ceres Solver
========================================================

Google's Ceres solver (http://ceres-solver.org/) is just a great tool for solving unconstrained non-linear least squares problems. Many of us engaged in 3D computer vision used it heavily to tackle camera calibration, multiview reconstruction, tracking, SLAM and etc. In an abundance of these computer vision problems, there exists an underlying geometric structure of the optimization variables (or parameter blocks in Ceres terminology). For example, a sphere in three dimensions is a two dimensional manifold, embedded in a three dimensional space. Using the 2D parameterization removes a redundant dimension from the optimization, making it numerically more robust and efficient. Ceres allows us to exploit this geometry by restricting the update of the parameteres to the **Riemannian manifold** of these parameters. For instance, EigenQuaternionParameterization [2] in Ceres, or Sophus Lie group library [3] do exactly that.

**LocalParameterization** class in Ceres requires us to implement two inherited functions : *Plus* and *ComputeJacobian*:

```C++
virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const = 0;
virtual bool ComputeJacobian(const double* x, double* jacobian) const = 0;
```

Plus overrides the standard + operation of the update, and enables the walk on the manifold (e.g. by the geodesic flow). *ComputeJacobian* though, is the differentiation of the plus operation with respect to the perturbations - that is . For instance, for the case of rigid poses *ComputeJacobian* would differentiate the incremented states w.r.t to the incremental Lie algebra, whereas *Plus* would designate the state increment method [1]. Once these are implemented, the standard trust region solvers work just the same way.

For some problems where the number of parameters are just too large, the aforementioned Jacobian computation becomes very memory intensive. If one is to update an N=100x100 matrix on the manifold, the Jacobian would be 10000 x 10000 matrix. This creates a challenge in for instance operating on the Matrix manifolds. Ideally, in such cases, we like to avoid Jacobian based solvers (Gauss-Newton, Levenberg Marquadt etc.) and resort to the gradient based ones (steepest descent or line-search methds such as LBFGS). **Ceres has both!** But this part of the library remains still undocumented, it is unclear how to make these methods work on the manifolds. The key is **MultiplyByJacobian**:

```C++
virtual bool MultiplyByJacobian(const double* x, const int num_rows,
                            const double* global_matrix, double* local_matrix) const;
```

In practice, instead of implementing *ComputeJacobian* it is possible to implement directly *MultiplyByJacobian*. In the simplest setting, the replacement would be (assembled from Ceres test functions):

```C++
// some dummy variables
const int kGlobalSize = 4; // size in the ambient space
const int kLocalSize = 3; // dimension in the tangent space

// some matrices in the global and local spaces
Matrix global_matrix = Matrix::Ones(10, kGlobalSize);
Matrix local_matrix = Matrix::Zero(10, kLocalSize);
  
// typical ComputeJacobian operation:
double jacobian[kGlobalSize * kLocalSize];
parameterization.ComputeJacobian(x, jacobian);
Matrix expected_local_matrix = global_matrix * MatrixRef(jacobian, kGlobalSize, kLocalSize);

// the lines below should generate the same result as the previous 2 lines combined.
parameterization.MultiplyByJacobian(x,10,global_matrix.data(),local_matrix.data());
```

**Geometric Interpretation of the MultiplyByJacobian**: At the first sight, to non-experts, it might seem unclear how this operation relates geometrically to the typical Riemannian operations such as projection, retraction, geodesic flow, exp, log and all that. The short answer is ***MultiplyByJacobian* maps the Euclidean gradient to the Riemannian gradient**. Quoting Sameer Agarwal:

> MultiplyByJacobian is not a generic projection operator onto the tangent space, but rather the matrix that takes the gradient vector/Jacobian matrix in the ambient coordinates to the tangent space.  What MultiplyByJacobian does is that instead of computing the Jacobian of the Plus operator at delta = 0, and then applying it to the gradient/Jacboian, it lets the user define how it is to be done, especially for high dimensional spaces. 

Note that all in all this method is very similar to what ***egrad2rgrad*** function of ManOpt [4]. 

With these operations well understood, it is possible to implement local parameterization operations regarding matrix manifolds such as Stiefel, Grassmann, Birkhoff and etc. We can then use these in line-search methods to perform optimization over large matrices **efficiently and easily**.

## Dependencies

Only dependencies are Google's Ceres solver itself (http://ceres-solver.org/) and *Eigen* (http://eigen.tuxfamily.org/index.php?title=Main_Page).

## Compilation and Usage

The code is mostly composed of multiple *hpp* files, that one can simply import into a project.

## Sample Code
I include a basic example that finds the closest matrix (in the Frobenius sense) on the manifold, to a given matrix in the ambient space. I guess this is called matrix denoising. 

Below lies a sample main file that can make use of the library. One could call it as follow:

```
sample_BA.exe savedViews.txt savedMatches.txt savedParameters.txt savedViewsBA.txt savedMatchesBA.txt savedParametersBA.txt 0
```

The sample:

```cpp
int main(int argc, char** argv)
{
	std::string fileNameViews = std::string(argv[1]);
	std::string fileNameViewMatches = std::string(argv[2]);
	std::string fileNameParameters = std::string(argv[3]);
	std::string fileNameViewsOut = std::string(argv[4]);
	std::string fileNameViewMatchesOut = std::string(argv[5]);
	std::string fileNameParametersOut = std::string(argv[6]);
	int camToAdjust = atoi(argv[7]);

	std::vector<TView> views;
	std::vector<TViewMatch> viewMatches;
	TBundleAdjustmentParams parameters;
	std::cout << "Loading...\n";
	loadViewsMatchesAndParameters(fileNameViews, fileNameViewMatches, fileNameParameters, views, viewMatches, parameters);

	TBundleAdjustment ba;
	std::vector<Eigen::Vector3d> points3d;
	std::vector<double> reprjErrors;

	std::vector<int> cameraIndicesToAdjust;
	cameraIndicesToAdjust.push_back(camToAdjust);

	parameters.verbose = 1;
	parameters.adjustIntrinsics = false; //don't wanna adjust cam intrinsics
	parameters.adjustmentMode = TBAMode::EPI_REFINE;
	parameters.robustLoss = TLossFunctionType::TUKEY; // use robust tukey loss. this is the default.

	parameters.print();
	viewMatches[0].print();
	views[camToAdjust].print();

	std::cout << "runMultiviewReconstruction...\n";
	bool status = ba.runMultiviewRefinement(views, viewMatches, cameraIndicesToAdjust, parameters, points3d, &reprjErrors);
	views[camToAdjust].print();

	std::cout << "saveViewsMatchesAndParameters...\n";
	saveViewsMatchesAndParameters(fileNameViewsOut, fileNameViewMatchesOut, fileNameParametersOut, views, viewMatches, parameters);

	std::cout << "bye bye!\n";

	return 0;
}
```

### Authors
**Tolga Birdal**  

Enjoy it!

### References

[1] https://fzheng.me/2018/01/23/ba-demo-ceres/
[2] http://www.lloydhughes.co.za/index.php/using-eigen-quaternions-and-ceres-solver/
[3] https://github.com/strasdat/Sophus/blob/master/test/ceres/local_parameterization_se3.hpp
[4] https://www.manopt.org
