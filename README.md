Matrix Manifold Local Parameterizations for Ceres Solver
========================================================

Google's Ceres solver (http://ceres-solver.org/) is just a great tool for solving unconstrained non-linear least squares problems. Many of us engaged in 3D computer vision used it heavily to tackle camera calibration, multiview reconstruction, tracking, SLAM and etc. In an abundance of these computer vision problems, there exists an underlying geometric structure of the optimization variables (or parameter blocks in Ceres terminology). For example, a sphere in three dimensions is a two dimensional manifold, embedded in a three dimensional space. Using the 2D parameterization removes a redundant dimension from the optimization, making it numerically more robust and efficient. Ceres allows us to exploit this geometry by restricting the update of the parameteres to the **Riemannian manifold** of these parameters. For instance, EigenQuaternionParameterization [2] in Ceres, or Sophus Lie group library [3] do exactly that.

The goal of this repo is to extend Ceres towards a Riemannian optimization library! Let's begin.

**LocalParameterization** class in Ceres requires us to implement two inherited functions : *Plus* and *ComputeJacobian*:

```C++
virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const = 0;
virtual bool ComputeJacobian(const double* x, double* jacobian) const = 0;
```

Plus overrides the standard + operation of the update, and enables the walk on the manifold (e.g. by the geodesic flow). *ComputeJacobian* though, is the differentiation of the plus operation with respect to the perturbations - that is . For instance, for the case of rigid poses *ComputeJacobian* would differentiate the incremented states w.r.t to the incremental Lie algebra, whereas *Plus* would designate the state increment method [1]. Once these two are implemented, the standard trust region solvers work just the same way.

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

#### Geometric Interpretation of the Plus
Often times *Plus* is related to the retraction of a move in the tangent space. While standard Ceres parameterizations (quaternion, se3 etc.) use analytical exponential maps (Exp) to retract points lying on the tangent space, it does not have to be this way. A retraction can be a first or second order approximation to the *true retraction*, the Exp. This geometric view is critical because for some manifolds an analytical exponential map is hard to derive or expensive to compute. 

#### Geometric Interpretation of the MultiplyByJacobian 
At the first sight, to non-experts, it might seem unclear how this operation relates geometrically to the typical Riemannian operations such as projection, retraction, geodesic flow, exp, log and all that. The short answer is ***MultiplyByJacobian* maps the Euclidean gradient to the Riemannian gradient**. Quoting Sameer Agarwal:

> MultiplyByJacobian is not a generic projection operator onto the tangent space, but rather the matrix that takes the gradient vector/Jacobian matrix in the ambient coordinates to the tangent space.  What MultiplyByJacobian does is that instead of computing the Jacobian of the Plus operator at delta = 0, and then applying it to the gradient/Jacboian, it lets the user define how it is to be done, especially for high dimensional spaces. 

*MultiplyByJacobian* also ensures that the second input to the Plus operator, *delta*, is always in the tangent space of the parameter *x*. Note that all in all this method is very similar to what ***egrad2rgrad*** function of ManOpt [4]. 

With these operations well understood, it is possible to implement local parameterization operations regarding matrix manifolds such as Stiefel, Grassmann, Birkhoff and etc. I further make use of Eigen so that the matrix operations are fast and robust. One can then use these in line-search methods to perform optimization over large matrices **efficiently and easily**.

### Product Manifolds

It is also possible to operate on the cartesian product of manifolds. In Ceres terms this is somewhat equivalent to specifying a different (or the same!) parameterization for each parameter block. According to me, a *ProductParameterization* is just another *LocalParameterization* that is composed of multiple local parameterizations. Such polymorphist view enables one to create a single class that can handle a composition / mix of different local parameterization.

## Usage & Sample Code
Within the samples folder, I include a basic example that finds the closest matrix (in the Frobenius sense) on the manifold, to a given matrix in the ambient space. I guess this is called matrix denoising. Below is a sample snippet that finds the closest doubly stochastic matrix and demonstrates the use of Birkhoff Polytope (multinomial doubly stochastic matrices). 

```cpp
// doubly stochastic denoising
int main()
{
	size_t N = 10;
	MatrixXd A = (MatrixXd)(MatrixXd::Random(N, N)); // just a random matrix
	A = A.cwiseAbs(); // make it non-negative.
	MatrixXd X = DSn_rand(N); // A random matrix on Birkhoff - initial solution
	
	cout << "Given Matrix:\n" << A << endl << endl;  // print the matrix
	cout << "Initial Solution:\n" << X << endl << endl;  // print the initial solution

	// create our first order gradient problem
	MatrixDenoising *denoiseFunction = new MatrixDenoising(A);
	BirkhoffParameterization *birkhoffParameterization = new BirkhoffParameterization(N);
	GradientProblem birkhoffProblem(denoiseFunction, birkhoffParameterization);

	GradientProblemSolver::Summary summary = solveGradientProblem(birkhoffProblem, X);

	cout<<summary.FullReport()<<endl;

	// check if X is on the manifold
	cout << "Final Solution:\n" << X << endl << endl;  // print the final solution
	cout << "Is X on Manifold: "<<DSn_check(X, 0.0001) << endl;

    return 0;
}
```

We can do the same thing on the Stiefel manifold too. This time, a closed form solution (projection onto the Stiefel manifold) exists and we can compare the two solutions:

```C++
// stiefel denoising
int main()
{
	size_t N = 10, K = 10; // specific case when N=K : orthogonal group
	MatrixXd A = (MatrixXd)(MatrixXd::Random(N, K)); // just a random matrix
	MatrixXd X = Stiefel_rand(N, K); // A random matrix on Birkhoff - initial solution

	cout << "Given Matrix:\n" << A << endl << endl;  // print the matrix
	cout << "Initial Solution:\n" << X << endl << endl;  // print the initial solution

	MatrixDenoising *denoiseFunction = new MatrixDenoising(A); // create our first order gradient problem
	StiefelParameterization *stiefelParameterization = new StiefelParameterization(N, K);
	GradientProblem stiefelProblem(denoiseFunction, stiefelParameterization);

	GradientProblemSolver::Summary summary = solveGradientProblem(stiefelProblem, X);

	cout << summary.FullReport() << endl;

	// check if X is on the manifold
	cout << "Final Solution:\n" << X << endl << endl;  // print the final solution
	cout << "Is X on Manifold: " << Stiefel_check(X) << endl << endl;
		
	// print the closed form solution
	cout << "Solution by projection (closed form solution should be close to Final Solution):\n" << Stiefel_projection_SVD(A) << endl << endl;  

	return 0;
}
```

Easy peasy lemon squeezy!

## Dependencies

Only dependencies are Google's Ceres solver itself (http://ceres-solver.org/) and *Eigen* (http://eigen.tuxfamily.org/index.php?title=Main_Page).

## Compilation and Usage

The code is mostly composed of multiple *hpp* files, that one can simply import into a project.

### Authors
**Tolga Birdal**  

Enjoy it!

### References

[1] https://fzheng.me/2018/01/23/ba-demo-ceres/<br>
[2] http://www.lloydhughes.co.za/index.php/using-eigen-quaternions-and-ceres-solver/<br>
[3] https://github.com/strasdat/Sophus/blob/master/test/ceres/local_parameterization_se3.hpp<br>
[4] https://www.manopt.org<br>
