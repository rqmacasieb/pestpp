#ifndef GPR_H_
#define GPR_H_

/*
	Gaussian Process Regression (GPR) utilities for the PEST++ suite.

	This is a C++/Eigen port of the core math in the laGPy library
	(https://github.com/, see the laGPy package).  It provides:
		- selectable isotropic covariance kernels: squared-exponential
		  (default), exponential, Matern 3/2 and Matern 5/2
		- a full Gaussian Process with predictive mean and variance
		- maximum-likelihood estimation of the lengthscale (and optionally the
		  nugget) using Newton's method with Brent fallback, matching laGPy
		- a local approximate GP (laGP) using nearest-neighbour and Active
		  Learning Cohn (ALC) greedy design selection
		- optional analytic gradients of the predictive mean and variance with
		  respect to the prediction-point coordinates

	The implementation is intentionally self contained and depends only on
	Eigen and the C++ standard library so that it can be reused by any tool in
	the suite (it is currently used by pestpp-sm).
*/

#include <vector>
#include <string>
#include <functional>
#include <utility>
#include <Eigen/Dense>

using namespace std;

enum class GPKernel
{
	SquaredExponential,
	Exponential,
	Matern32,
	Matern52
};

GPKernel gpr_kernel_from_string(const string& s);
string gpr_kernel_to_string(GPKernel k);

// prior/initialization information for a GP hyper-parameter
struct GPPrior
{
	double start = 1.0; // initial value
	double min = 1.0e-6; // lower bound for optimization
	double max = 1.0e+6; // upper bound for optimization
	bool mle = false;    // whether to estimate by maximum likelihood
	double ab0 = 0.0;    // gamma prior shape (0 => no prior)
	double ab1 = 0.0;    // gamma prior rate  (0 => no prior)
};

// single Gaussian Process model (internal building block)
class GP
{
public:
	GP() {}

	// (re)build the GP from a design X (n x m) and response Z (n) using
	// lengthscale d and nugget g and the requested covariance kernel
	void build(const Eigen::MatrixXd& X_, const Eigen::VectorXd& Z_, double d_, double g_,
		GPKernel kernel_ = GPKernel::SquaredExponential);

	// update the hyper-parameters and refresh all derived quantities
	void update_params(double d_, double g_);

	// append new design point(s) and refresh (used by local GP design)
	void update(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& Z_new);

	// log-likelihood (proportional), optionally including gamma priors on
	// the lengthscale (dab) and/or nugget (gab); pass nullptr to skip a prior
	double log_likelihood(const double* dab, const double* gab) const;

	// estimate a single hyper-parameter by maximizing the (penalized)
	// log-likelihood with Newton's method (Brent fallback)
	// lengthscale=true optimizes d, otherwise g.  returns the optimized value;
	// if n_its is non-null it receives the number of Newton/Brent iterations.
	double mle(bool lengthscale, double tmin, double tmax, const double* ab, int verb, int* n_its = nullptr);

	// joint (coordinate-wise) MLE of lengthscale and nugget
	void jmle(double dmin, double dmax, double gmin, double gmax, const double* dab, const double* gab, int verb);

	// allocate / clear covariance derivative matrices used by Newton MLE
	void new_dK();
	void delete_dK();
	bool has_dK() const { return has_dK_; }

	// predictive mean and variance (Student-t predictive, "lite" diagonal
	// only version) at the rows of Xref.  if dmean and/or ds2 are non-null they
	// are filled (nref x m) with the analytic gradients of the predictive mean
	// and variance with respect to the prediction-point coordinates.
	void predict_lite(const Eigen::MatrixXd& Xref, Eigen::VectorXd& mean, Eigen::VectorXd& s2,
		Eigen::MatrixXd* dmean = nullptr, Eigen::MatrixXd* ds2 = nullptr) const;

	int get_n() const { return (int)X.rows(); }
	int get_m() const { return (int)X.cols(); }
	double get_d() const { return d; }
	double get_g() const { return g; }
	double get_phi() const { return phi; }
	GPKernel get_kernel() const { return kernel; }
	const Eigen::MatrixXd& get_X() const { return X; }
	const Eigen::MatrixXd& get_Ki() const { return Ki; }

private:
	void refresh();
	void refresh_dK();
	std::pair<double, double> derivatives(bool lengthscale, const double* ab) const;
	double optimize(bool lengthscale, double tmin, double tmax, const double* ab, int verb);

	GPKernel kernel = GPKernel::SquaredExponential;
	Eigen::MatrixXd X; // design (n x m)
	Eigen::VectorXd Z; // response (n)
	Eigen::MatrixXd K;  // covariance (n x n)
	Eigen::MatrixXd Ki; // inverse covariance
	Eigen::VectorXd KiZ;
	Eigen::MatrixXd dK;  // dK/dd (optional, for Newton MLE)
	Eigen::MatrixXd d2K; // d2K/dd2 (optional)
	bool has_dK_ = false;
	double ldetK = 0.0;
	double phi = 0.0;
	double d = 1.0;
	double g = 1.0e-4;
};

// internal math/helpers for GP and GPR (not used outside this module)
class GPutils
{
public:
	static double dbl_eps();

	static Eigen::MatrixXd distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
	static Eigen::MatrixXd covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d, GPKernel kernel = GPKernel::SquaredExponential);
	static Eigen::MatrixXd covar_symm(const Eigen::MatrixXd& X, double d, double g, GPKernel kernel = GPKernel::SquaredExponential);

	// first and second derivatives of the symmetric covariance matrix w.r.t. the lengthscale d
	static void diff_covar_symm(const Eigen::MatrixXd& X, double d, GPKernel kernel, Eigen::MatrixXd& dK, Eigen::MatrixXd& d2K);

	// derivative of the covariance vector k(xref, X) with respect to input
	// dimension 'dim' of the (single) reference point xref; returns a length-n vector
	static Eigen::VectorXd dk_dx(const Eigen::RowVectorXd& xref, const Eigen::MatrixXd& X, int dim, double d, GPKernel kernel);

	static GPPrior darg(double d_start, const Eigen::MatrixXd& X, int samp_size = 1000);
	static GPPrior garg(double g_start, const Eigen::VectorXd& Z);

	static vector<int> closest_indices(int start, const Eigen::RowVectorXd& xref, const Eigen::MatrixXd& X, int close, bool sorted_flag);
	static Eigen::VectorXd alc(const GP& gp, const Eigen::MatrixXd& Xcand, const Eigen::RowVectorXd& xref);
	static void optimize_parameters(GP& gp, const GPPrior& dp, const GPPrior& gp_, int verb);

	static double gamma_logpdf_rate(double x, double a, double b);
	static double brent_fmin(double ax, double bx, const function<double(double)>& f, double tol);

private:
	static double gamma_p(double a, double x);
	static double quantile(vector<double> v, double q);
	static double gamma_quantile(double a, double p);
};

class GPR
{
public:
	GPR() {}
	explicit GPR(GPKernel _kernel) : kernel(_kernel) {}

	void set_kernel(GPKernel _kernel) { kernel = _kernel; }
	GPKernel get_kernel() const { return kernel; }

	// Full GP regression: build a GP on all of (Xtrain, Ztrain), optimize the
	// requested hyper-parameters and predict the mean/variance at the rows of
	// Xref.  d_in<=0 => estimate lengthscale; g_in<=0 => estimate nugget.
	// If dmean and/or ds2 are non-null they are filled (nref x m) with the
	// analytic gradients of the prediction with respect to the input coordinates.
	void full_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, double d_in, double g_in, int verb, Eigen::VectorXd& mean, Eigen::VectorXd& s2,
		double& d_used, double& g_used, Eigen::MatrixXd* dmean = nullptr, Eigen::MatrixXd* ds2 = nullptr);

	// Local approximate GP regression (laGP): for each reference row build a
	// local sub-design of size 'end' (starting from the 'start' nearest
	// neighbours and greedily adding points by 'method'), optimize the
	// hyper-parameters on that sub-design and predict.
	// Each reference row is independent; num_threads > 1 enables a dynamic
	// work queue over prediction rows (num_threads < 2 => serial).
	// If dmean and/or ds2 are non-null they are filled (nref x m) with the
	// analytic gradients of the prediction with respect to the input coordinates.
	void local_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, int start, int end, const string& method, double d_in, double g_in, int verb,
		Eigen::VectorXd& mean, Eigen::VectorXd& s2, int num_threads = 1, Eigen::MatrixXd* dmean = nullptr, Eigen::MatrixXd* ds2 = nullptr);

private:
	GPKernel kernel = GPKernel::SquaredExponential;
};

#endif // GPR_H_
