#ifndef GPR_H_
#define GPR_H_

/*
	Gaussian Process Regression (GPR) utilities for the PEST++ suite.

	This is a C++/Eigen port of the core math in the laGPy library
	(https://github.com/, see the laGPy package).  It provides:
		- isotropic Gaussian (squared-exponential) covariance
		- a full Gaussian Process with predictive mean and variance
		- maximum-likelihood estimation of the lengthscale (and optionally the
		  nugget) using Brent's method with a gamma prior
		- a local approximate GP (laGP) using nearest-neighbour and Active
		  Learning Cohn (ALC) greedy design selection

	The implementation is intentionally self contained and depends only on
	Eigen and the C++ standard library so that it can be reused by any tool in
	the suite (it is currently used by pestpp-sm).
*/

#include <vector>
#include <string>
#include <functional>
#include <Eigen/Dense>

using namespace std;

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
	// lengthscale d and nugget g
	void build(const Eigen::MatrixXd& X_, const Eigen::VectorXd& Z_, double d_, double g_);

	// update the hyper-parameters and refresh all derived quantities
	void update_params(double d_, double g_);

	// append new design point(s) and refresh (used by local GP design)
	void update(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& Z_new);

	// log-likelihood (proportional), optionally including gamma priors on
	// the lengthscale (dab) and/or nugget (gab); pass nullptr to skip a prior
	double log_likelihood(const double* dab, const double* gab) const;

	// estimate a single hyper-parameter by maximizing the (penalized)
	// log-likelihood with Brent's method.  lengthscale=true optimizes d,
	// otherwise g.  returns the optimized value.
	double mle(bool lengthscale, double tmin, double tmax, const double* ab, int verb);

	// joint (coordinate-wise) MLE of lengthscale and nugget
	void jmle(double dmin, double dmax, double gmin, double gmax,
		const double* dab, const double* gab, int verb);

	// predictive mean and variance (Student-t predictive, "lite" diagonal
	// only version) at the rows of Xref
	void predict_lite(const Eigen::MatrixXd& Xref, Eigen::VectorXd& mean, Eigen::VectorXd& s2) const;

	int get_n() const { return (int)X.rows(); }
	int get_m() const { return (int)X.cols(); }
	double get_d() const { return d; }
	double get_g() const { return g; }
	double get_phi() const { return phi; }
	const Eigen::MatrixXd& get_X() const { return X; }
	const Eigen::MatrixXd& get_Ki() const { return Ki; }

private:
	void refresh();

	Eigen::MatrixXd X; // design (n x m)
	Eigen::VectorXd Z; // response (n)
	Eigen::MatrixXd K;  // covariance (n x n)
	Eigen::MatrixXd Ki; // inverse covariance
	Eigen::VectorXd KiZ;
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
	static double sqrt_eps();

	static Eigen::MatrixXd distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
	static Eigen::MatrixXd covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d);
	static Eigen::MatrixXd covar_symm(const Eigen::MatrixXd& X, double d, double g);

	static GPPrior darg(double d_start, const Eigen::MatrixXd& X, int samp_size = 1000);
	static GPPrior garg(double g_start, const Eigen::VectorXd& Z);

	static vector<int> closest_indices(int start, const Eigen::RowVectorXd& xref,
		const Eigen::MatrixXd& X, int close, bool sorted_flag);
	static Eigen::VectorXd alc(const GP& gp, const Eigen::MatrixXd& Xcand,
		const Eigen::RowVectorXd& xref);
	static void optimize_parameters(GP& gp, const GPPrior& dp, const GPPrior& gp_, int verb);

	static double gamma_logpdf_rate(double x, double a, double b);
	static double brent_fmin(double ax, double bx, const function<double(double)>& f, double tol);

private:
	static double gamma_p(double a, double x);
	static double quantile(vector<double> v, double q);
	static double gamma_quantile(double a, double p);
};

// high-level GPR emulator interface (used by pestpp-sm and other tools)
class GPR
{
public:
	GPR() {}

	// Full GP regression: build a GP on all of (Xtrain, Ztrain), optimize the
	// requested hyper-parameters and predict the mean/variance at the rows of
	// Xref.  d_in<=0 => estimate lengthscale; g_in<=0 => estimate nugget.
	void full_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, double d_in, double g_in, int verb,
		Eigen::VectorXd& mean, Eigen::VectorXd& s2,
		double& d_used, double& g_used);

	// Local approximate GP regression (laGP): for each reference row build a
	// local sub-design of size 'end' (starting from the 'start' nearest
	// neighbours and greedily adding points by 'method'), optimize the
	// hyper-parameters on that sub-design and predict.
	// Each reference row is independent; num_threads > 1 enables a dynamic
	// work queue over prediction rows (num_threads < 2 => serial).
	void local_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, int start, int end, const string& method,
		double d_in, double g_in, int verb,
		Eigen::VectorXd& mean, Eigen::VectorXd& s2, int num_threads = 1);
};

#endif // GPR_H_
