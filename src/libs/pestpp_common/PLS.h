#ifndef PLS_H_
#define PLS_H_

/*
	PLS-K: Partial Least Squares + Kriging surrogate for pestpp-sm.
	Implements the PLS-K algorithm of Liu et al. (2022)
*/

#include <vector>
#include <Eigen/Dense>
#include "GPR.h"

class PLSGP
{
public:
	PLSGP() {}

	// Fit the PLS-K model.
	//   Xtrain   : N x n_input  training inputs (parameters)
	//   Ytrain   : N x n_output training outputs (observations)
	//   n_comp   : number of PLS components to extract (0 = auto)
	//   var_thresh: cumulative Y-variance explained threshold used when
	//               n_comp==0 to stop adding components (e.g. 0.99)
	//   d_in     : GP lengthscale initial value (<=0 => MLE)
	//   g_in     : GP nugget initial value (<=0 => MLE)
	//   kernel   : GP covariance kernel
	//   verb     : verbosity forwarded to GPR
	void fit(const Eigen::MatrixXd& Xtrain, const Eigen::MatrixXd& Ytrain,
		int n_comp, double var_thresh,
		double d_in, double g_in, GPKernel kernel, int verb);

	// Predict at Xpred (n_pred x n_input).
	//   pred_mean : n_pred x n_output  predictive mean
	//   pred_std  : n_pred x n_output  predictive std (propagated through Q)
	void predict(const Eigen::MatrixXd& Xpred,
		Eigen::MatrixXd& pred_mean, Eigen::MatrixXd& pred_std) const;

	int    get_n_comp()         const { return (int)gps_.size(); }
	double get_var_explained()  const { return var_explained_; }

private:
	Eigen::RowVectorXd x_mean_;   // 1 x n_input
	Eigen::RowVectorXd y_mean_;   // 1 x n_output
	Eigen::MatrixXd    R_;        // n_input  x d  — maps Xc to T without deflation
	Eigen::MatrixXd    Q_;        // n_output x d  — normalized output loadings
	vector<GP>    gps_;      // d fitted 1-D GP models  (t_k -> u_k)
	double             var_explained_ = 0.0;
};

#endif // PLS_H_
