#include "PLS.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

using namespace std;


// ---------------------------------------------------------------------------
// PLSGP::fit
//
// NIPALS PLS2 to extract d latent score pairs (t_k, u_k), followed by a
// one-dimensional GP fitted on each pair.
//
// Notation follows Liu et al. (2022) PLS-K paper:
//   X   - mean-centred inputs   (N x n)
//   Y   - mean-centred outputs  (N x m)
//   w_k - input weight vector   (n x 1), unit norm
//   t_k - input scores          (N x 1),  t_k = X_k w_k
//   q_k - output loading vector (m x 1), unit norm
//   u_k - output scores         (N x 1),  u_k = Y_k q_k
//   p_k - input loading vector  (n x 1),  p_k = X_k^T t_k / ||t_k||^2
//   b_k - inner model coeff     (scalar),  b_k = t_k^T u_k / ||t_k||^2
//
// After d components:
//   R = W (P^T W)^{-1}  =>  T = X_c R  (direct projection, no re-deflation)
//   Prediction: Y^hat = y_mean + sum_k( GP_k(T[:,k]) ) * q_k^T
// ---------------------------------------------------------------------------
void PLSGP::fit(const Eigen::MatrixXd& Xtrain, const Eigen::MatrixXd& Ytrain,
	int n_comp, double var_thresh,
	double d_in, double g_in, GPKernel kernel, int verb)
{
	int N = (int)Xtrain.rows();
	int n = (int)Xtrain.cols();
	int m = (int)Ytrain.cols();

	if (N < 2)
		throw runtime_error("PLSGP::fit: need at least 2 training rows");

	// mean-center
	x_mean_ = Xtrain.colwise().mean();
	y_mean_ = Ytrain.colwise().mean();

	Eigen::MatrixXd Xk = Xtrain.rowwise() - x_mean_;
	Eigen::MatrixXd Yk = Ytrain.rowwise() - y_mean_;

	double total_Yvar = Yk.squaredNorm();
	if (total_Yvar <= 0.0)
		total_Yvar = 1.0;

	// upper bound on components
	int max_comp = (n_comp > 0) ? n_comp : min({ N - 1, n, m });
	if (max_comp < 1) max_comp = 1;

	Eigen::MatrixXd W(n, max_comp);
	Eigen::MatrixXd P(n, max_comp);
	Eigen::MatrixXd Qmat(m, max_comp);
	Eigen::MatrixXd T(N, max_comp);
	Eigen::MatrixXd U(N, max_comp);

	int d_actual = 0;
	double cum_var = 0.0;

	for (int k = 0; k < max_comp; k++)
	{
		// ----- NIPALS inner loop -----
		// Initialise u from first column of Yk (or the column with max norm)
		int init_col = 0;
		double best_norm = -1.0;
		for (int c = 0; c < m; c++)
		{
			double nrm = Yk.col(c).norm();
			if (nrm > best_norm) { best_norm = nrm; init_col = c; }
		}
		Eigen::VectorXd u_k = Yk.col(init_col);

		Eigen::VectorXd w_k, t_k, q_k;
		bool converged = false;

		for (int it = 0; it < 500; it++)
		{
			// w = X^T u / ||X^T u||
			Eigen::VectorXd Xtu = Xk.transpose() * u_k;
			double nw = Xtu.norm();
			if (nw < 1e-14) break;
			w_k = Xtu / nw;

			// t = X w
			t_k = Xk * w_k;

			// q = Y^T t / ||Y^T t||
			Eigen::VectorXd Ytk = Yk.transpose() * t_k;
			double nq = Ytk.norm();
			if (nq < 1e-14) break;
			q_k = Ytk / nq;

			// u_new = Y q;  check convergence
			Eigen::VectorXd u_new = Yk * q_k;
			double diff = (u_new - u_k).norm() / (u_k.norm() + 1e-14);
			u_k = u_new;

			if (diff < 1e-10) { converged = true; break; }
		}

		if (!converged && verb > 0)
			cout << "  PLS-K: component " << k + 1 << " NIPALS did not fully converge" << endl;

		double ttt = t_k.squaredNorm();
		if (ttt < 1e-14)
		{
			if (verb > 0)
				cout << "  PLS-K: component " << k + 1 << " has zero t-norm, stopping early" << endl;
			break;
		}

		// input loadings and inner model coefficient
		Eigen::VectorXd p_k = Xk.transpose() * t_k / ttt;
		double b_k = t_k.dot(u_k) / ttt;

		// store
		W.col(k) = w_k;
		P.col(k) = p_k;
		Qmat.col(k) = q_k;
		T.col(k) = t_k;
		U.col(k) = u_k;
		d_actual++;

		// variance of Y removed by this component
		double y_removed = (b_k * (t_k * q_k.transpose())).squaredNorm();
		cum_var += y_removed / total_Yvar;

		if (verb > 0)
			cout << "  PLS-K: component " << k + 1
			<< "  cumvar=" << cum_var * 100.0 << "%" << endl;

		// deflate
		Xk -= t_k * p_k.transpose();
		Yk -= b_k * (t_k * q_k.transpose());

		// auto stop if variance threshold met
		if (n_comp <= 0 && cum_var >= var_thresh)
			break;
	}

	if (d_actual == 0)
		throw runtime_error("PLSGP::fit: no PLS components could be extracted");

	var_explained_ = cum_var;

	// trim to d_actual columns
	Eigen::MatrixXd W_d = W.leftCols(d_actual);
	Eigen::MatrixXd P_d = P.leftCols(d_actual);
	Q_ = Qmat.leftCols(d_actual);
	Eigen::MatrixXd T_d = T.leftCols(d_actual);
	Eigen::MatrixXd U_d = U.leftCols(d_actual);

	// R = W (P^T W)^{-1}  — direct projection matrix
	Eigen::MatrixXd PtW = P_d.transpose() * W_d;
	R_ = W_d * PtW.inverse();

	// ----- fit one 1-D GP per component -----
	gps_.clear();
	gps_.resize(d_actual);

	for (int k = 0; k < d_actual; k++)
	{
		Eigen::VectorXd t_col = T_d.col(k);
		Eigen::VectorXd u_col = U_d.col(k);
		Eigen::MatrixXd Tk_mat = t_col;   // N x 1 design matrix

		GPPrior dp  = GPutils::darg(d_in, Tk_mat);
		GPPrior gp_ = GPutils::garg(g_in, u_col);

		gps_[k].build(Tk_mat, u_col, dp.start, gp_.start, kernel);
		GPutils::optimize_parameters(gps_[k], dp, gp_, verb);

		if (verb > 0)
			cout << "  PLS-K GP[" << k << "]: d=" << gps_[k].get_d()
			<< " g=" << gps_[k].get_g() << endl;
	}
}


// ---------------------------------------------------------------------------
// PLSGP::predict
//
// For each prediction row x_i:
//   t_i = (x_i - x_mean) R              (d-vector of input scores)
//   u^hat_k = GP_k( t_i[k] )             (predictive mean, scalar)
//   s2^hat_k = GP_k.s2( t_i[k] )         (predictive variance, scalar)
//
// Output reconstruction:
//   y^hat_i = y_mean + sum_k( u^hat_k * q_k^T )
//   Var[y^hat_ij] = sum_k( Q[j,k]^2 * s2^hat_k )
// ---------------------------------------------------------------------------
void PLSGP::predict(const Eigen::MatrixXd& Xpred,
	Eigen::MatrixXd& pred_mean, Eigen::MatrixXd& pred_std) const
{
	int n_pred = (int)Xpred.rows();
	int n_out  = (int)y_mean_.cols();
	int d      = (int)gps_.size();

	// project prediction inputs: T_pred (n_pred x d)
	Eigen::MatrixXd Xc = Xpred.rowwise() - x_mean_;
	Eigen::MatrixXd Tpred = Xc * R_;       // n_pred x d

	pred_mean.setZero(n_pred, n_out);
	Eigen::MatrixXd pred_var = Eigen::MatrixXd::Zero(n_pred, n_out);

	// broadcast y_mean to all rows
	for (int i = 0; i < n_pred; i++)
		pred_mean.row(i) = y_mean_;

	// accumulate component-wise contributions
	for (int k = 0; k < d; k++)
	{
		// 1-D GP prediction at the k-th score column
		Eigen::VectorXd t_k = Tpred.col(k);
		Eigen::MatrixXd Tk_mat = t_k;     // n_pred x 1

		Eigen::VectorXd mean_k, s2_k;
		gps_[k].predict_lite(Tk_mat, mean_k, s2_k);

		// mean_k (n_pred) * q_k^T (n_out) — outer product added row-wise
		for (int i = 0; i < n_pred; i++)
			pred_mean.row(i) += mean_k[i] * Q_.col(k).transpose();

		// propagate variance: Var[y_j] += Q[j,k]^2 * s2_k
		for (int j = 0; j < n_out; j++)
		{
			double qjk2 = Q_(j, k) * Q_(j, k);
			for (int i = 0; i < n_pred; i++)
				pred_var(i, j) += qjk2 * max(0.0, s2_k[i]);
		}
	}

	// std = sqrt(variance)
	pred_std = pred_var.cwiseSqrt();
}
