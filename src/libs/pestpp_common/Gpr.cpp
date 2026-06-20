#include "Gpr.h"

#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

using namespace std;

namespace gpr
{
	static const double DBL_EPS = numeric_limits<double>::epsilon();
	static const double SQRT_EPS = sqrt(numeric_limits<double>::epsilon());

	// lower regularized incomplete gamma P(a,x), Numerical-Recipes style
	static double gamma_p(double a, double x)
	{
		if (x <= 0.0)
			return 0.0;
		double gln = lgamma(a);
		if (x < a + 1.0)
		{
			double ap = a;
			double sum = 1.0 / a;
			double del = sum;
			for (int n = 0; n < 1000; n++)
			{
				ap += 1.0;
				del *= x / ap;
				sum += del;
				if (fabs(del) < fabs(sum) * 1.0e-15)
					break;
			}
			return sum * exp(-x + a * log(x) - gln);
		}
		else
		{
			double tiny = 1.0e-300;
			double b = x + 1.0 - a;
			double c = 1.0 / tiny;
			double dd = 1.0 / b;
			double h = dd;
			for (int i = 1; i <= 1000; i++)
			{
				double an = -i * (i - a);
				b += 2.0;
				dd = an * dd + b;
				if (fabs(dd) < tiny) dd = tiny;
				c = b + an / c;
				if (fabs(c) < tiny) c = tiny;
				dd = 1.0 / dd;
				double del = dd * c;
				h *= del;
				if (fabs(del - 1.0) < 1.0e-15)
					break;
			}
			double q = exp(-x + a * log(x) - gln) * h;
			return 1.0 - q;
		}
	}

	// log pdf of a gamma distribution with shape a and rate b (scale 1/b)
	static double gamma_logpdf_rate(double x, double a, double b)
	{
		if (x <= 0.0 || a <= 0.0 || b <= 0.0)
			return 0.0;
		return (a - 1.0) * log(x) - b * x + a * log(b) - lgamma(a);
	}

	static double quantile(vector<double> v, double q)
	{
		if (v.empty())
			return 0.0;
		sort(v.begin(), v.end());
		if (v.size() == 1)
			return v[0];
		double pos = q * (double)(v.size() - 1);
		int lo = (int)floor(pos);
		int hi = (int)ceil(pos);
		double frac = pos - (double)lo;
		return v[lo] * (1.0 - frac) + v[hi] * frac;
	}

	// Brent's method (port of laGPy.utils.brent_fmin) minimizing a functor
	template <typename F>
	static double brent_fmin(double ax, double bx, F&& f, double tol)
	{
		const double golden = (3.0 - sqrt(5.0)) * 0.5;
		const double eps = SQRT_EPS;
		double tol3 = tol / 3.0;

		double a = ax, b = bx;
		double x = a + golden * (b - a);
		double v = x, w = x;
		double fx = f(x);
		double fv = fx, fw = fx;
		double dstep = 0.0, e = 0.0;

		for (int iter = 0; iter < 1000; iter++)
		{
			double xm = (a + b) * 0.5;
			double tol1 = eps * fabs(x) + tol3;
			double t2 = tol1 * 2.0;
			if (fabs(x - xm) <= t2 - (b - a) * 0.5)
				break;

			bool use_golden = true;
			if (fabs(e) > tol1)
			{
				double r = (x - w) * (fx - fv);
				double q = (x - v) * (fx - fw);
				double p = (x - v) * q - (x - w) * r;
				q = (q - r) * 2.0;
				if (q > 0.0)
					p = -p;
				else
					q = -q;
				r = e;
				e = dstep;
				if (!((fabs(p) >= fabs(q * 0.5 * r)) || (p <= q * (a - x)) || (p >= q * (b - x))))
				{
					use_golden = false;
					dstep = p / q;
					double u = x + dstep;
					if ((u - a < t2) || (b - u < t2))
						dstep = copysign(tol1, xm - x);
				}
			}
			if (use_golden)
			{
				e = (x < xm) ? (b - x) : (a - x);
				dstep = golden * e;
			}

			double u = x + ((fabs(dstep) < tol1) ? copysign(tol1, dstep) : dstep);
			double fu = f(u);

			if (fu <= fx)
			{
				if (u < x) b = x; else a = x;
				v = w; w = x; x = u;
				fv = fw; fw = fx; fx = fu;
			}
			else
			{
				if (u < x) a = u; else b = u;
				if (fu <= fw || w == x)
				{
					v = w; w = u;
					fv = fw; fw = fu;
				}
				else if (fu <= fv || v == x || v == w)
				{
					v = u; fv = fu;
				}
			}
		}
		return x;
	}

	Eigen::MatrixXd distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2)
	{
		Eigen::VectorXd n1 = X1.rowwise().squaredNorm();
		Eigen::VectorXd n2 = X2.rowwise().squaredNorm();
		Eigen::MatrixXd D = (-2.0 * (X1 * X2.transpose()));
		D.colwise() += n1;
		D.rowwise() += n2.transpose();
		return D.cwiseMax(0.0);
	}

	Eigen::MatrixXd covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d)
	{
		Eigen::MatrixXd D = distance(X1, X2);
		return (-D / d).array().exp().matrix();
	}

	Eigen::MatrixXd covar_symm(const Eigen::MatrixXd& X, double d, double g)
	{
		Eigen::MatrixXd K = covar(X, X, d);
		for (int i = 0; i < K.rows(); i++)
			K(i, i) += g;
		return K;
	}

	double gamma_quantile(double a, double p)
	{
		if (p <= 0.0)
			return 0.0;
		if (p >= 1.0)
			return a + 10.0 * sqrt(a) + 100.0;
		double lo = 0.0;
		double hi = a + 10.0 * sqrt(a) + 100.0;
		for (int k = 0; k < 100 && gamma_p(a, hi) < p; k++)
			hi *= 2.0;
		double mid = 0.5 * (lo + hi);
		for (int it = 0; it < 200; it++)
		{
			mid = 0.5 * (lo + hi);
			double val = gamma_p(a, mid);
			if (fabs(val - p) < 1.0e-12)
				break;
			if (val < p)
				lo = mid;
			else
				hi = mid;
		}
		return mid;
	}

	Prior darg(double d_start, const Eigen::MatrixXd& X, int samp_size)
	{
		Prior p;
		Eigen::MatrixXd Xs = X;
		if (X.rows() > samp_size)
			Xs = X.topRows(samp_size);
		Eigen::MatrixXd D = distance(Xs, Xs);
		vector<double> dvals;
		double dmin = numeric_limits<double>::max();
		double dmax = 0.0;
		for (int i = 0; i < D.rows(); i++)
			for (int j = i + 1; j < D.cols(); j++)
			{
				double val = D(i, j);
				if (val > 0.0)
				{
					dvals.push_back(val);
					dmin = min(dmin, val);
					dmax = max(dmax, val);
				}
			}
		if (dvals.empty())
		{
			dmin = SQRT_EPS;
			dmax = 1.0;
		}

		if (d_start > 0.0)
		{
			p.start = d_start;
			p.min = d_start;
			p.max = d_start;
			p.mle = false;
			p.ab0 = 0.0;
			p.ab1 = 0.0;
			return p;
		}

		p.mle = true;
		p.start = quantile(dvals, 0.1);
		p.max = dmax;
		p.min = dmin / 2.0;
		if (p.min < SQRT_EPS)
			p.min = SQRT_EPS;
		if (p.start < p.min)
			p.start = p.min;
		if (p.start > p.max)
			p.start = p.max;
		p.ab0 = 1.5;
		p.ab1 = gamma_quantile(p.ab0, 0.95) / (p.max > 0.0 ? p.max : 1.0);
		return p;
	}

	Prior garg(double g_start, const Eigen::VectorXd& Z)
	{
		Prior p;
		double mean = Z.mean();
		vector<double> r2s(Z.size());
		for (int i = 0; i < Z.size(); i++)
			r2s[i] = (Z[i] - mean) * (Z[i] - mean);

		if (g_start > 0.0)
		{
			p.start = g_start;
			p.min = SQRT_EPS;
			p.max = g_start;
			p.mle = false;
			p.ab0 = 0.0;
			p.ab1 = 0.0;
			return p;
		}

		p.mle = true;
		p.start = quantile(r2s, 0.025);
		if (p.start <= 0.0)
			p.start = SQRT_EPS;
		p.min = SQRT_EPS;
		p.max = *max_element(r2s.begin(), r2s.end());
		if (p.max <= p.min)
			p.max = p.min * 100.0 + 1.0;
		p.ab0 = 1.5;
		double s2max = mean;
		double rmean = 0.0;
		for (double v : r2s) rmean += v;
		rmean /= (double)r2s.size();
		s2max = rmean > 0.0 ? rmean : 1.0;
		p.ab1 = gamma_quantile(p.ab0, 0.95) / s2max;
		return p;
	}

	void GP::build(const Eigen::MatrixXd& X_, const Eigen::VectorXd& Z_, double d_, double g_)
	{
		X = X_;
		Z = Z_;
		d = d_;
		g = g_;
		refresh();
	}

	void GP::refresh()
	{
		int n = (int)X.rows();
		if (n == 0)
			throw runtime_error("gpr::GP::refresh(): empty design");
		K = covar_symm(X, d, g);
		Eigen::LLT<Eigen::MatrixXd> llt(K);
		if (llt.info() != Eigen::Success)
			throw runtime_error("gpr::GP::refresh(): covariance not positive definite");
		Eigen::MatrixXd L = llt.matrixL();
		ldetK = 2.0 * L.diagonal().array().log().sum();
		Ki = llt.solve(Eigen::MatrixXd::Identity(n, n));
		KiZ = llt.solve(Z);
		phi = Z.dot(KiZ);
		if (phi <= 0.0)
			phi = numeric_limits<double>::min();
	}

	void GP::update_params(double d_, double g_)
	{
		d = d_;
		g = g_;
		refresh();
	}

	void GP::update(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& Z_new)
	{
		Eigen::MatrixXd Xn(X.rows() + X_new.rows(), X.cols());
		Xn << X, X_new;
		Eigen::VectorXd Zn(Z.size() + Z_new.size());
		Zn << Z, Z_new;
		X = Xn;
		Z = Zn;
		refresh();
	}

	double GP::log_likelihood(const double* dab, const double* gab) const
	{
		int n = (int)X.rows();
		double llik = -0.5 * ((double)n * log(0.5 * phi) + ldetK);
		if (d > 0.0 && dab != nullptr && dab[0] > 0.0 && dab[1] > 0.0)
			llik += gamma_logpdf_rate(d, dab[0], dab[1]);
		if (g > 0.0 && gab != nullptr && gab[0] > 0.0 && gab[1] > 0.0)
			llik += gamma_logpdf_rate(g, gab[0], gab[1]);
		return llik;
	}

	double GP::mle(bool lengthscale, double tmin, double tmax, const double* ab, int verb)
	{
		if (tmin >= tmax)
			return lengthscale ? d : g;

		const double* dab = lengthscale ? ab : nullptr;
		const double* gab = lengthscale ? nullptr : ab;

		double lo = tmin, hi = tmax;
		double tnew = lengthscale ? d : g;
		for (int outer = 0; outer < 100; outer++)
		{
			auto objective = [&](double xval) -> double {
				try
				{
					if (lengthscale)
						update_params(xval, g);
					else
						update_params(d, xval);
					return -log_likelihood(dab, gab);
				}
				catch (...)
				{
					return numeric_limits<double>::max();
				}
			};
			tnew = brent_fmin(lo, hi, objective, DBL_EPS);
			if (lo < tnew && tnew < hi)
				break;
			if (tnew <= lo)
				lo *= 2.0;
			else
				hi *= 0.5;
			if (lo >= hi)
				break;
		}
		if (lengthscale)
			update_params(tnew, g);
		else
			update_params(d, tnew);
		if (verb > 0)
			cout << "    gpr mle " << (lengthscale ? "d=" : "g=") << tnew
			<< " llik=" << log_likelihood(dab, gab) << endl;
		return tnew;
	}

	void GP::jmle(double dmin, double dmax, double gmin, double gmax,
		const double* dab, const double* gab, int verb)
	{
		for (int i = 0; i < 100; i++)
		{
			double d_old = d, g_old = g;
			mle(true, dmin, dmax, dab, verb);
			mle(false, gmin, gmax, gab, verb);
			if (fabs(d - d_old) < SQRT_EPS && fabs(g - g_old) < SQRT_EPS)
				break;
		}
	}

	void GP::predict_lite(const Eigen::MatrixXd& Xref, Eigen::VectorXd& mean, Eigen::VectorXd& s2) const
	{
		int n = (int)X.rows();
		int nref = (int)Xref.rows();
		Eigen::MatrixXd k = covar(Xref, X, d);
		Eigen::MatrixXd ktKi = k * Ki;
		Eigen::VectorXd ktKik = (ktKi.array() * k.array()).rowwise().sum();
		mean = ktKi * Z;
		double df = (double)n;
		double phidf = phi / df;
		s2.resize(nref);
		for (int i = 0; i < nref; i++)
		{
			double var = phidf * (1.0 + g - ktKik[i]);
			if (var < 0.0)
				var = 0.0;
			s2[i] = var;
		}
	}

	vector<int> closest_indices(int start, const Eigen::RowVectorXd& xref,
		const Eigen::MatrixXd& X, int close, bool sorted_flag)
	{
		int n = (int)X.rows();
		Eigen::MatrixXd xr = xref;
		Eigen::MatrixXd D = distance(X, xr);
		vector<int> idx(n);
		iota(idx.begin(), idx.end(), 0);

		int keep = min(close, n);
		if (n > close)
		{
			partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
				[&](int a, int b) { return D(a, 0) < D(b, 0); });
			idx.resize(keep);
		}

		if (sorted_flag)
		{
			sort(idx.begin(), idx.end(), [&](int a, int b) { return D(a, 0) < D(b, 0); });
		}
		else if (start < (int)idx.size())
		{
			nth_element(idx.begin(), idx.begin() + start, idx.end(),
				[&](int a, int b) { return D(a, 0) < D(b, 0); });
		}
		return idx;
	}

	Eigen::VectorXd alc(const GP& gp, const Eigen::MatrixXd& Xcand, const Eigen::RowVectorXd& xref)
	{
		const Eigen::MatrixXd& X = gp.get_X();
		const Eigen::MatrixXd& Ki = gp.get_Ki();
		double d = gp.get_d();
		double g = gp.get_g();
		double phi = gp.get_phi();
		int n = (int)X.rows();
		int ncand = (int)Xcand.rows();
		double df = (double)n;

		Eigen::MatrixXd xr = xref;
		Eigen::MatrixXd kref = covar(xr, X, d);
		Eigen::RowVectorXd kvec = kref.row(0);

		Eigen::MatrixXd kx = covar(X, Xcand, d);
		Eigen::MatrixXd kxy = covar(Xcand, xr, d);
		Eigen::MatrixXd gvec = kx.transpose() * Ki;
		Eigen::VectorXd mui(ncand);
		for (int c = 0; c < ncand; c++)
		{
			double dot = kx.col(c).dot(gvec.row(c).transpose());
			mui[c] = 1.0 + g - dot;
		}

		Eigen::VectorXd scores = Eigen::VectorXd::Constant(ncand, -numeric_limits<double>::infinity());
		double dfrat = df / (df - 2.0);
		for (int c = 0; c < ncand; c++)
		{
			if (mui[c] <= DBL_EPS)
				continue;
			Eigen::RowVectorXd gc = -gvec.row(c) / mui[c];
			double dotp = gc.dot(kvec);
			double kxyc = kxy(c, 0);
			double ktKikx = dotp * dotp * mui[c] + 2.0 * dotp * kxyc + (kxyc * kxyc) / mui[c];
			double zphi = phi * ktKikx;
			double ts2 = zphi / df;
			scores[c] = ts2 * dfrat;
		}
		return scores;
	}

	static void optimize_parameters(GP& gp, const Prior& dp, const Prior& gp_, int verb)
	{
		double dab[2] = { dp.ab0, dp.ab1 };
		double gab[2] = { gp_.ab0, gp_.ab1 };
		if (dp.mle && gp_.mle)
			gp.jmle(dp.min, dp.max, gp_.min, gp_.max, dab, gab, verb);
		else if (dp.mle)
			gp.mle(true, dp.min, dp.max, dab, verb);
		else if (gp_.mle)
			gp.mle(false, gp_.min, gp_.max, gab, verb);
	}

	void full_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, double d_in, double g_in, int verb,
		Eigen::VectorXd& mean, Eigen::VectorXd& s2,
		double& d_used, double& g_used)
	{
		Prior dp = darg(d_in, Xtrain);
		Prior gp_ = garg(g_in, Ztrain);

		GP gp;
		gp.build(Xtrain, Ztrain, dp.start, gp_.start);
		optimize_parameters(gp, dp, gp_, verb);
		gp.predict_lite(Xref, mean, s2);
		d_used = gp.get_d();
		g_used = gp.get_g();
	}

	void local_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
		const Eigen::MatrixXd& Xref, int start, int end, const string& method,
		double d_in, double g_in, int verb,
		Eigen::VectorXd& mean, Eigen::VectorXd& s2)
	{
		int n = (int)Xtrain.rows();
		int nref = (int)Xref.rows();
		mean.resize(nref);
		s2.resize(nref);

		string m = method;
		transform(m.begin(), m.end(), m.begin(), ::tolower);
		bool is_nn = (m == "nn");

		if (end > n) end = n;
		if (start >= end) start = max(1, end - 1);
		if (start < 1) start = 1;

		int close = min(1000 + end, n);

		Prior dp = darg(d_in, Xtrain);
		Prior gp_ = garg(g_in, Ztrain);

		for (int r = 0; r < nref; r++)
		{
			Eigen::RowVectorXd xref = Xref.row(r);
			vector<int> idx = closest_indices(start, xref, Xtrain, close, is_nn);

			GP gp;
			if (is_nn)
			{
				vector<int> sel(idx.begin(), idx.begin() + min(end, (int)idx.size()));
				Eigen::MatrixXd Xl(sel.size(), Xtrain.cols());
				Eigen::VectorXd Zl(sel.size());
				for (size_t i = 0; i < sel.size(); i++)
				{
					Xl.row(i) = Xtrain.row(sel[i]);
					Zl[i] = Ztrain[sel[i]];
				}
				gp.build(Xl, Zl, dp.start, gp_.start);
			}
			else
			{
				vector<int> cand(idx.begin() + start, idx.end());
				Eigen::MatrixXd Xinit(start, Xtrain.cols());
				Eigen::VectorXd Zinit(start);
				for (int i = 0; i < start; i++)
				{
					Xinit.row(i) = Xtrain.row(idx[i]);
					Zinit[i] = Ztrain[idx[i]];
				}
				gp.build(Xinit, Zinit, dp.start, gp_.start);

				for (int i = start; i < end && !cand.empty(); i++)
				{
					Eigen::MatrixXd Xcand(cand.size(), Xtrain.cols());
					for (size_t c = 0; c < cand.size(); c++)
						Xcand.row(c) = Xtrain.row(cand[c]);
					Eigen::VectorXd scores = alc(gp, Xcand, xref);
					int w = 0;
					double best = scores[0];
					for (int c = 1; c < scores.size(); c++)
						if (scores[c] > best) { best = scores[c]; w = c; }

					Eigen::MatrixXd Xnew = Xtrain.row(cand[w]);
					Eigen::VectorXd Znew(1);
					Znew[0] = Ztrain[cand[w]];
					gp.update(Xnew, Znew);

					cand[w] = cand.back();
					cand.pop_back();
				}
			}

			optimize_parameters(gp, dp, gp_, verb - 1);

			Eigen::VectorXd m1, s1;
			Eigen::MatrixXd xr = xref;
			gp.predict_lite(xr, m1, s1);
			mean[r] = m1[0];
			s2[r] = s1[0];
		}
	}
}
