#include "Gpr.h"

#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <thread>
#include <exception>
#include <mutex>
#include <tuple>

using namespace std;

GPKernel gpr_kernel_from_string(const string& s)
{
	string c;
	for (char ch : s)
	{
		if (ch == ' ' || ch == '-' || ch == '_' || ch == '/')
			continue;
		c.push_back((char)tolower((unsigned char)ch));
	}
	if (c == "squaredexponential" || c == "gaussian" || c == "se" || c == "sqexp" || c == "rbf")
		return GPKernel::SquaredExponential;
	if (c == "exponential" || c == "exp" || c == "matern12")
		return GPKernel::Exponential;
	if (c == "matern32")
		return GPKernel::Matern32;
	if (c == "matern52")
		return GPKernel::Matern52;
	throw runtime_error("gpr_kernel_from_string(): unrecognized kernel '" + s +
		"' (expected one of: squared_exponential, exponential, matern32, matern52)");
}

string gpr_kernel_to_string(GPKernel k)
{
	switch (k)
	{
	case GPKernel::SquaredExponential: return "squared_exponential";
	case GPKernel::Exponential: return "exponential";
	case GPKernel::Matern32: return "matern32";
	case GPKernel::Matern52: return "matern52";
	}
	return "squared_exponential";
}

double GPutils::dbl_eps()
{
	return numeric_limits<double>::epsilon();
}

double GPutils::gamma_p(double a, double x)
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

double GPutils::gamma_logpdf_rate(double x, double a, double b)
	{
		if (x <= 0.0 || a <= 0.0 || b <= 0.0)
			return 0.0;
		return (a - 1.0) * log(x) - b * x + a * log(b) - lgamma(a);
	}

double GPutils::quantile(vector<double> v, double q)
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

double GPutils::brent_fmin(double ax, double bx, const function<double(double)>& f, double tol)
{
	const double golden = (3.0 - sqrt(5.0)) * 0.5;
	const double eps = sqrt(dbl_eps());
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

Eigen::MatrixXd GPutils::distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2)
	{
		Eigen::VectorXd n1 = X1.rowwise().squaredNorm();
		Eigen::VectorXd n2 = X2.rowwise().squaredNorm();
		Eigen::MatrixXd D = (-2.0 * (X1 * X2.transpose()));
		D.colwise() += n1;
		D.rowwise() += n2.transpose();
		return D.cwiseMax(0.0);
	}

Eigen::MatrixXd GPutils::covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d, GPKernel kernel)
	{
		// distance() returns SQUARED euclidean distances
		Eigen::ArrayXXd Dsq = distance(X1, X2).array();
		switch (kernel)
		{
		case GPKernel::SquaredExponential:
			// k = exp(-r^2 / d)
			return (-Dsq / d).exp().matrix();
		case GPKernel::Exponential:
		{
			// k = exp(-r / d)
			Eigen::ArrayXXd r = Dsq.max(0.0).sqrt();
			return (-r / d).exp().matrix();
		}
		case GPKernel::Matern32:
		{
			// k = (1 + sqrt(3) r / d) exp(-sqrt(3) r / d)
			const double sqrt3 = sqrt(3.0);
			Eigen::ArrayXXd a = (sqrt3 / d) * Dsq.max(0.0).sqrt();
			return ((1.0 + a) * (-a).exp()).matrix();
		}
		case GPKernel::Matern52:
		{
			// k = (1 + sqrt(5) r / d + 5 r^2 / (3 d^2)) exp(-sqrt(5) r / d)
			const double sqrt5 = sqrt(5.0);
			Eigen::ArrayXXd a = (sqrt5 / d) * Dsq.max(0.0).sqrt();
			return ((1.0 + a + (5.0 / (3.0 * d * d)) * Dsq) * (-a).exp()).matrix();
		}
		}
		return (-Dsq / d).exp().matrix();
	}

Eigen::MatrixXd GPutils::covar_symm(const Eigen::MatrixXd& X, double d, double g, GPKernel kernel)
{
	Eigen::MatrixXd K = covar(X, X, d, kernel);
	for (int i = 0; i < K.rows(); i++)
		K(i, i) += g;
	return K;
}

void GPutils::diff_covar_symm(const Eigen::MatrixXd& X, double d, GPKernel kernel, Eigen::MatrixXd& dK, Eigen::MatrixXd& d2K)
{
	int n = (int)X.rows();
	double d2 = d * d;
	Eigen::ArrayXXd Dsq = distance(X, X).array().max(0.0);
	Eigen::ArrayXXd D = Dsq.sqrt();
	dK.resize(n, n);
	d2K.resize(n, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j)
			{
				dK(i, j) = 0.0;
				d2K(i, j) = 0.0;
				continue;
			}
			double dsq = Dsq(i, j);
			double dist = D(i, j);
			double dk = 0.0, d2k = 0.0;
			switch (kernel)
			{
			case GPKernel::SquaredExponential:
			{
				double exp_term = exp(-dsq / d);
				dk = dsq * exp_term / d2;
				d2k = dk * (dsq - 2.0 * d) / d2;
				break;
			}
			case GPKernel::Exponential:
			{
				double exp_term = exp(-dist / d);
				dk = dist * exp_term / d2;
				d2k = dk * (dist - 2.0 * d) / d2;
				break;
			}
			case GPKernel::Matern32:
			{
				const double sqrt3 = sqrt(3.0);
				double sqrt3_D_d = sqrt3 * dist / d;
				double exp_term = exp(-sqrt3_D_d);
				dk = 3.0 * dsq * exp_term / (d * d * d);
				d2k = 3.0 * dsq * exp_term * (sqrt3_D_d - 3.0) / (d * d * d * d);
				break;
			}
			case GPKernel::Matern52:
			{
				const double sqrt5 = sqrt(5.0);
				double sqrt5_D_d = sqrt5 * dist / d;
				double D2_d2 = dsq / d2;
				double exp_term = exp(-sqrt5_D_d);
				dk = 5.0 * dsq * (1.0 + sqrt5_D_d) * exp_term / (3.0 * d * d * d);
				d2k = 5.0 * dsq * exp_term / (3.0 * d2 * d2)
					* (5.0 * D2_d2 - 3.0 * (1.0 + sqrt5_D_d));
				break;
			}
			}
			dK(i, j) = dk;
			d2K(i, j) = d2k;
		}
	}
}

Eigen::VectorXd GPutils::dk_dx(const Eigen::RowVectorXd& xref, const Eigen::MatrixXd& X,
	int dim, double d, GPKernel kernel)
{
	// derivative of the covariance vector k(xref, X) w.r.t. coordinate
	// 'dim' of the single reference point xref
	int n = (int)X.rows();
	Eigen::MatrixXd xr = xref;
	Eigen::ArrayXd Dsq = distance(xr, X).row(0).transpose().array().max(0.0);
	Eigen::ArrayXd D = Dsq.sqrt();
	Eigen::ArrayXd xdiff = (Eigen::ArrayXd::Constant(n, xref(dim)) - X.col(dim).array());
	Eigen::ArrayXd dk(n);
	switch (kernel)
	{
	case GPKernel::SquaredExponential:
	{
		Eigen::ArrayXd k = (-Dsq / d).exp();
		dk = -2.0 * k * xdiff / d;
		break;
	}
	case GPKernel::Exponential:
	{
		Eigen::ArrayXd k = (-D / d).exp();
		Eigen::ArrayXd Dsafe = D.max(dbl_eps());
		dk = -k * xdiff / (d * Dsafe);
		break;
	}
	case GPKernel::Matern32:
	{
		const double sqrt3 = sqrt(3.0);
		Eigen::ArrayXd a = (sqrt3 / d) * D;
		dk = -3.0 * xdiff * (-a).exp() / (d * d);
		break;
	}
	case GPKernel::Matern52:
	{
		const double sqrt5 = sqrt(5.0);
		Eigen::ArrayXd a = (sqrt5 / d) * D;
		dk = -5.0 * xdiff * (1.0 + a) * (-a).exp() / (3.0 * d * d);
		break;
	}
	default:
	{
		Eigen::ArrayXd k = (-Dsq / d).exp();
		dk = -2.0 * k * xdiff / d;
		break;
	}
	}
	return dk.matrix();
}

double GPutils::gamma_quantile(double a, double p)
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

GPPrior GPutils::darg(double d_start, const Eigen::MatrixXd& X, int samp_size)
{
	GPPrior p;
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
		dmin = sqrt(dbl_eps());
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
	if (p.min < sqrt(dbl_eps()))
		p.min = sqrt(dbl_eps());
	if (p.start < p.min)
		p.start = p.min;
	if (p.start > p.max)
		p.start = p.max;
	p.ab0 = 1.5;
	p.ab1 = gamma_quantile(p.ab0, 0.95) / (p.max > 0.0 ? p.max : 1.0);
	return p;
}

GPPrior GPutils::garg(double g_start, const Eigen::VectorXd& Z)
{
	GPPrior p;
	double mean = Z.mean();
	vector<double> r2s(Z.size());
	for (int i = 0; i < Z.size(); i++)
		r2s[i] = (Z[i] - mean) * (Z[i] - mean);

	if (g_start > 0.0)
	{
		p.start = g_start;
		p.min = sqrt(dbl_eps());
		p.max = g_start;
		p.mle = false;
		p.ab0 = 0.0;
		p.ab1 = 0.0;
		return p;
	}

	p.mle = true;
	p.start = quantile(r2s, 0.025);
	if (p.start <= 0.0)
		p.start = sqrt(dbl_eps());
	p.min = sqrt(dbl_eps());
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

vector<int> GPutils::closest_indices(int start, const Eigen::RowVectorXd& xref, const Eigen::MatrixXd& X, int close, bool sorted_flag)
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

Eigen::VectorXd GPutils::alc(const GP& gp, const Eigen::MatrixXd& Xcand, const Eigen::RowVectorXd& xref)
{
	const Eigen::MatrixXd& X = gp.get_X();
	const Eigen::MatrixXd& Ki = gp.get_Ki();
	double d = gp.get_d();
	double g = gp.get_g();
	double phi = gp.get_phi();
	GPKernel kernel = gp.get_kernel();
	int n = (int)X.rows();
	int ncand = (int)Xcand.rows();
	double df = (double)n;

	Eigen::MatrixXd xr = xref;
	Eigen::MatrixXd kref = covar(xr, X, d, kernel);
	Eigen::RowVectorXd kvec = kref.row(0);

	Eigen::MatrixXd kx = covar(X, Xcand, d, kernel);
	Eigen::MatrixXd kxy = covar(Xcand, xr, d, kernel);
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
	if (mui[c] <= dbl_eps())
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

void GPutils::optimize_parameters(GP& gp, const GPPrior& dp, const GPPrior& gp_, int verb)
{
	double dab[2] = { dp.ab0, dp.ab1 };
	double gab[2] = { gp_.ab0, gp_.ab1 };
	if (dp.mle && gp_.mle)
	{
		if (!gp.has_dK())
			gp.new_dK();
		gp.jmle(dp.min, dp.max, gp_.min, gp_.max, dab, gab, verb);
	}
	else if (dp.mle)
	{
		if (!gp.has_dK())
			gp.new_dK();
		gp.mle(true, dp.min, dp.max, dab, verb);
	}
	else if (gp_.mle)
		gp.mle(false, gp_.min, gp_.max, gab, verb);
}

void GP::build(const Eigen::MatrixXd& X_, const Eigen::VectorXd& Z_, double d_, double g_, GPKernel kernel_)
{
	X = X_;
	Z = Z_;
	d = d_;
	g = g_;
	kernel = kernel_;
	has_dK_ = false;
	dK.resize(0, 0);
	d2K.resize(0, 0);
	refresh();
}

void GP::refresh()
{
	int n = (int)X.rows();
	if (n == 0)
		throw runtime_error("GP::refresh(): empty design");
	K = GPutils::covar_symm(X, d, g, kernel);
	Eigen::LLT<Eigen::MatrixXd> llt(K);
	if (llt.info() != Eigen::Success)
		throw runtime_error("GP::refresh(): covariance not positive definite");
	Eigen::MatrixXd L = llt.matrixL();
	ldetK = 2.0 * L.diagonal().array().log().sum();
	Ki = llt.solve(Eigen::MatrixXd::Identity(n, n));
	KiZ = llt.solve(Z);
	phi = Z.dot(KiZ);
	if (phi <= 0.0)
		phi = numeric_limits<double>::min();
	if (has_dK_)
		refresh_dK();
}

void GP::refresh_dK()
{
	GPutils::diff_covar_symm(X, d, kernel, dK, d2K);
}

void GP::new_dK()
{
	if (has_dK_)
		return;
	GPutils::diff_covar_symm(X, d, kernel, dK, d2K);
	has_dK_ = true;
}

void GP::delete_dK()
{
	has_dK_ = false;
	dK.resize(0, 0);
	d2K.resize(0, 0);
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
		llik += GPutils::gamma_logpdf_rate(d, dab[0], dab[1]);
	if (g > 0.0 && gab != nullptr && gab[0] > 0.0 && gab[1] > 0.0)
		llik += GPutils::gamma_logpdf_rate(g, gab[0], gab[1]);
	return llik;
}

pair<double, double> GP::derivatives(bool lengthscale, const double* ab) const
{
	if (!has_dK_)
		throw runtime_error("GP::derivatives(): derivative matrices not initialized");
	int n = (int)X.rows();
	double th = lengthscale ? d : g;

	double dlp = 0.0, d2lp = 0.0;
	if (ab != nullptr && ab[0] > 0.0 && ab[1] > 0.0)
	{
		dlp = (ab[0] - 1.0) / th - ab[1];
		d2lp = -(ab[0] - 1.0) / (th * th);
	}

	double dllik = dlp;
	double d2llik = d2lp;

	Eigen::MatrixXd dKKi = dK * Ki;
	Eigen::MatrixXd dKKidK = dKKi * dK;
	dllik -= 0.5 * (Ki * dK).trace();
	d2llik -= 0.5 * (Ki * (d2K - dKKidK)).trace();

	Eigen::MatrixXd two = 2.0 * dKKidK - d2K;
	Eigen::VectorXd KiZtwo = two * KiZ;
	d2llik -= 0.5 * (double)n * KiZ.dot(KiZtwo) / phi;

	KiZtwo = dK * KiZ;
	double phirat = KiZ.dot(KiZtwo) / phi;
	d2llik += 0.5 * (double)n * phirat * phirat;
	dllik += 0.5 * (double)n * phirat;

	return { dllik, d2llik };
}

double GP::optimize(bool lengthscale, double tmin, double tmax, const double* ab, int verb)
{
	if (tmin >= tmax)
		return lengthscale ? d : g;

	const double* dab = lengthscale ? ab : nullptr;
	const double* gab = lengthscale ? nullptr : ab;

	double lo = tmin, hi = tmax;
	double th = lengthscale ? d : g;
	double tnew = th;
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
		tnew = GPutils::brent_fmin(lo, hi, objective, GPutils::dbl_eps());
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
		cout << "    gpr opt: " << (lengthscale ? "d=" : "g=") << tnew
			<< " llik=" << log_likelihood(dab, gab) << endl;
	return tnew;
}

double GP::mle(bool lengthscale, double tmin, double tmax, const double* ab, int verb, int* n_its)
{
	if (n_its != nullptr)
		*n_its = 0;
	if (tmin >= tmax)
		return lengthscale ? d : g;

	const double* dab = lengthscale ? ab : nullptr;
	const double* gab = lengthscale ? nullptr : ab;

	double th = lengthscale ? d : g;
	if (!lengthscale && fabs(th - tmin) < GPutils::dbl_eps())
		return th;

	int its = 0;
	bool restored_dK = false;
	double llik_init = log_likelihood(dab, gab);
	double llik_new = -numeric_limits<double>::infinity();
	bool goto_mledone = false;

	while (true)
	{
		while (true)
		{
			llik_new = -numeric_limits<double>::infinity();
			while (true)
			{
				double dllik, d2llik;
				if (lengthscale)
				{
					if (!has_dK_)
						throw runtime_error("GP::mle(): dK required for lengthscale Newton MLE");
					tie(dllik, d2llik) = derivatives(true, dab);
				}
				else
				{
					if (!has_dK_)
						throw runtime_error("GP::mle(): dK required for nugget Newton MLE");
					tie(dllik, d2llik) = derivatives(false, gab);
				}

				if (fabs(dllik) < GPutils::dbl_eps())
				{
					if (its == 0)
					{
						if (n_its != nullptr)
							*n_its = its;
						return th;
					}
					break;
				}

				double rat = dllik / d2llik;
				double adj = 1.0;
				its++;
				bool wrong_way = (dllik < 0.0 && rat < 0.0) || (dllik > 0.0 && rat > 0.0);
				double tnew = th;
				if (wrong_way)
				{
					if (has_dK_ && !restored_dK)
					{
						delete_dK();
						restored_dK = true;
					}
					th = optimize(lengthscale, tmin, tmax, ab, verb);
					goto_mledone = true;
					break;
				}
				else
					tnew = th - adj * rat;

				while ((tnew <= tmin || tnew >= tmax) && adj > GPutils::dbl_eps())
				{
					adj *= 0.5;
					tnew = th - adj * rat;
				}
				if (tnew <= tmin || tnew >= tmax)
				{
					if (has_dK_ && !restored_dK)
					{
						delete_dK();
						restored_dK = true;
					}
					th = optimize(lengthscale, tmin, tmax, ab, verb);
					goto_mledone = true;
					break;
				}

				if (lengthscale)
					update_params(tnew, g);
				else
				{
					if (has_dK_ && !restored_dK)
					{
						delete_dK();
						restored_dK = true;
					}
					update_params(d, tnew);
				}

				if (fabs(tnew - th) < GPutils::dbl_eps())
					break;
				th = tnew;

				if (its >= 100)
				{
					if (verb > 0)
						cout << "    gpr mle warning: Newton max iterations" << endl;
					if (n_its != nullptr)
						*n_its = its;
					if (restored_dK)
						new_dK();
					return lengthscale ? d : g;
				}
			}

			if (goto_mledone)
				break;

			llik_new = log_likelihood(dab, gab);
			if (llik_new < llik_init - GPutils::dbl_eps())
			{
				if (verb > 0)
					cout << "    gpr mle llik_new = " << llik_new << endl;
				llik_new = -numeric_limits<double>::infinity();
				if (has_dK_ && !restored_dK)
				{
					delete_dK();
					restored_dK = true;
				}
				th = optimize(lengthscale, tmin, tmax, ab, verb);
				goto_mledone = true;
				break;
			}
			else
				break;
		}

		if (goto_mledone)
			break;
		break;
	}

	if (!isfinite(llik_new))
		llik_new = log_likelihood(dab, gab);

	if (restored_dK)
		new_dK();

	if (verb > 0)
		cout << "    gpr mle " << (lengthscale ? "d=" : "g=") << (lengthscale ? d : g)
			<< " llik=" << llik_new << " its=" << its << endl;
	if (n_its != nullptr)
		*n_its = its;
	return lengthscale ? d : g;
}

void GP::jmle(double dmin, double dmax, double gmin, double gmax,
	const double* dab, const double* gab, int verb)
{
	for (int i = 0; i < 100; i++)
	{
		int dit = 0, git = 0;
		mle(true, dmin, dmax, dab, verb, &dit);
		mle(false, gmin, gmax, gab, verb, &git);
		if (dit <= 1 && git <= 1)
			break;
	}
}

void GP::predict_lite(const Eigen::MatrixXd& Xref, Eigen::VectorXd& mean, Eigen::VectorXd& s2,
	Eigen::MatrixXd* dmean, Eigen::MatrixXd* ds2) const
{
	int n = (int)X.rows();
	int m = (int)X.cols();
	int nref = (int)Xref.rows();
	Eigen::MatrixXd k = GPutils::covar(Xref, X, d, kernel);
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

	// optional analytic gradients of the predictive mean and variance with
	// respect to the prediction-point coordinates:
	//   d(mean)/dx_j = (dk/dx_j) . (Ki Z)
	//   d(s2)/dx_j   = -2 phidf (dk/dx_j) . (Ki k_i)
	if (dmean != nullptr || ds2 != nullptr)
	{
		if (dmean != nullptr)
			dmean->setZero(nref, m);
		if (ds2 != nullptr)
			ds2->setZero(nref, m);
		for (int i = 0; i < nref; i++)
		{
			Eigen::RowVectorXd xr_i = Xref.row(i);
			Eigen::VectorXd ktKi_i = ktKi.row(i).transpose();
			for (int j = 0; j < m; j++)
			{
				Eigen::VectorXd dkv = GPutils::dk_dx(xr_i, X, j, d, kernel);
				if (dmean != nullptr)
					(*dmean)(i, j) = dkv.dot(KiZ);
				if (ds2 != nullptr)
					(*ds2)(i, j) = -2.0 * phidf * dkv.dot(ktKi_i);
			}
		}
	}
}

void GPR::full_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
	const Eigen::MatrixXd& Xref, double d_in, double g_in, int verb,
	Eigen::VectorXd& mean, Eigen::VectorXd& s2,
	double& d_used, double& g_used,
	Eigen::MatrixXd* dmean, Eigen::MatrixXd* ds2)
{
	GPPrior dp = GPutils::darg(d_in, Xtrain);
	GPPrior gp_ = GPutils::garg(g_in, Ztrain);

	GP gp;
	gp.build(Xtrain, Ztrain, dp.start, gp_.start, kernel);
	GPutils::optimize_parameters(gp, dp, gp_, verb);
	gp.predict_lite(Xref, mean, s2, dmean, ds2);
	d_used = gp.get_d();
	g_used = gp.get_g();
}

void GPR::local_gp_predict(const Eigen::MatrixXd& Xtrain, const Eigen::VectorXd& Ztrain,
	const Eigen::MatrixXd& Xref, int start, int end, const string& method,
	double d_in, double g_in, int verb,
	Eigen::VectorXd& mean, Eigen::VectorXd& s2, int num_threads,
	Eigen::MatrixXd* dmean, Eigen::MatrixXd* ds2)
{
	int n = (int)Xtrain.rows();
	int nref = (int)Xref.rows();
	int n_dim = (int)Xtrain.cols();
	mean.resize(nref);
	s2.resize(nref);
	if (dmean != nullptr)
		dmean->setZero(nref, n_dim);
	if (ds2 != nullptr)
		ds2->setZero(nref, n_dim);

	string m = method;
	transform(m.begin(), m.end(), m.begin(), ::tolower);
	bool is_nn = (m == "nn");

	if (end > n) end = n;
	if (start >= end) start = max(1, end - 1);
	if (start < 1) start = 1;

	int close = min(1000 + end, n);

	GPPrior dp = GPutils::darg(d_in, Xtrain);
	GPPrior gp_ = GPutils::garg(g_in, Ztrain);

	auto predict_row = [&](int r)
	{
		Eigen::RowVectorXd xref = Xref.row(r);
		vector<int> idx = GPutils::closest_indices(start, xref, Xtrain, close, is_nn);

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
			gp.build(Xl, Zl, dp.start, gp_.start, kernel);
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
			gp.build(Xinit, Zinit, dp.start, gp_.start, kernel);

			for (int i = start; i < end && !cand.empty(); i++)
			{
				Eigen::MatrixXd Xcand(cand.size(), Xtrain.cols());
				for (size_t c = 0; c < cand.size(); c++)
					Xcand.row(c) = Xtrain.row(cand[c]);
				Eigen::VectorXd scores = GPutils::alc(gp, Xcand, xref);
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

		GPutils::optimize_parameters(gp, dp, gp_, verb - 1);

		Eigen::VectorXd m1, s1;
		Eigen::MatrixXd xr = xref;
		Eigen::MatrixXd dm1, ds1;
		gp.predict_lite(xr, m1, s1,
			dmean != nullptr ? &dm1 : nullptr,
			ds2 != nullptr ? &ds1 : nullptr);
		mean[r] = m1[0];
		s2[r] = s1[0];
		if (dmean != nullptr)
			dmean->row(r) = dm1.row(0);
		if (ds2 != nullptr)
			ds2->row(r) = ds1.row(0);
	};

	if (num_threads < 2 || nref < 2)
	{
		for (int r = 0; r < nref; r++)
			predict_row(r);
		return;
	}

	Eigen::setNbThreads(1);
	int nthreads = min(num_threads, nref);
	vector<thread> threads;
	vector<exception_ptr> eptrs(nthreads, nullptr);
	int next_row = 0;
	mutex next_lock;

	auto queue_work = [&](int tid)
	{
		try
		{
			while (true)
			{
				int r;
				{
					lock_guard<mutex> guard(next_lock);
					if (next_row >= nref)
						break;
					r = next_row;
					next_row++;
				}
				predict_row(r);
			}
		}
		catch (...)
		{
			eptrs[tid] = current_exception();
		}
	};

	for (int t = 0; t < nthreads; t++)
		threads.push_back(thread(queue_work, t));
	for (int t = 0; t < nthreads; t++)
		threads[t].join();
	for (int t = 0; t < nthreads; t++)
		if (eptrs[t])
			rethrow_exception(eptrs[t]);
}
