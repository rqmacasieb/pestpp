#include "gpr.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>

mt19937_64 GP::rand_engine = mt19937_64(chrono::high_resolution_clock::now().time_since_epoch().count());

GP::GP(Pest& _pest_scenario, FileManager& _file_manager, OutputFileWriter& _output_file_writer,
    PerformanceLog* _performance_log, RunManagerAbstract* _run_mgr_ptr)
    : pest_scenario(_pest_scenario),
    file_manager(_file_manager),
    output_file_writer(_output_file_writer),
    performance_log(_performance_log),
    run_mgr_ptr(_run_mgr_ptr) {
    d = 1.0;
    g = 0.0001;
    verbosity = 0;
}

void GP::initialize(const Eigen::MatrixXd& X_init, const Eigen::VectorXd& Z_init, double d_init, double g_init) {
    X = X_init;
    Z = Z_init;
    d = d_init;
    g = g_init;
    update_covariance();
}

Eigen::MatrixXd GP::distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2)
{
    Eigen::VectorXd X1_norm = X1.rowwise().squaredNorm();
    Eigen::VectorXd X2_norm = X2.rowwise().squaredNorm();

    Eigen::MatrixXd cross_term = -2.0 * X1 * X2.transpose();

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(X1.rows(), X2.rows());

    for (int i = 0; i < X1.rows(); i++) {
        for (int j = 0; j < X2.rows(); j++) {
            D(i, j) = X1_norm(i) + X2_norm(j) + cross_term(i, j);
        }
    }

    for (int i = 0; i < D.rows(); i++) {
        for (int j = 0; j < D.cols(); j++) {
            if (D(i, j) < 0) {
                D(i, j) = 0;
            }
        }
    }

    return D;
}

Eigen::MatrixXd GP::covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d)
{
    Eigen::MatrixXd D = distance(X1, X2);
    Eigen::MatrixXd K = (-D.array() / d).exp().matrix();

    return K;
}

Eigen::MatrixXd GP::covar_symm(const Eigen::MatrixXd& X, double d, double g)
{
    Eigen::MatrixXd K = covar(X, X, d);

    for (int i = 0; i < K.rows(); i++) {
        K(i, i) += g;
    }

    return K;
}


void GP::update_covariance() {
    K = covar(X, X);

    for (int i = 0; i < K.rows(); ++i) {
        K(i, i) += g;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-10 * svd.singularValues().array().abs().maxCoeff();
    Ki = svd.matrixV() *
        (svd.singularValues().array().abs() > tolerance).select(
            svd.singularValues().array().inverse(), 0).matrix().asDiagonal() *
        svd.matrixU().transpose();
}
//
//Eigen::MatrixXd GP::calculate_covariance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2) {
//    int n1 = X1.rows();
//    int n2 = X2.rows();
//    Eigen::MatrixXd cov(n1, n2);
//
//    for (int i = 0; i < n1; ++i) {
//        for (int j = 0; j < n2; ++j) {
//            double dist_squared = 0.0;
//            for (int k = 0; k < X1.cols(); ++k) {
//                double diff = X1(i, k) - X2(j, k);
//                dist_squared += diff * diff;
//            }
//
//            cov(i, j) = exp(-dist_squared / d);
//        }
//    }
//
//    return cov;
//}

void GP::update(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& Z_new, int verb) {
    int old_size = X.rows();
    int new_size = old_size + X_new.rows();

    Eigen::MatrixXd X_combined(new_size, X.cols());
    X_combined.block(0, 0, old_size, X.cols()) = X;
    X_combined.block(old_size, 0, X_new.rows(), X.cols()) = X_new;

    Eigen::VectorXd Z_combined(new_size);
    Z_combined.segment(0, old_size) = Z;
    Z_combined.segment(old_size, X_new.rows()) = Z_new;

    X = X_combined;
    Z = Z_combined;

    update_covariance();

    if (verb > 0) {
        cout << "GP updated with " << X_new.rows() << " new points. Total points: " << X.rows() << endl;
    }
}

map<string, Eigen::MatrixXd> GP::predict(const Eigen::MatrixXd& Xref, bool nonug)
{
    int nn = Xref.rows();

    double g = nonug ? sqrt(numeric_limits<double>::epsilon()) : this->nugget;

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(nn);
    Eigen::MatrixXd Sigma = covar_symm(Xref, this->theta, g);
    Eigen::MatrixXd k = covar(Xref, this->X, this->theta);

    double df = static_cast<double>(this->n);
    double phidf = this->phi / df;
    Eigen::MatrixXd ktKi = k * this->Ki;
    mean = ktKi * this->Z;
    Eigen::MatrixXd ktKik = ktKi * k.transpose();
    Sigma = phidf * (Sigma - ktKik);

    map<string, Eigen::MatrixXd> result;
    Eigen::MatrixXd mean_mat(nn, 1);
    mean_mat.col(0) = mean;
    result["mean"] = mean_mat;

    result["Sigma"] = Sigma;

    Eigen::MatrixXd df_mat(1, 1);
    df_mat(0, 0) = df;
    result["df"] = df_mat;

    Eigen::MatrixXd llik_mat(1, 1);
    llik_mat(0, 0) = this->llik;
    result["llik"] = llik_mat;

    return result;
}

map<string, Eigen::MatrixXd> GP::predict_lite(const Eigen::MatrixXd& Xref, bool nonug) 
{
    double g_value = nonug ? sqrt(numeric_limits<double>::epsilon()) : g;

    Eigen::MatrixXd k, ktKi;
    Eigen::VectorXd ktKik;
    tie(k, ktKi, ktKik) = new_predutilGP_lite(Xref.rows(), Xref);


    Eigen::VectorXd mean = ktKi * Z;
    double df = static_cast<double>(X.rows());
    double phi = (Z.transpose() * Ki * Z)(0, 0);
    double phidf = phi / df;
    Eigen::VectorXd var = Eigen::VectorXd::Constant(Xref.rows(), phidf);

    //var = phidf * (1.0 + g - ktKik)
    for (int i = 0; i < var.size(); i++) {
        var(i) *= (1.0 + g_value - ktKik(i));
    }

    double n = static_cast<double>(X.rows());
    double ldetK = log(K.determinant());
    double llik = -0.5 * (n * log(0.5 * phi) + ldetK);

    map<string, Eigen::MatrixXd> results;
    results["mean"] = mean;
    results["s2"] = var;
    results["df"] = Eigen::VectorXd::Constant(1, df);
    results["llik"] = Eigen::VectorXd::Constant(1, llik);

    return results;
}

double GP::calculate_log_likelihood()
{
    this->phi = this->Z.transpose() * this->Ki * this->Z;
    this->ldetK = log(this->K.determinant());

    this->llik = -0.5 * (this->n * log(0.5 * this->phi) + this->ldetK);

    return this->llik;
}

tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> GP::new_predutilGP_lite(int nn, const Eigen::MatrixXd& XX) {
    Eigen::MatrixXd k = calculate_covariance(XX, X);

    Eigen::MatrixXd ktKi = k * Ki;
    Eigen::VectorXd ktKik(nn);

    for (int i = 0; i < nn; i++) {
        ktKik(i) = 0.0;
        for (int j = 0; j < k.cols(); j++) {
            ktKik(i) += ktKi(i, j) * k(i, j);
        }
    }

    return make_tuple(k, ktKi, ktKik);
}

void GP::throw_gpr_error(const string& message) 
{
    string error_message = "GPR error: " + message;
    throw runtime_error(error_message);
}

void GP::message(int level, const string& message) 
{
    if (verbosity >= level) {
        cout << "GPR: " << message << endl;
    }
}

void GP::optimize_parameters(double d_param, double g_param, int verb) 
{
    if (d_param < 0 && g_param < 0) {
        double best_llik = -numeric_limits<double>::infinity();
        double best_d = 1.0;
        double best_g = 0.0001;

        for (double d_try : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            for (double g_try : {0.0001, 0.001, 0.01, 0.1}) {
                set_d(d_try);
                set_g(g_try);
                update_covariance();

                auto results = predict(X);
                double llik = results["llik"](0);

                if (llik > best_llik) {
                    best_llik = llik;
                    best_d = d_try;
                    best_g = g_try;
                }
            }
        }

        set_d(best_d);
        set_g(best_g);
        update_covariance();

        if (verb > 0) {
            cout << "Optimized parameters: d = " << best_d << ", g = " << best_g
                << ", log-likelihood = " << best_llik << endl;
        }
    }
    else if (d_param < 0) {
        double best_llik = -numeric_limits<double>::infinity();
        double best_d = 1.0;

        for (double d_try : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
            set_d(d_try);
            update_covariance();

            auto results = predict(X);
            double llik = results["llik"](0);

            if (llik > best_llik) {
                best_llik = llik;
                best_d = d_try;
            }
        }

        set_d(best_d);
        update_covariance();

        if (verb > 0) {
            cout << "Optimized length scale: d = " << best_d
                << ", log-likelihood = " << best_llik << endl;
        }
    }
    else if (g_param < 0) {
        double best_llik = -numeric_limits<double>::infinity();
        double best_g = 0.0001;

        for (double g_try : {0.0001, 0.001, 0.01, 0.1}) {
            set_g(g_try);
            update_covariance();

            auto results = predict(X);
            double llik = results["llik"](0);

            if (llik > best_llik) {
                best_llik = llik;
                best_g = g_try;
            }
        }

        set_g(best_g);
        update_covariance();

        if (verb > 0) {
            cout << "Optimized nugget: g = " << best_g
                << ", log-likelihood = " << best_llik << endl;
        }
    }
}

vector<int> GP::closest_indices(int start, const Eigen::MatrixXd& Xref, int n,
    const Eigen::MatrixXd& X_data, int close, bool need_extra) 
{
    vector<pair<double, int>> distances;

    for (int i = 0; i < n; ++i) {
        double dist_squared = 0.0;
        for (int j = 0; j < X_data.cols(); ++j) {
            double diff = Xref(0, j) - X_data(i, j);
            dist_squared += diff * diff;
        }
        distances.push_back(make_pair(sqrt(dist_squared), i));
    }

    sort(distances.begin(), distances.end());
    int n_close = (close > 0) ? close : n;
    n_close = min(n_close, n);

    vector<int> indices;
    for (int i = 0; i < n_close; ++i) {
        indices.push_back(distances[i].second);
        if (indices.size() >= start && !need_extra) 
            break;
    }

    return indices;
}

Eigen::MatrixXd GP::get_data_rect(const Eigen::MatrixXd& X_data) 
{

    int m = X_data.cols();
    Eigen::MatrixXd rect(2, m);

    for (int j = 0; j < m; ++j) {
        rect(0, j) = X_data.col(j).minCoeff();
        rect(1, j) = X_data.col(j).maxCoeff();
    }

    return rect;
}

Eigen::VectorXd GP::alc(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref, int verb) 
{
    int ncand = Xcand.rows();
    int nref = Xref.rows();
    Eigen::VectorXd scores(ncand);
    Eigen::MatrixXd k_ref_train = calculate_covariance(Xref, X);

    for (int i = 0; i < ncand; ++i) 
    {
        Eigen::MatrixXd X_new(1, Xcand.cols());
        X_new.row(0) = Xcand.row(i);

        Eigen::MatrixXd k_cand_train = calculate_covariance(X_new, X);
        Eigen::MatrixXd k_cand_ref = calculate_covariance(X_new, Xref);

        double score = 0.0;
        for (int j = 0; j < nref; ++j) {
            double var_reduction = k_cand_ref(0, j) * k_cand_ref(0, j) / (g + 1e-6);
            score += var_reduction;
        }

        scores(i) = score;
    }

    if (verb > 0) {
        cout << "ALC scores calculated for " << ncand << " candidates" << endl;
    }

    return scores;
}

Eigen::VectorXd GP::mspe(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref, int verb) {
    int ncand = Xcand.rows();
    Eigen::VectorXd scores(ncand);

    for (int i = 0; i < ncand; ++i) {
        Eigen::MatrixXd X_new(1, Xcand.cols());
        X_new.row(0) = Xcand.row(i);

        Eigen::MatrixXd k_cand_train = calculate_covariance(X_new, X);

        double var = 1.0 - k_cand_train * Ki * k_cand_train.transpose()(0, 0);
        if (var < 0) var = 0;

        scores(i) = var;
    }

    if (verb > 0) {
        cout << "MSPE scores calculated for " << ncand << " candidates" << endl;
    }

    return scores;
}

int GP::alcray_selection(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref,
    int offset, int numstart, const Eigen::MatrixXd& rect, int verb) {
    //TODO:add actual implementation
    int idx = offset % Xcand.rows();

    if (verb > 0) {
        cout << "ALCRAY selected point " << idx << " (offset: " << offset << ")" << endl;
    }

    return idx;
}

map<string, Eigen::MatrixXd> GP::run_gpr_analysis(
    const Eigen::MatrixXd& Xref,
    const Eigen::MatrixXd& X_data,
    const Eigen::VectorXd& Z_data,
    int start,
    int end,
    double d_param,
    double g_param,
    GPRMethod method,
    int close,
    int numstart,
    Eigen::MatrixXd rect,
    bool lite,
    int verb) 
{

    verbosity = verb;
    int n = X_data.rows();
    int m = X_data.cols();
    int nref = Xref.rows();

    bool need_extra = (method == GPRMethod::ALCRAY || method == GPRMethod::ALCOPT);
    vector<int> idx = closest_indices(start, Xref, n, X_data, close, need_extra);

    vector<int> cand_idx(idx.begin() + start, idx.end());
    Eigen::MatrixXd Xcand(cand_idx.size(), m);
    for (size_t i = 0; i < cand_idx.size(); ++i) 
    {
        Xcand.row(i) = X_data.row(cand_idx[i]);
    }

    vector<int> selected(end);
    for (int i = 0; i < start; ++i) 
    {
        selected[i] = idx[i];
    }

    Eigen::MatrixXd X_init(start, m);
    Eigen::VectorXd Z_init(start);
    for (int i = 0; i < start; ++i) 
    {
        X_init.row(i) = X_data.row(idx[i]);
        Z_init(i) = Z_data(idx[i]);
    }

    initialize(X_init, Z_init, d_param, g_param);

    if ((method == GPRMethod::ALCRAY || method == GPRMethod::ALCOPT) && rect.size() == 0) 
    {
        rect = get_data_rect(Xcand);
    }

    for (int i = start; i < end; ++i) 
    {
        int w = 0;  
        if (method == GPRMethod::ALCRAY) {
            int offset = (i - start + 1) % static_cast<int>(sqrt(i - start + 1));
            w = alcray_selection(Xcand, Xref, offset, numstart, rect, verb);
        }
        else if (method == GPRMethod::ALC) {
            Eigen::VectorXd scores = alc(Xcand, Xref, verb);
            w = 0;
            for (int j = 1; j < scores.size(); ++j) {
                if (scores(j) > scores(w)) {
                    w = j;
                }
            }
        }
        else if (method == GPRMethod::MSPE) {
            Eigen::VectorXd scores = mspe(Xcand, Xref, verb);
            w = 0;
            for (int j = 1; j < scores.size(); ++j) {
                if (scores(j) < scores(w)) {
                    w = j;
                }
            }
        }
        else {
            w = i - start;
        }

        selected[i] = cand_idx[w];

        Eigen::MatrixXd X_new(1, m);
        X_new.row(0) = Xcand.row(w);
        Eigen::VectorXd Z_new(1);
        Z_new(0) = Z_data(cand_idx[w]);
        update(X_new, Z_new, verb - 1);

        if (w != cand_idx.size() - 1) {
            if (method == GPRMethod::ALCRAY || method == GPRMethod::ALCOPT) {
                if (w == 0) {
                    cand_idx.erase(cand_idx.begin());
                    Xcand = Xcand.block(1, 0, Xcand.rows() - 1, Xcand.cols());
                }
                else {
                    for (size_t j = w; j < cand_idx.size() - 1; ++j) {
                        cand_idx[j] = cand_idx[j + 1];
                        Xcand.row(j) = Xcand.row(j + 1);
                    }
                    cand_idx.pop_back();
                    Xcand.conservativeResize(Xcand.rows() - 1, Xcand.cols());
                }
            }
            else {
                cand_idx[w] = cand_idx.back();
                Xcand.row(w) = Xcand.row(Xcand.rows() - 1);
                cand_idx.pop_back();
                Xcand.conservativeResize(Xcand.rows() - 1, Xcand.cols());
            }
        }
        else {
            cand_idx.pop_back();
            Xcand.conservativeResize(Xcand.rows() - 1, Xcand.cols());
        }
    }

    optimize_parameters(d_param, g_param, verb);

    map<string, Eigen::MatrixXd> results;
    if (lite) {
        results = predict_lite(Xref);
    }
    else {
        results = predict(Xref);
    }

    Eigen::VectorXd selected_vec(end);
    for (int i = 0; i < end; ++i) {
        selected_vec(i) = selected[i];
    }
    results["selected"] = selected_vec;

    Eigen::VectorXd d_posterior(1);
    d_posterior(0) = get_d();
    results["d_posterior"] = d_posterior;

    Eigen::VectorXd g_posterior(1);
    g_posterior(0) = get_g();
    results["g_posterior"] = g_posterior;

    return results;
}

void GP::iterate_to_solution() 
{
    message(1, "Starting GPR analysis");

    //TODO:implement the actual data loading
    Eigen::MatrixXd X_data, Xref;
    Eigen::VectorXd Z_data;

    int start = 6;
    int end = 50;
    double d_param = -1.0;
    double g_param = 0.0001;
    GPRMethod method = GPRMethod::ALC;
    int close = -1;
    int numstart = -1;
    bool lite = true;

    map<string, Eigen::MatrixXd> results = run_gpr_analysis(
        Xref, X_data, Z_data,
        start, end, d_param, g_param, method,
        close, numstart, Eigen::MatrixXd(), lite, verbosity);

    //TODO:implement the actual result handling
    message(1, "GPR analysis completed");
}

void GP::finalize() 
{
    message(1, "Finalizing GPR analysis");
}