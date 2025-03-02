#ifndef GPR_H_
#define GPR_H_

#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <random>
#include "Pest.h"
#include "FileManager.h"
#include "OutputFileWriter.h"
#include "RunManagerAbstract.h"
#include "PerformanceLog.h"

using namespace std;

// Enum for method selection
enum class GPRMethod {
    ALC,
    ALCRAY,
    ALCOPT,
    MSPE,
    NN
};

class GP {
public:
    static mt19937_64 rand_engine;


    GP(Pest& _pest_scenario, FileManager& _file_manager, OutputFileWriter& _output_file_writer,
        PerformanceLog* _performance_log = nullptr, RunManagerAbstract* _run_mgr_ptr = nullptr);

    void initialize(const Eigen::MatrixXd& X, const Eigen::VectorXd& Z, double d, double g);
    void update_covariance();
    void update(const Eigen::MatrixXd& X_new, const Eigen::VectorXd& Z_new, int verb = 0);

    Eigen::MatrixXd distance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
    Eigen::MatrixXd covar(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2, double d);
    Eigen::MatrixXd covar_symm(const Eigen::MatrixXd& X, double d, double g);

    map<string, Eigen::MatrixXd> predict(const Eigen::MatrixXd& Xref, bool nonug = false);
    map<string, Eigen::MatrixXd> predict_lite(const Eigen::MatrixXd& Xref, bool nonug = false);

    double calculate_log_likelihood();

    void iterate_to_solution();
    void finalize();

    map<string, Eigen::MatrixXd> run_gpr_analysis(
        const Eigen::MatrixXd& Xref,
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& Z,
        int start = 6,
        int end = 50,
        double d = -1.0,
        double g = 1.0 / 10000.0,
        GPRMethod method = GPRMethod::ALC,
        int close = -1,
        int numstart = -1,
        Eigen::MatrixXd rect = Eigen::MatrixXd(),
        bool lite = true,
        int verb = 0);

    // Utility methods
    //Eigen::MatrixXd calculate_covariance(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
    vector<int> closest_indices(int start, const Eigen::MatrixXd& Xref, int n,
        const Eigen::MatrixXd& X, int close = -1, bool need_extra = false);
    Eigen::MatrixXd get_data_rect(const Eigen::MatrixXd& X);

    // Helper methods for prediction
    tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> new_predutilGP_lite(int nn, const Eigen::MatrixXd& XX);

    // Selection methods for different algorithms
    Eigen::VectorXd alc(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref, int verb = 0);
    Eigen::VectorXd mspe(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref, int verb = 0);
    int alcray_selection(const Eigen::MatrixXd& Xcand, const Eigen::MatrixXd& Xref,
        int offset, int numstart, const Eigen::MatrixXd& rect, int verb = 0);

    // Parameter optimization
    void optimize_parameters(double d = -1, double g = -1, int verb = 0);

    // Getters and setters
    double get_d() const { return d; }
    double get_g() const { return g; }
    void set_d(double new_d) { d = new_d; }
    void set_g(double new_g) { g = new_g; }
    void set_verbosity(int v) { verbosity = v; }

    // Error handling
    void throw_gpr_error(const string& message);
    void message(int level, const string& message);

private:
    Eigen::MatrixXd X;       // Training input points
    Eigen::VectorXd Z;       // Training output values
    Eigen::MatrixXd K;       // Covariance matrix
    Eigen::MatrixXd Ki;      // Inverse of covariance matrix
    double theta;            // Length scale parameter
    double nugget;           // Nugget parameter
    int n;                   // Number of training points
    double phi;              // Z.transpose() * Ki * Z
    double ldetK;            // Log determinant of K
    double llik;             // Log likelihood


    // Parameters
    double d;  // Length scale parameter
    double g;  // Nugget parameter
    int verbosity = 0;

    // PEST-related members
    Pest& pest_scenario;
    FileManager& file_manager;
    OutputFileWriter& output_file_writer;
    PerformanceLog* performance_log;
    RunManagerAbstract* run_mgr_ptr;
};

#endif // GPR_H_