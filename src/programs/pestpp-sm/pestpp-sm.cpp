/**
 * @file pestpp-sm.cpp
 * @brief A surrogate-model emulator utility.
 *
 * pestpp-sm behaves like pestpp-swp, but rather than simply evaluating a set
 * of input parameters with the model, it:
 *   1. runs the model for a (training) suite of parameter sets,
 *   2. records the training inputs and the corresponding model outputs,
 *   3. trains a surrogate-model emulator (one per observation); the currently
 *      implemented surrogate is Gaussian Process Regression (GPR),
 *   4. uses the trained emulator to predict the outputs for a second
 *      (prediction) suite of parameter sets - without running the model.
 *
 * The GPR framework is a C++/Eigen port of the laGPy library (see GPR.h).
 */

#include "RunManagerPanther.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <climits>
#include <limits>
#include <chrono>
#include <cmath>
#include "config_os.h"
#include "Pest.h"
#include "Transformable.h"
#include "Transformation.h"
#include "ParamTransformSeq.h"
#include "utilities.h"
#include "pest_error.h"
#include "ModelRunPP.h"
#include "FileManager.h"
#include "RunManagerSerial.h"
#include "OutputFileWriter.h"
#include "PantherAgent.h"
#include "Serialization.h"
#include "system_variables.h"
#include "RestartController.h"
#include "PerformanceLog.h"
#include "debug.h"
#include "logger.h"
#include "Jacobian.h"
#include "RunManagerExternal.h"
#include "GPR.h"
#include "Ensemble.h"
#include <random>
#include <thread>
#include <exception>
#include <vector>
#include <mutex>

using namespace std;
using namespace pest_utils;


/**
 * @brief Read the header of a parameter csv file, returning a map of control
 * file parameter name -> column index.
 */
static map<string, int> prepare_parameter_csv(Parameters pars, ifstream& csv, bool forgive,
	vector<string>* file_par_cols = nullptr)
{
	if (!csv.good())
		throw runtime_error("ifstream not good");

	string line;
	vector<string> header_tokens;
	if (!getline(csv, line))
		throw runtime_error("error reading header (first) line from csv file");
	strip_ip(line);
	upper_ip(line);
	tokenize(line, header_tokens, ",", false);
	for (auto& t : header_tokens)
		strip_ip(t);

	vector<string> missing_names;
	set<string> stokens(header_tokens.begin(), header_tokens.end());
	for (auto& p : pars)
		if (stokens.find(p.first) == stokens.end())
			missing_names.push_back(p.first);

	if (missing_names.size() > 0)
	{
		stringstream ss;
		ss << " the following pest control file parameter names were not found in the parameter csv file:" << endl;
		for (auto& n : missing_names) ss << n << endl;
		if (!forgive)
			throw runtime_error(ss.str());
		else
			cout << ss.str() << endl << "continuing anyway..." << endl;
	}

	if (header_tokens[header_tokens.size() - 1].size() == 0)
		header_tokens.pop_back();

	vector<string> ctl_pnames = pars.get_keys();
	unordered_set<string> s_pnames(ctl_pnames.begin(), ctl_pnames.end());
	unordered_set<string>::iterator end = s_pnames.end();
	map<string, int> header_info;
	for (int i = 0; i < (int)header_tokens.size(); i++)
		if (s_pnames.find(header_tokens[i]) != end)
			header_info[header_tokens[i]] = i;
	if (file_par_cols != nullptr)
	{
		file_par_cols->clear();
		for (int i = 0; i < (int)header_tokens.size(); i++)
			if (header_info.find(header_tokens[i]) != header_info.end())
				file_par_cols->push_back(header_tokens[i]);
	}
	return header_info;
}

/**
 * @brief Read every realization from a parameter csv file into a vector of
 * Parameters (missing values are taken from ctl_pars).
 */
static void load_all_parameters_from_csv(map<string, int>& header_info, ifstream& csv,
	const Parameters& ctl_pars, vector<string>& run_ids, vector<Parameters>& all_pars)
{
	int lcount = 1;
	run_ids.clear();
	all_pars.clear();
	double val;
	string line;
	vector<string> tokens;
	Parameters pars = ctl_pars;
	string run_id;

	while (getline(csv, line))
	{
		strip_ip(line);
		if (line.size() == 0)
			continue;
		tokens.clear();
		tokenize(line, tokens, ",", false);
		if (tokens[tokens.size() - 1].size() == 0)
			tokens.pop_back();

		if (tokens.size() != header_info.size() + 1)
		{
			stringstream ss;
			ss << "error parsing csv file on line " << lcount << ": wrong number of tokens, ";
			ss << "expecting " << header_info.size() + 1 << ", found " << tokens.size();
			throw runtime_error(ss.str());
		}
		convert_ip(tokens[0], run_id);
		for (auto hi : header_info)
		{
			try
			{
				val = stod(tokens[hi.second]);
			}
			catch (exception& e)
			{
				stringstream ss;
				ss << "error converting token '" << tokens[hi.second] << "' at location " << hi.first
					<< " to double on line " << lcount << ": " << line << endl << e.what();
				throw runtime_error(ss.str());
			}
			pars.update_rec(hi.first, val);
		}
		all_pars.push_back(pars);
		run_ids.push_back(run_id);
		lcount++;
	}
}

/**
 * @brief Read all realizations from a parameter file (csv, jcb/jco binary
 * jacobian, or dense binary) into a vector of Parameters.
 */
static void read_all_parameters(const string& par_file, Pest& pest_scenario, bool forgive,
	ofstream& fout_rec, vector<string>& run_ids, vector<Parameters>& all_pars, vector<string>* file_par_columns = nullptr)
{
	run_ids.clear();
	all_pars.clear();
	string par_ext = par_file.substr(par_file.size() - 3, par_file.size());
	lower_ip(par_ext);

	if ((par_ext.compare("jcb") == 0) || (par_ext.compare("jco") == 0))
	{
		cout << "  ---  binary jco-type file detected: " << par_file << endl;
		fout_rec << "  ---  binary jco-type file detected: " << par_file << endl;
		FileManager fm;
		Jacobian jco(fm);
		jco.read(par_file);
		vector<string> jnames = jco.get_base_numeric_par_names();
		vector<string> row_names = jco.get_sim_obs_names();
		set<string> jset(jnames.begin(), jnames.end());
		set<string> pset(pest_scenario.get_ctl_ordered_par_names().begin(),
			pest_scenario.get_ctl_ordered_par_names().end());
		vector<string> missing;
		set_symmetric_difference(jset.begin(), jset.end(), pset.begin(), pset.end(), back_inserter(missing));
		if ((missing.size() > 0) && (!forgive))
		{
			stringstream ss;
			ss << "binary jco file does not have the same parameters as the pest control file. Mismatched: ";
			for (auto& m : missing) ss << m << " , ";
			throw runtime_error(ss.str());
		}
		Eigen::MatrixXd mat = jco.get_matrix(row_names, pest_scenario.get_ctl_ordered_par_names(), forgive).toDense();
		vector<string> par_names = pest_scenario.get_ctl_ordered_par_names();
		if (file_par_columns != nullptr)
			*file_par_columns = par_names;
		Parameters base = pest_scenario.get_ctl_parameters();
		for (int i = 0; i < mat.rows(); i++)
		{
			Parameters p = base;
			p.update_without_clear(par_names, mat.row(i));
			all_pars.push_back(p);
			run_ids.push_back(row_names[i]);
		}
	}
	else if (par_ext.compare("csv") == 0)
	{
		ifstream par_stream(par_file);
		if (!par_stream.good())
			throw runtime_error("could not open parameter file " + par_file);
		map<string, int> header_info = prepare_parameter_csv(pest_scenario.get_ctl_parameters(), par_stream, forgive, file_par_columns);
		load_all_parameters_from_csv(header_info, par_stream, pest_scenario.get_ctl_parameters(), run_ids, all_pars);
		par_stream.close();
	}
	else
	{
		throw runtime_error("unrecognized parameter file extension (expecting .csv, .jcb, .jco): '" + par_ext + "'");
	}
}

/**
 * @brief Read the header of an observation csv file, returning a map of control
 * file observation name -> column index.
 */
static map<string, int> prepare_observation_csv(const Observations& ctl_obs, ifstream& csv, bool forgive)
{
	if (!csv.good())
		throw runtime_error("ifstream not good");

	string line;
	vector<string> header_tokens;
	if (!getline(csv, line))
		throw runtime_error("error reading header (first) line from csv file");
	strip_ip(line);
	upper_ip(line);
	tokenize(line, header_tokens, ",", false);
	for (auto& t : header_tokens)
		strip_ip(t);

	vector<string> missing_names;
	set<string> stokens(header_tokens.begin(), header_tokens.end());
	for (auto& o : ctl_obs)
		if (stokens.find(o.first) == stokens.end())
			missing_names.push_back(o.first);

	if (missing_names.size() > 0)
	{
		stringstream ss;
		ss << " the following pest control file observation names were not found in the observation csv file:" << endl;
		for (auto& n : missing_names) ss << n << endl;
		if (!forgive)
			throw runtime_error(ss.str());
		else
			cout << ss.str() << endl << "continuing anyway..." << endl;
	}

	if (header_tokens[header_tokens.size() - 1].size() == 0)
		header_tokens.pop_back();

	vector<string> ctl_onames = ctl_obs.get_keys();
	unordered_set<string> s_onames(ctl_onames.begin(), ctl_onames.end());
	unordered_set<string>::iterator end = s_onames.end();
	map<string, int> header_info;
	for (int i = 0; i < (int)header_tokens.size(); i++)
		if (s_onames.find(header_tokens[i]) != end)
			header_info[header_tokens[i]] = i;
	return header_info;
}

/**
 * @brief Read every realization from an observation csv file.
 */
static void load_all_observations_from_csv(map<string, int>& header_info, ifstream& csv,
	const Observations& ctl_obs, vector<string>& run_ids, vector<Observations>& all_obs)
{
	int lcount = 1;
	run_ids.clear();
	all_obs.clear();
	double val;
	string line;
	vector<string> tokens;
	Observations obs = ctl_obs;
	string run_id;

	while (getline(csv, line))
	{
		strip_ip(line);
		if (line.size() == 0)
			continue;
		tokens.clear();
		tokenize(line, tokens, ",", false);
		if (tokens[tokens.size() - 1].size() == 0)
			tokens.pop_back();

		if (tokens.size() < header_info.size() + 1)
		{
			stringstream ss;
			ss << "error parsing observation csv file on line " << lcount << ": wrong number of tokens, ";
			ss << "expecting at least " << header_info.size() + 1 << ", found " << tokens.size();
			throw runtime_error(ss.str());
		}
		convert_ip(tokens[0], run_id);
		for (auto hi : header_info)
		{
			try
			{
				val = stod(tokens[hi.second]);
			}
			catch (exception& e)
			{
				stringstream ss;
				ss << "error converting token '" << tokens[hi.second] << "' at location " << hi.first
					<< " to double on line " << lcount << ": " << line << endl << e.what();
				throw runtime_error(ss.str());
			}
			obs.update_rec(hi.first, val);
		}
		all_obs.push_back(obs);
		run_ids.push_back(run_id);
		lcount++;
	}
}

/**
 * @brief Read all realizations from a training-output observation csv file.
 */
static void read_all_observations(const string& obs_file, Pest& pest_scenario, bool forgive,
	ofstream& fout_rec, vector<string>& run_ids, vector<Observations>& all_obs)
{
	run_ids.clear();
	all_obs.clear();
	string obs_ext = obs_file.substr(obs_file.size() - 3, obs_file.size());
	lower_ip(obs_ext);

	if (obs_ext.compare("csv") == 0)
	{
		ifstream obs_stream(obs_file);
		if (!obs_stream.good())
			throw runtime_error("could not open observation file " + obs_file);
		map<string, int> header_info = prepare_observation_csv(pest_scenario.get_ctl_observations(), obs_stream, forgive);
		load_all_observations_from_csv(header_info, obs_stream, pest_scenario.get_ctl_observations(), run_ids, all_obs);
		obs_stream.close();
	}
	else
	{
		throw runtime_error("unrecognized training output file extension (expecting .csv): '" + obs_ext + "'");
	}
}

/**
 * @brief Require training input and output files to list the same real_name values
 * in the same order.
 */
static void check_training_run_id_consistency(const vector<string>& input_ids,
	const vector<string>& output_ids)
{
	if (input_ids.size() != output_ids.size())
	{
		stringstream ss;
		ss << "sm_training_input_file has " << input_ids.size()
			<< " realizations but sm_training_output_file has " << output_ids.size();
		throw runtime_error(ss.str());
	}
	for (int i = 0; i < (int)input_ids.size(); i++)
	{
		if (input_ids[i] != output_ids[i])
		{
			stringstream ss;
			ss << "real_name mismatch between sm_training_input_file and sm_training_output_file "
				<< "at row " << i << ": '" << input_ids[i] << "' vs '" << output_ids[i] << "'";
			throw runtime_error(ss.str());
		}
	}
}

/**
 * @brief Draw training parameter realizations uniformly within adjustable
 * parameter bounds.
 */
static void draw_uniform_training_parameters(Pest& pest_scenario, int num_reals,
	const vector<string>& adj_names, PerformanceLog& performance_log,
	ofstream& fout_rec, vector<string>& run_ids, vector<Parameters>& all_pars)
{
	random_device rd;
	mt19937 rand_gen(rd());
	ParameterEnsemble pe(&pest_scenario, &rand_gen);
	pe.draw_uniform(num_reals, adj_names, &performance_log, 0, fout_rec);
	pe.transform_ip(ParameterEnsemble::transStatus::CTL);

	run_ids = pe.get_real_names();
	Parameters ctl_base = pest_scenario.get_ctl_parameters();
	all_pars.clear();
	all_pars.reserve(num_reals);
	for (int i = 0; i < num_reals; i++)
	{
		Parameters pars = ctl_base;
		map<string, double> real_map = pe.get_real_map(run_ids[i]);
		for (const auto& pname : adj_names)
			pars.update_rec(pname, real_map.at(pname));
		all_pars.push_back(pars);
	}
}

/**
 * @brief Write parameter realizations to a csv file (real_name + parameter columns).
 */
static void write_par_csv(const string& filename, const vector<string>& col_names,
	const vector<string>& run_ids, const vector<Parameters>& pars_vec)
{
	ofstream out(filename);
	if (!out.good())
		throw runtime_error("could not open parameter file for writing: " + filename);
	out << setprecision(numeric_limits<double>::digits10);
	out << "real_name";
	for (auto& n : col_names)
		out << ',' << lower_cp(n);
	out << endl;
	for (int i = 0; i < (int)run_ids.size(); i++)
	{
		out << run_ids[i];
		Eigen::VectorXd xvec = pars_vec[i].get_data_eigen_vec(col_names);
		for (int j = 0; j < (int)xvec.size(); j++)
			out << ',' << xvec[j];
		out << endl;
	}
	out.close();
}

/**
 * @brief Write successful training inputs to ``<pst>.training.in.csv``.
 */
static void write_training_in_csv(const string& filename, const vector<string>& adj_names,
	const vector<string>& run_ids, const vector<Parameters>& pars_vec)
{
	write_par_csv(filename, adj_names, run_ids, pars_vec);
}

/**
 * @brief Write emulator predictions to ``<pst>.obs.csv`` (real_name, obs, obs_sd).
 */
static void write_pred_obs_csv(const string& filename, const vector<string>& obs_names,
	const vector<string>& run_ids, const Eigen::MatrixXd& pred_mean,
	const Eigen::MatrixXd& pred_std)
{
	ofstream out(filename);
	if (!out.good())
		throw runtime_error("could not open prediction output file for writing: " + filename);
	out << setprecision(numeric_limits<double>::digits10);
	out << "real_name";
	for (auto& n : obs_names)
	{
		out << ',' << lower_cp(n);
		out << ',' << lower_cp(n) << "_sd";
	}
	out << endl;
	for (int i = 0; i < (int)run_ids.size(); i++)
	{
		out << run_ids[i];
		for (int j = 0; j < (int)obs_names.size(); j++)
		{
			out << ',' << pred_mean(i, j);
			out << ',' << pred_std(i, j);
		}
		out << endl;
	}
	out.close();
}

/**
 * @brief Write emulator prediction gradients to a CSV.
 *
 * One row per (prediction realization, observation) pair; one column per
 * adjustable parameter.  ``deriv[j]`` holds the (n_pred x n_adj) gradient
 * matrix for observation ``j``.  Used for both the predictive-mean gradients
 * (``<pst>.pred.dmean.csv``) and the predictive-variance gradients
 * (``<pst>.pred.ds2.csv``).
 */
static void write_pred_deriv_csv(const string& filename, const vector<string>& obs_names,
	const vector<string>& adj_names, const vector<string>& run_ids,
	const vector<Eigen::MatrixXd>& deriv)
{
	ofstream out(filename);
	if (!out.good())
		throw runtime_error("could not open prediction gradient file for writing: " + filename);
	out << setprecision(numeric_limits<double>::digits10);
	out << "real_name,obs_name";
	for (auto& n : adj_names)
		out << ',' << lower_cp(n);
	out << endl;
	for (int i = 0; i < (int)run_ids.size(); i++)
	{
		for (int j = 0; j < (int)obs_names.size(); j++)
		{
			out << run_ids[i] << ',' << lower_cp(obs_names[j]);
			for (int c = 0; c < (int)adj_names.size(); c++)
				out << ',' << deriv[j](i, c);
			out << endl;
		}
	}
	out.close();
}

/**
 * @brief Write training outputs to ``<pst>.training.out.csv`` (real_name + observations only).
 */
static void write_training_out_csv(const string& filename, const vector<string>& obs_names,
	const vector<string>& run_ids, const vector<Observations>& obs_vec)
{
	ofstream out(filename);
	if (!out.good())
		throw runtime_error("could not open training output file for writing: " + filename);
	out << setprecision(numeric_limits<double>::digits10);
	out << "real_name";
	for (auto& n : obs_names)
		out << ',' << lower_cp(n);
	out << endl;
	for (int i = 0; i < (int)run_ids.size(); i++)
	{
		out << run_ids[i];
		Eigen::VectorXd zvec = obs_vec[i].get_data_eigen_vec(obs_names);
		for (int j = 0; j < (int)zvec.size(); j++)
			out << ',' << zvec[j];
		out << endl;
	}
	out.close();
}


int main(int argc, char* argv[])
{
#ifndef _DEBUG
	try
	{
#endif
		string version = PESTPP_VERSION;
		cout << endl << endl;
		cout << "             pestpp-sm - a surrogate-model emulator utility, version " << version << endl;
		cout << "                     for PEST(++) datasets " << endl << endl;
		cout << "                 by the PEST++ development team" << endl << endl << endl;
		auto start = chrono::steady_clock::now();
		string start_string = get_time_string();

		CmdLine cmdline(argc, argv);

		if (quit_file_found())
		{
			cerr << "'pest.stp' found, please remove this file " << endl;
			return 1;
		}

		FileManager file_manager;
		string filename = cmdline.ctl_file_name;
		string pathname = ".";
		file_manager.initialize_path(get_filename_without_ext(filename), pathname);
		string rns_file = file_manager.build_filename("rns");
		remove(rns_file.c_str());

		if (cmdline.runmanagertype == CmdLine::RunManagerType::GENIE)
		{
			cerr << "Genie run manager ('/e') deprecated, please use panther instead" << endl;
			exit(1);
		}

		if (cmdline.runmanagertype == CmdLine::RunManagerType::PANTHER_WORKER)
		{
			try
			{
				ofstream frec("panther_worker.rec");
				if (frec.bad())
					throw runtime_error("error opening 'panther_worker.rec'");
				cmdline.startup_report(frec, start_string);
				cmdline.startup_report(cout, start_string);
				PANTHERAgent yam_agent(frec);
				string ctl_file = "";
				try
				{
					ctl_file = file_manager.build_filename("pst");
					yam_agent.process_ctl_file(ctl_file);
				}
				catch (exception& e)
				{
					frec << "Error processing control file: " << ctl_file << endl << endl;
					frec << e.what() << endl << endl;
					cerr << "Error processing control file: " << ctl_file << endl << endl;
					cerr << e.what() << endl << endl;
					throw(e);
				}
				yam_agent.start(cmdline.panther_host_name, cmdline.panther_port);
			}
			catch (PestError& perr)
			{
				cerr << perr.what();
				throw(perr);
			}
			cout << endl << "Work Done..." << endl;
			exit(0);
		}

		debug_initialize(file_manager.build_filename("dbg"));
		if (cmdline.jac_restart)
			throw runtime_error("/j option not supported by pestpp-sm");
		if (cmdline.restart)
			throw runtime_error("/r option not supported by pestpp-sm");

		file_manager.open_default_files();
		ofstream& fout_rec = file_manager.rec_ofstream();
		PerformanceLog performance_log(file_manager.open_ofile_ext("log"));

		fout_rec << "             pestpp-sm.exe - a surrogate-model (GPR) emulator utility" << endl
			<< "for PEST(++) datasets " << endl << endl;
		fout_rec << "                 by the PEST++ development team" << endl << endl << endl;
		cmdline.startup_report(fout_rec, start_string);
		cmdline.startup_report(cout, start_string);

		// create pest run and process control file to initialize it
		Pest pest_scenario;
		pest_scenario.set_default_dynreg();
		try
		{
			performance_log.log_event("starting to process control file");
			pest_scenario.process_ctl_file(file_manager.open_ifile_ext("pst"), file_manager.build_filename("pst"), fout_rec);
			file_manager.close_file("pst");
			performance_log.log_event("finished processing control file");
		}
		catch (exception& e)
		{
			cerr << "Error processing control file: " << filename << endl << endl;
			cerr << e.what() << endl << endl;
			fout_rec << "Error processing control file: " << filename << endl << endl;
			fout_rec << e.what() << endl << endl;
			fout_rec.close();
			throw(e);
		}
		pest_scenario.check_inputs(fout_rec, true);

		if (!pest_scenario.get_pestpp_options().get_sweep_include_regul_phi())
		{
			pest_scenario.get_regul_scheme_ptr()->set_zero();
			pest_scenario.get_prior_info_ptr()->clear();
		}

		OutputFileWriter ofw(file_manager, pest_scenario, false, false, 0);
		ofw.scenario_report(fout_rec, false);

		PestppOptions ppo = pest_scenario.get_pestpp_options();
		string train_input_file = ppo.get_sm_training_input_file();
		string train_output_file = ppo.get_sm_training_output_file();
		int sm_training_data_size = ppo.get_sm_training_data_size();
		bool use_preevaluated_training = (train_output_file.size() > 0);
		string input_file = ppo.get_sm_input_file();
		bool forgive = ppo.get_sweep_forgive();
		int chunk = ppo.get_sweep_chunk();
		double d_in = ppo.get_gpr_lengthscale();
		double g_in = ppo.get_gpr_nugget();
		bool use_local = ppo.get_gpr_local();
		int local_start = ppo.get_gpr_local_start();
		int local_end = ppo.get_gpr_local_end();
		string local_method = ppo.get_gpr_local_method();
		int verb = ppo.get_gpr_verbose();

		int sm_num_threads = ppo.get_sm_num_threads();
		if (sm_num_threads < 1)
			sm_num_threads = 1;
		
		GPKernel gpr_kernel = gpr_kernel_from_string(ppo.get_gpr_kernel());
		bool compute_derivs = ppo.get_gpr_compute_derivatives();

		string pst_base = file_manager.get_base_filename();
		string train_in_archive = pst_base + ".training.in.csv";
		string train_out_archive = pst_base + ".training.out.csv";
		string pred_par_archive = pst_base + ".par.csv";
		string pred_obs_archive = pst_base + ".obs.csv";
		string pred_dmean_archive = pst_base + ".pred.dmean.csv";
		string pred_ds2_archive = pst_base + ".pred.ds2.csv";

		fout_rec << endl << "    sm training input parameter file = " << train_input_file << endl;
		fout_rec << "    sm training data size = " << sm_training_data_size << endl;
		if (use_preevaluated_training)
			fout_rec << "    sm training output file (pre-evaluated) = " << train_output_file << endl;
		fout_rec << "    gpr prediction (emulated) parameter file = " << input_file << endl;
		fout_rec << "    gpr training input archive file = " << train_in_archive << endl;
		fout_rec << "    gpr training output archive file = " << train_out_archive << endl;
		fout_rec << "    gpr prediction parameter archive file = " << pred_par_archive << endl;
		fout_rec << "    gpr prediction output archive file = " << pred_obs_archive << endl;
		fout_rec << "    gpr lengthscale (<=0 => MLE) = " << d_in << endl;
		fout_rec << "    gpr nugget (<=0 => MLE) = " << g_in << endl;
		fout_rec << "    gpr local approximate GP = " << use_local << endl;
		if (use_local)
		{
			fout_rec << "    gpr local design start size = " << local_start << endl;
			fout_rec << "    gpr local design end size = " << local_end << endl;
			fout_rec << "    gpr local design method = " << local_method << endl;
		}
		fout_rec << "    sm number of threads = " << sm_num_threads << endl;
		fout_rec << "    gpr kernel = " << gpr_kernel_to_string(gpr_kernel) << endl;
		fout_rec << "    gpr compute derivatives = " << compute_derivs << endl;

		if (pest_scenario.get_pestpp_options().get_debug_parse_only())
		{
			cout << endl << endl << "DEBUG_PARSE_ONLY is true, exiting..." << endl << endl;
			exit(0);
		}

		vector<string> adj_names = pest_scenario.get_ctl_ordered_adj_par_names();
		vector<string> obs_names = pest_scenario.get_ctl_ordered_obs_names();
		int n_adj = (int)adj_names.size();
		int n_obs = (int)obs_names.size();
		if (n_adj == 0)
			throw runtime_error("pestpp-sm requires at least one adjustable parameter");

		//-------------------------------------------------------------
		// Step 1: training data (model runs or pre-evaluated outputs)
		//-------------------------------------------------------------
		vector<Parameters> train_pars;
		vector<string> train_run_ids;
		bool drew_training = false;
		if (train_input_file.size() == 0)
		{
			if (sm_training_data_size < 2)
				throw runtime_error("pestpp-sm requires sm_training_data_size >= 2 when sm_training_input_file is not supplied");
			if (use_preevaluated_training)
				throw runtime_error("sm_training_output_file requires sm_training_input_file when training data is not drawn");
			cout << endl << "...drawing " << sm_training_data_size << " uniform training parameter sets" << endl;
			fout_rec << endl << "...drawing " << sm_training_data_size << " uniform training parameter sets" << endl;
			draw_uniform_training_parameters(pest_scenario, sm_training_data_size, adj_names, performance_log, fout_rec, train_run_ids, train_pars);
			drew_training = true;
		}
		else
		{
			read_all_parameters(train_input_file, pest_scenario, forgive, fout_rec, train_run_ids, train_pars);
			int n_file = (int)train_pars.size();
			if (sm_training_data_size > 0)
			{
				if (n_file > sm_training_data_size)
				{
					stringstream ss;
					ss << "   sm_training_input_file has " << n_file << " realizations, "
						<< "overriding sm_training_data_size(" << sm_training_data_size << ")";
					cout << ss.str() << endl;
					fout_rec << ss.str() << endl;
				}
				else if (n_file < sm_training_data_size)
				{
					stringstream ss;
					ss << "WARNING: sm_training_input_file has " << n_file << " realizations, "
						<< "less than sm_training_data_size(" << sm_training_data_size << ")";
					cout << ss.str() << endl;
					fout_rec << ss.str() << endl;
				}
			}
		}
		int n_train_in = (int)train_pars.size();
		if (n_train_in < 2)
			throw runtime_error("pestpp-sm requires at least 2 training parameter sets");

		vector<vector<double>> Xtrain_rows;
		vector<vector<double>> Ztrain_rows;
		vector<Observations> train_success_obs;
		vector<Parameters> train_success_pars;
		vector<string> train_success_ids;
		int n_train = 0;

		if (use_preevaluated_training)
		{
			cout << endl << "...loading pre-evaluated training data" << endl;
			fout_rec << endl << "...loading pre-evaluated training data" << endl;
			cout << "   " << n_train_in << " training parameter sets read from " << train_input_file << endl;
			fout_rec << "   " << n_train_in << " training parameter sets read from " << train_input_file << endl;

			vector<string> train_obs_ids;
			vector<Observations> train_obs;
			read_all_observations(train_output_file, pest_scenario, forgive, fout_rec, train_obs_ids, train_obs);
			cout << "   " << train_obs.size() << " training output sets read from " << train_output_file << endl;
			fout_rec << "   " << train_obs.size() << " training output sets read from " << train_output_file << endl;
			check_training_run_id_consistency(train_run_ids, train_obs_ids);

			n_train = n_train_in;
			train_success_ids = train_run_ids;
			train_success_pars = train_pars;
			train_success_obs = train_obs;
			Xtrain_rows.reserve(n_train);
			Ztrain_rows.reserve(n_train);
			for (int i = 0; i < n_train; i++)
			{
				Eigen::VectorXd xvec = train_pars[i].get_data_eigen_vec(adj_names);
				Eigen::VectorXd zvec = train_obs[i].get_data_eigen_vec(obs_names);
				vector<double> xr(xvec.data(), xvec.data() + xvec.size());
				vector<double> zr(zvec.data(), zvec.data() + zvec.size());
				Xtrain_rows.push_back(xr);
				Ztrain_rows.push_back(zr);
			}
		}
		else
		{
			cout << endl << "...running training parameter sets through the model" << endl;
			fout_rec << endl << "...running training parameter sets through the model" << endl;
			if (drew_training)
			{
				cout << "   " << n_train_in << " training parameter sets drawn uniformly within bounds" << endl;
				fout_rec << "   " << n_train_in << " training parameter sets drawn uniformly within bounds" << endl;
			}
			else
			{
				cout << "   " << n_train_in << " training parameter sets read from " << train_input_file << endl;
				fout_rec << "   " << n_train_in << " training parameter sets read from " << train_input_file << endl;
			}

			// set up the run manager (only needed when evaluating the model)
			RunManagerAbstract* run_manager_ptr;
			if (cmdline.runmanagertype == CmdLine::RunManagerType::PANTHER_MASTER)
			{
				run_manager_ptr = new RunManagerPanther(
					rns_file, cmdline.panther_port,
					file_manager.open_ofile_ext("rmr"),
					pest_scenario.get_pestpp_options().get_max_run_fail(),
					pest_scenario.get_pestpp_options().get_overdue_reched_fac(),
					pest_scenario.get_pestpp_options().get_overdue_giveup_fac(),
					pest_scenario.get_pestpp_options().get_overdue_giveup_minutes(),
					pest_scenario.get_pestpp_options().get_panther_echo(),
					vector<string>{}, vector<string>{},
					pest_scenario.get_pestpp_options().get_panther_timeout_milliseconds(),
					pest_scenario.get_pestpp_options().get_panther_echo_interval_milliseconds(),
					pest_scenario.get_pestpp_options().get_panther_persistent_workers(),
					pest_scenario.get_pestpp_options().get_panther_ping_interval_secs());
			}
			else if (cmdline.runmanagertype == CmdLine::RunManagerType::EXTERNAL)
			{
				const ModelExecInfo& exi = pest_scenario.get_model_exec_info();
				run_manager_ptr = new RunManagerExternal(exi.comline_vec,
					exi.tplfile_vec, exi.inpfile_vec, exi.insfile_vec, exi.outfile_vec, rns_file);
			}
			else
			{
				performance_log.log_event("starting basic model IO error checking");
				cout << "checking model IO files...";
				pest_scenario.check_io(fout_rec);
				performance_log.log_event("finished basic model IO error checking");
				cout << "done" << endl;
				const ModelExecInfo& exi = pest_scenario.get_model_exec_info();
				run_manager_ptr = new RunManagerSerial(exi.comline_vec,
					exi.tplfile_vec, exi.inpfile_vec, exi.insfile_vec, exi.outfile_vec,
					file_manager.build_filename("rns"), pathname,
					pest_scenario.get_pestpp_options().get_max_run_fail(),
					pest_scenario.get_pestpp_options().get_fill_tpl_zeros(),
					pest_scenario.get_pestpp_options().get_additional_ins_delimiters(),
					pest_scenario.get_pestpp_options().get_num_tpl_ins_threads(),
					pest_scenario.get_pestpp_options().get_tpl_force_decimal());
			}

			const ParamTransformSeq& base_trans_seq = pest_scenario.get_base_par_tran_seq();
			run_manager_ptr->initialize(base_trans_seq.ctl2model_cp(pest_scenario.get_ctl_parameters()),
				pest_scenario.get_ctl_observations());

			int done = 0;
			while (done < n_train_in)
			{
				int this_chunk = min(chunk, n_train_in - done);
				run_manager_ptr->reinitialize();
				vector<int> idmap(this_chunk);
				for (int i = 0; i < this_chunk; i++)
					idmap[i] = run_manager_ptr->add_run(base_trans_seq.active_ctl2model_cp(train_pars[done + i]));

				cout << "   running training runs " << done << " --> " << done + this_chunk << endl;
				run_manager_ptr->run();

				for (int i = 0; i < this_chunk; i++)
				{
					int row = done + i;
					Parameters rpars;
					Observations robs;
					bool success = run_manager_ptr->get_run(idmap[i], rpars, robs);
					if (success)
					{
						Eigen::VectorXd xvec = train_pars[row].get_data_eigen_vec(adj_names);
						Eigen::VectorXd zvec = robs.get_data_eigen_vec(obs_names);
						vector<double> xr(xvec.data(), xvec.data() + xvec.size());
						vector<double> zr(zvec.data(), zvec.data() + zvec.size());
						Xtrain_rows.push_back(xr);
						Ztrain_rows.push_back(zr);
						train_success_obs.push_back(robs);
						train_success_pars.push_back(train_pars[row]);
						train_success_ids.push_back(train_run_ids[row]);
					}
				}
				done += this_chunk;

				int q = pest_utils::quit_file_found();
				if ((q == 1) || (q == 2))
				{
					cout << "'pest.stp' found, quitting" << endl;
					fout_rec << "'pest.stp' found, quitting" << endl;
					break;
				}
			}
			delete run_manager_ptr;

			n_train = (int)train_success_ids.size();
			cout << "   " << n_train << " of " << n_train_in << " training runs completed successfully" << endl;
			fout_rec << "   " << n_train << " of " << n_train_in << " training runs completed successfully" << endl;
		}

		if (n_train < 2)
			throw runtime_error("fewer than 2 successful training runs - cannot train a GPR emulator");

		cout << endl << "...writing training archive files" << endl;
		fout_rec << endl << "...writing training archive files" << endl;
		write_training_in_csv(train_in_archive, adj_names, train_success_ids, train_success_pars);
		write_training_out_csv(train_out_archive, obs_names, train_success_ids, train_success_obs);

		cout << "   training inputs written to " << train_in_archive << endl;
		cout << "   training outputs written to " << train_out_archive << endl;
		fout_rec << "   training inputs written to " << train_in_archive << endl;
		fout_rec << "   training outputs written to " << train_out_archive << endl;

		// assemble training matrices
		Eigen::MatrixXd Xtrain(n_train, n_adj);
		Eigen::MatrixXd Ztrain(n_train, n_obs);
		for (int i = 0; i < n_train; i++)
		{
			for (int j = 0; j < n_adj; j++)
				Xtrain(i, j) = Xtrain_rows[i][j];
			for (int j = 0; j < n_obs; j++)
				Ztrain(i, j) = Ztrain_rows[i][j];
		}

		//-------------------------------------------------------------
		// Step 2: read the prediction parameter sets
		//-------------------------------------------------------------
		cout << endl << "...reading prediction parameter sets" << endl;
		fout_rec << endl << "...reading prediction parameter sets" << endl;
		vector<Parameters> pred_pars;
		vector<string> pred_run_ids;
		vector<string> pred_par_cols;
		read_all_parameters(input_file, pest_scenario, forgive, fout_rec, pred_run_ids, pred_pars, &pred_par_cols);
		int n_pred = (int)pred_pars.size();
		cout << "   " << n_pred << " prediction parameter sets read from " << input_file << endl;
		fout_rec << "   " << n_pred << " prediction parameter sets read from " << input_file << endl;
		if (n_pred == 0)
			throw runtime_error("no prediction parameter sets found in " + input_file);
		if (pred_par_cols.empty())
			pred_par_cols = adj_names;

		Eigen::MatrixXd Xpred(n_pred, n_adj);
		for (int i = 0; i < n_pred; i++)
		{
			Eigen::VectorXd xvec = pred_pars[i].get_data_eigen_vec(adj_names);
			for (int j = 0; j < n_adj; j++)
				Xpred(i, j) = xvec[j];
		}

		//-------------------------------------------------------------
		// Step 3: train one GPR per observation and predict
		//-------------------------------------------------------------
		cout << endl << "...training GPR emulator(s) and predicting" << endl;
		fout_rec << endl << "...training GPR emulator(s) and predicting" << endl;

		Eigen::MatrixXd PredMean(n_pred, n_obs);
		Eigen::MatrixXd PredStd(n_pred, n_obs);

		// optional per-observation analytic gradients of the prediction with
		// respect to the adjustable parameters (n_pred x n_adj each); only
		// allocated when GPR_COMPUTE_DERIVATIVES is true.
		vector<Eigen::MatrixXd> PredDMean, PredDS2;
		if (compute_derivs)
		{
			PredDMean.assign(n_obs, Eigen::MatrixXd::Zero(n_pred, n_adj));
			PredDS2.assign(n_obs, Eigen::MatrixXd::Zero(n_pred, n_adj));
		}

		GPR gpr_engine(gpr_kernel);
		performance_log.log_event("starting GPR training/prediction");

		// train one GP for observation 'j' and write its predictive mean/sd
		// columns.  shared inputs (Xtrain, Xpred, ...) are read-only and the
		// only writes are to the distinct column 'j' of PredMean/PredStd, so
		// this is safe to run concurrently across observations.  'row_threads'
		// is forwarded to the local GP so its per-prediction-row loop can be
		// parallelized when the observation loop itself is serial.
		mutex sm_log_mutex;
		auto process_obs = [&](int j, int row_threads)
		{
			Eigen::VectorXd Zj = Ztrain.col(j);
			double zmin = Zj.minCoeff();
			double zmax = Zj.maxCoeff();

			Eigen::VectorXd mean_j, s2_j;
			Eigen::MatrixXd dmean_j, ds2_j;
			Eigen::MatrixXd* dmean_ptr = compute_derivs ? &dmean_j : nullptr;
			Eigen::MatrixXd* ds2_ptr = compute_derivs ? &ds2_j : nullptr;
			if ((zmax - zmin) < (1.0e-30 + 1.0e-10 * (fabs(zmax) + fabs(zmin))))
			{
				// observation is (effectively) constant over the training set
				mean_j = Eigen::VectorXd::Constant(n_pred, Zj.mean());
				s2_j = Eigen::VectorXd::Zero(n_pred);
				if (compute_derivs)
				{
					dmean_j = Eigen::MatrixXd::Zero(n_pred, n_adj);
					ds2_j = Eigen::MatrixXd::Zero(n_pred, n_adj);
				}
			}
			else
			{
				try
				{
					if (use_local)
						gpr_engine.local_gp_predict(Xtrain, Zj, Xpred, local_start, local_end,
							local_method, d_in, g_in, verb, mean_j, s2_j, row_threads, dmean_ptr, ds2_ptr);
					else
					{
						double d_used, g_used;
						gpr_engine.full_gp_predict(Xtrain, Zj, Xpred, d_in, g_in, verb,
							mean_j, s2_j, d_used, g_used, dmean_ptr, ds2_ptr);
						if (verb > 0)
						{
							lock_guard<mutex> lk(sm_log_mutex);
							fout_rec << "   obs '" << obs_names[j] << "': d=" << d_used << " g=" << g_used << endl;
						}
					}
				}
				catch (exception& e)
				{
					// fall back to the training mean if the GP build fails
					{
						lock_guard<mutex> lk(sm_log_mutex);
						fout_rec << "   WARNING: GPR failed for observation '" << obs_names[j]
							<< "' (" << e.what() << "); using training mean" << endl;
						cout << "   WARNING: GPR failed for observation '" << obs_names[j]
							<< "'; using training mean" << endl;
					}
					mean_j = Eigen::VectorXd::Constant(n_pred, Zj.mean());
					double var = 0.0;
					double zm = Zj.mean();
					for (int i = 0; i < n_train; i++) var += (Zj[i] - zm) * (Zj[i] - zm);
					var /= (double)n_train;
					s2_j = Eigen::VectorXd::Constant(n_pred, var);
					if (compute_derivs)
					{
						dmean_j = Eigen::MatrixXd::Zero(n_pred, n_adj);
						ds2_j = Eigen::MatrixXd::Zero(n_pred, n_adj);
					}
				}
			}
			PredMean.col(j) = mean_j;
			for (int i = 0; i < n_pred; i++)
				PredStd(i, j) = sqrt(max(0.0, s2_j[i]));
			if (compute_derivs)
			{
				PredDMean[j] = dmean_j;
				PredDS2[j] = ds2_j;
			}
		};

		if (use_local)
		{
			if (sm_num_threads > 1)
				cout << "   training/predicting with up to " << sm_num_threads << " threads over prediction points (per observation)" << endl;
			int report_every = max(1, n_obs / 20);
			for (int j = 0; j < n_obs; j++)
			{
				process_obs(j, sm_num_threads);
				if ((j + 1) % report_every == 0 || j == n_obs - 1)
					cout << "   trained/predicted " << (j + 1) << " of " << n_obs << " observations\r" << flush;
			}
			cout << endl;
		}
		else if ((sm_num_threads < 2) || (n_obs < 2))
		{
			// full GP, serial over observations
			int report_every = max(1, n_obs / 20);
			for (int j = 0; j < n_obs; j++)
			{
				process_obs(j, 1);
				if ((j + 1) % report_every == 0 || j == n_obs - 1)
					cout << "   trained/predicted " << (j + 1) << " of " << n_obs << " observations\r" << flush;
			}
			cout << endl;
		}
		else
		{
			// full GP: each observation is an independent global GP, so spread
			// the observation loop across threads using a dynamic work queue
			// (Option B).  threads pull the next observation index from a
			// shared counter guarded by a mutex, self-balancing when per-obs
			// cost varies.  keep Eigen single-threaded so the per-observation
			// linear algebra does not over-subscribe the cores we dispatch over.
			int nthreads = min(sm_num_threads, n_obs);
			cout << "   training/predicting " << n_obs << " observations using "
				<< nthreads << " threads" << endl;
			Eigen::setNbThreads(1);
			vector<thread> obs_threads;
			vector<exception_ptr> obs_eptrs(nthreads, nullptr);
			int next_obs = 0;
			mutex next_obs_lock;
			auto obs_queue = [&](int tid)
			{
				try
				{
					while (true)
					{
						int j;
						{
							lock_guard<mutex> guard(next_obs_lock);
							if (next_obs >= n_obs)
								break;
							j = next_obs;
							next_obs++;
						}
						process_obs(j, 1);
					}
				}
				catch (...)
				{
					obs_eptrs[tid] = current_exception();
				}
			};
			for (int t = 0; t < nthreads; t++)
				obs_threads.push_back(thread(obs_queue, t));
			for (int t = 0; t < nthreads; t++)
				obs_threads[t].join();
			for (int t = 0; t < nthreads; t++)
				if (obs_eptrs[t])
					rethrow_exception(obs_eptrs[t]);
		}
		performance_log.log_event("finished GPR training/prediction");

		//-------------------------------------------------------------
		// write outputs
		//-------------------------------------------------------------
		cout << endl << "...writing emulator prediction archives" << endl;
		fout_rec << endl << "...writing emulator prediction archives" << endl;
		write_par_csv(pred_par_archive, pred_par_cols, pred_run_ids, pred_pars);
		write_pred_obs_csv(pred_obs_archive, obs_names, pred_run_ids, PredMean, PredStd);
		cout << "   prediction parameters written to " << pred_par_archive << endl;
		cout << "   emulator predictions written to " << pred_obs_archive << endl;
		fout_rec << "   prediction parameters written to " << pred_par_archive << endl;
		fout_rec << "   emulator predictions written to " << pred_obs_archive << endl;
		if (compute_derivs)
		{
			write_pred_deriv_csv(pred_dmean_archive, obs_names, adj_names, pred_run_ids, PredDMean);
			write_pred_deriv_csv(pred_ds2_archive, obs_names, adj_names, pred_run_ids, PredDS2);
			cout << "   prediction mean gradients written to " << pred_dmean_archive << endl;
			cout << "   prediction variance gradients written to " << pred_ds2_archive << endl;
			fout_rec << "   prediction mean gradients written to " << pred_dmean_archive << endl;
			fout_rec << "   prediction variance gradients written to " << pred_ds2_archive << endl;
		}

		string case_name = file_manager.get_base_filename();
		file_manager.close_file("rst");
		pest_utils::try_clean_up_run_storage_files(case_name);

		cout << endl << endl << "pestpp-sm analysis complete..." << endl;
		fout_rec << endl << endl << "pestpp-sm analysis complete..." << endl;
		auto end = chrono::steady_clock::now();
		cout << "started at " << start_string << endl;
		cout << "finished at " << get_time_string() << endl;
		cout << "took " << setprecision(6) << (double)chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0 << " minutes" << endl;
		cout << flush;
		fout_rec << "started at " << start_string << endl;
		fout_rec << "finished at " << get_time_string() << endl;
		fout_rec << "took " << setprecision(6) << (double)chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0 << " minutes" << endl;
		fout_rec.close();

		return 0;
#ifndef _DEBUG
	}
	catch (exception& e)
	{
		cout << "Error condition prevents further execution: " << endl << e.what() << endl;
		return 1;
	}
	catch (...)
	{
		cout << "Error condition prevents further execution" << endl;
		return 1;
	}
#endif
}
