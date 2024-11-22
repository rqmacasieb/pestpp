#ifndef MOEA_H_
#define MOEA_H_

#include <unordered_map>
#include <random>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include "FileManager.h"
#include "ObjectiveFunc.h"
#include "OutputFileWriter.h"
#include "PerformanceLog.h"
#include "Ensemble.h"
#include "constraints.h"
#include "EnsembleMethodUtils.h"


const string POP_SUM_TAG = "pareto.summary.csv";
const string ARC_SUM_TAG = "pareto.archive.summary.csv";
const string INNER_POP_SUM_TAG = "inner.pareto.summary.csv";
const string INNER_ARC_SUM_TAG = "inner.pareto.archive.summary.csv";

const string ARC_TRIM_SUM_TAG = "pareto.trimmed.archive.summary.csv";
const string BGO_POP_SUM_TAG = "enbgo.summary.csv";
const string BGO_ARC_SUM_TAG = "enbgo.archive.summary.csv";
const string BGO_TRAINING_SUM_TAG = "training_dataset.summary.csv";
const string BGO_SELECTION_SUM_TAG = "infill_search.summary.csv";
const string RISK_NAME = "_RISK_";
const string DE_F_NAME = "_DE_F_";
const string CR_NAME = "_CR_";
const string MR_NAME = "_MR_";
const double CROWDING_EXTREME = 1.0e+30;
const bool bgo_mode = true;

enum MouGenType { DE, SBX, PM, PSO, SMP };
enum MouEnvType { NSGA, SPEA, NSGA_PPD, NSGA_BGO };
enum MouMateType { RANDOM, TOURNAMENT };

class ParetoObjectives
{
public:
	ParetoObjectives(Pest& _pest_scenario, FileManager& _file_manager, 
		PerformanceLog* _performance_log);

	pair<vector<string>, vector<string>> get_nsga2_pareto_dominance(int generation, ObservationEnsemble& op, 
		ParameterEnsemble& dp, Constraints* constraints_ptr=nullptr, bool ppd=false, bool report=true, string sum_tag=string());

	map<string, map<string, double>> get_members(ObservationEnsemble& op, ParameterEnsemble& dp) { return get_member_struct(op, dp); };
	void set_ppd_beta() { ppd_beta = pest_scenario.get_pestpp_options().get_mou_ppd_beta(); }
	void set_prob_pareto(bool ppd) { prob_pareto = ppd; }
	void set_hypervolume_partitions(map<string, map<string, double>> _hv_parts);
	void get_ehvi(ObservationEnsemble& op, ParameterEnsemble& dp);
	void update_ppd_criteria(ObservationEnsemble& op, ParameterEnsemble& dp);
	
	pair<vector<string>, vector<string>> get_bgo_ensemble(int generation, ObservationEnsemble& _op, 
		ParameterEnsemble& _dp, Constraints* constraints_ptr = nullptr, bool report=true, string sum_tag=string());
	void set_bgo_mode(bool bgo_switch) { bgo = bgo_switch; }
	map<string, double> get_enbgo_fitness_map() { return enbgo_fitness_map; }
	map<string, double> get_bgo_aqf_map() { return bgo_aqf_map; }
	void update_bgo_ensemble(ObservationEnsemble& op, ParameterEnsemble& dp, Constraints* constraints_ptr);
	vector<pair<string, double>> sort_by_aqf(ObservationEnsemble& op, ParameterEnsemble& dp, Constraints* constraints_ptr);

	void write_bgo_ensemble_summary(string& sum_tag, int generation, ObservationEnsemble& op, ParameterEnsemble& dp, Constraints* constr_ptr);
	void prep_bgo_ensemble_summary_file(string summary_tag);
	void write_training_summary(int generation, int inner_iter, ParameterEnsemble& dt, ObservationEnsemble& ot, string sum_tag);
	void prep_bgo_training_summary_file(string summary_tag);

	//this must be called at least once before the diversity metrixs can be called...
	void set_pointers(vector<string>& _obj_names, vector<string>& _obs_obj_names, vector<string>& _obs_obj_sd_names, vector<string>& _pi_obj_names, vector<string>& _pi_obj_sd_names, map<string, double>& _obj_dir_mult)
	{
		obj_names_ptr = &_obj_names;
		obs_obj_names_ptr = &_obs_obj_names;
		obs_obj_sd_names_ptr = &_obs_obj_sd_names;
		pi_obj_names_ptr = &_pi_obj_names;
		pi_obj_sd_names_ptr = &_pi_obj_sd_names;
		obj_dir_mult_ptr = &_obj_dir_mult;
		/*if (bgo)
		{
			prep_bgo_ensemble_summary_file(BGO_POP_SUM_TAG);
			prep_bgo_ensemble_summary_file(BGO_ARC_SUM_TAG);
			prep_bgo_training_summary_file(BGO_TRAINING_SUM_TAG);

		}
		else
		{*/
		prep_pareto_summary_file(POP_SUM_TAG);
		prep_pareto_summary_file(ARC_SUM_TAG);
		//prep_pareto_summary_file(ARC_TRIM_SUM_TAG);
		//}
		
	}
	
	void set_curr_opt(map<string, map<string,double>>& curr_tobs)
	{
		map<string, double> curr;
		for (auto& c : curr_tobs)
			curr = c.second;
		for (auto& o : *obj_names_ptr)
			c_opt = curr.at(o);
	}

	void update(ObservationEnsemble& oe, ParameterEnsemble& dp, Constraints* constraints_ptr = nullptr);

	bool compare_two(string& first,string& second, MouEnvType envtyp, bool bgomode = false);

	map<string, double> get_spea2_fitness(int generation, ObservationEnsemble& op, ParameterEnsemble& dp, 
		Constraints* constraints_ptr = nullptr, bool report = true, string sum_tag = string());
	
	map<string, double> get_spea2_kth_nn_crowding_distance(ObservationEnsemble& oe, ParameterEnsemble& dp);
	 
	void get_spea2_archive_names_to_keep(int num_members, vector<string>& keep, const ObservationEnsemble& op, const ParameterEnsemble& dp);

	void prep_pareto_summary_file(string summary_tag);
	void write_pareto_summary(string& sum_tag, int generation, ObservationEnsemble& op, ParameterEnsemble& dp, 
		Constraints* constr_ptr=nullptr);

	//sort specific members
	map<string, double> get_cuboid_crowding_distance(vector<string>& members);
	map<string, double> get_cluster_crowding_fitness(vector<string>& members);
	void prep_expected_distance_lookup_table(ObservationEnsemble& op, ParameterEnsemble& dp);
	map<string, double> get_ehvi(vector<string>& members);
	map<string, double> get_mopso_fitness(vector<string> members, ObservationEnsemble& op, ParameterEnsemble& dp);

	double get_ei(map<string, double> phi, string obj, double curr_opt);

	set<string> get_duplicates() { return duplicates;  }

	int get_num_feasible(){ return feas_member_struct.size();}

private:
	
	Pest& pest_scenario;
	FileManager& file_manager;
	PerformanceLog* performance_log;
	//vector<string> obj_names;
	vector<string> sort_members_by_crowding_distance(int front, vector<string>& members, map<string, double>& crowd_map, map<string, map<string, double>>& _member_struct);
	bool first_dominates_second(map<string, double>& first, map<string, double>& second);
	map<string, map<string, double>> get_member_struct(ObservationEnsemble& oe, ParameterEnsemble& dp);
	map<string, map<string, double>> get_bgo_ensemble_struct(map<string, map<string, double>> _member_struct);
	void drop_duplicates(map<string, map<string, double>>& _member_struct);
		
	bool first_equals_second(map<string, double>& first, map<string, double>& second);

	map<int, vector<string>> sort_members_by_dominance_into_fronts(map<string, map<string, double>>& _member_struct);
	map<int, vector<string>> sort_members_by_dominance_into_prob_fronts(map<int, vector<string>>& front_map, map<string, map<string, double>>& _member_struct);
	map<string, double> get_mopso_fitness(vector<string> members, map<string, map<string, double>>& _member_struct);
	pair<map<string, double>, map<string, double>> get_spea2_fitness(map<string, map<string, double>>& _member_struct);

	void fill_domination_containers(map<string, map<string, double>>& _member_struct, map<string,
		vector<string>>&solutions_dominated_map, map<string, int>& num_dominating_map, bool dup_as_dom=false);

	bool compare_two_nsga(string& first, string& second);
	bool compare_two_bgo(string& first, string& second);
	bool compare_two_spea(string& first, string& second);

	//sort all members in member struct
	//map<string, double> get_cuboid_crowding_distance();
	map<string, double> get_cuboid_crowding_distance(map<string, map<string, double>>& _member_struct);
	map<string, double> get_cuboid_crowding_distance(vector<string>& members, map<string, map<string, double>>& _member_struct);

	map<string, double> get_spea2_kth_nn_crowding_distance(map<string, map<string, double>>& _member_struct);
	map<string, double> get_spea2_kth_nn_crowding_distance(vector<string>& members, map<string, map<string, double>>& _member_struct);	
	map<string, double> get_cuboid_crowding_distance(ObservationEnsemble& oe, ParameterEnsemble& dp);

	vector<double> get_euclidean_distance(map<string, double> first, map<string, double> second);
	double get_euclidean_fitness(double E, double V);
	map<string, double> get_cluster_crowding_fitness(vector<string>& members, map<string, map<string, double>>& _member_struct);

	map<int, string> sort_members_of_bgo_ensemble(map<string, map<string, double>>& _ensemble_struct, 
		ParameterEnsemble& dp, ObservationEnsemble& _op, Constraints* constraints_ptr, double lambda);

	map<string, map<string, double>> member_struct;
	vector<string>* obj_names_ptr;
	vector<string>* obs_obj_names_ptr;
	vector<string>* obs_obj_sd_names_ptr;
	vector<string>* pi_obj_names_ptr;
	vector<string>* pi_obj_sd_names_ptr;
	map<string, double>* obj_dir_mult_ptr;
	set<string> duplicates;

	map<string, map<string, double>> feas_member_struct;
	map<int, vector<string>> front_map;
	map<int, vector<string>> prob_front_map;
	map<string, double> crowd_map, expected_crowd_map, var_crowd_map, fitness_map, probnondom_map, min_sd, nn_map;
	map<string, int> member_front_map, bgo_member_repo_map;
	map<string, double> member_cvar;
	map<string, double> infeas;
	vector<string> infeas_ordered;
	map<string, double> enbgo_fitness_map;
	map<string, double> spea2_constrained_fitness_map;
	map<string, double> spea2_unconstrained_fitness_map;
	

	//PPD-related stuff
	double dominance_probability(map<string, double>& first, map<string, double>& second);
	double dominance_prob_adhoc(map<string, double>& first, map<string, double>& second);
	double nondominance_probability(map<string, double>& first, map<string, double>& second);
	bool prob_pareto = false, ppd_sort, bgo = false;
	double ppd_beta;
	vector<double> ppd_range;

	//EHVI-related stuff
	double std_norm_df(double x, double mu, double sd, bool cumdf);
	double psi_function(double aa, double bb, double mu, double sd);
	map<string, double> ehvi_member_map;
	map<string, map<string, double>> incumbent_front_extreme, expdist_lookup, fit_lookup;
	map<int, vector<double>> hypervolume_partitions;
	double EHVI;
	int iter, inner_iter;
	double get_ehvi(string& member, map<string, map<string, double>>& _member_struct);

	double c_opt;
	map<string, map<string, double>> bgo_ensemble_struct;
	map<string, double> bgo_aqf_map, decspace_dist_map;
};


class MOEA
{
	
public:
	
	static mt19937_64 rand_engine;
	MOEA(Pest &_pest_scenario, FileManager &_file_manager, OutputFileWriter &_output_file_writer,
		PerformanceLog *_performance_log, RunManagerAbstract* _run_mgr_ptr);
	void initialize();
    void iterate_to_solution();
	void finalize();
	void process_bgo_objectives(bool bgomode = false);
	typedef pair<vector<string>, vector<string>> DomPair;
private:
	MouEnvType envtype;
	MouMateType mattype;
	double epsilon = 1.0e-15;
	Pest& pest_scenario;
	set<string> pp_args;
	vector<MouGenType> gen_types;
	vector<string> act_obs_names, act_par_names;
	int iter, inner_iter, warn_min_members, error_min_members;
	int member_count;
	int archive_size, infill_size, infill_pool_size;
	string population_dv_file, population_obs_restart_file;
	string dv_pop_file_tag = "dv_pop";
	string obs_pop_file_tag = "obs_pop";
	/*string dv_inner_file_tag = "inner.dv_pop";
	string obs_inner_file_tag = "inner.obs_pop";*/
	string training_dv_file_tag = "training.dv_pop";
	string training_obs_file_tag = "training.obs_pop";
	string lineage_tag = "lineage.csv";
	chancePoints chancepoints;
	FileManager &file_manager; 
	std::mt19937 rand_gen;
	vector<string> og_obj_names, obj_names, obs_obj_names, pi_obj_names, obs_obj_sd_names, pi_obj_sd_names;
	vector<string> dv_names;
	vector<string> defcmd_vec;
	map<string, double> obj_dir_mult;
	int n_adaptive_dvs;
	map<string, map<string, double>> previous_obj_summary, previous_dv_summary;
	bool risk_obj;
	bool prob_pareto = false, bgo = false;
	bool ppd_sort;
	int restart_iter_offset;
	int save_every, save_inner_every;
	map<int,int> population_schedule;

	ParetoObjectives objectives;
	Constraints constraints;
	const ParameterInfo *ctl_par_info_ptr;
	const ParameterGroupInfo *par_group_info_ptr;
	ParamTransformSeq par_transform;
	OutputFileWriter &output_file_writer;
	PerformanceLog *performance_log;
	RunManagerAbstract* run_mgr_ptr;
	const ObservationInfo *obs_info_ptr;

	ParameterEnsemble dp, dp_archive, dt, idp, idp_archive, dp_infill;
	ObservationEnsemble op, op_archive, ot, iop, iop_archive, op_infill;

	map<string,Eigen::VectorXd> par_sim_map, obs_sim_map, pso_velocity_map;

	ParameterEnsemble pso_velocity, pso_pbest_dp;
	ObservationEnsemble pso_pbest_op;

	void update_sim_maps(ParameterEnsemble& _dp, ObservationEnsemble& _op);
	void fill_populations_from_maps(ParameterEnsemble& new_dp, ObservationEnsemble& new_op );
	void fill_infill_ensemble(ParameterEnsemble& _dp, ObservationEnsemble& _op);

	void update_archive_bgo(ObservationEnsemble& _op, ParameterEnsemble& _dp);
	void update_archive_nsga(ObservationEnsemble& _op, ParameterEnsemble& _dp);
	void update_archive_spea(ObservationEnsemble& _op, ParameterEnsemble& _dp);

	void throw_moea_error(const string& message);

	template<typename T, typename A>
	void message(int level, const string& _message, vector<T, A> _extras, bool echo = true);
	void message(int level, const string& _message);
	template<typename T>
	void message(int level, const string& _message, T extra);

	void sanity_checks();
	vector<int> run_population(ParameterEnsemble& _dp, ObservationEnsemble& _op, bool allow_chance);
	vector<int> run_surrogate(ParameterEnsemble& _dp, ObservationEnsemble& _op);

	void queue_chance_runs(ParameterEnsemble& _dp);
	ObservationEnsemble get_chance_shifted_op(ParameterEnsemble& _dp, ObservationEnsemble& _op, string& opt_member);

	void initialize_pso();
    ParameterEnsemble get_initial_pso_velocities(int num_members);
    void update_pso_velocity_map(ParameterEnsemble& _pso_velocity);
    void initialize_population_schedule();
	void initialize_training_dataset();
	ParameterEnsemble reinitialize_dv_population();
	bool initialize_dv_population();
	void initialize_obs_restart_population();

	ParameterEnsemble generate_population(bool bgomode = false);

	ParameterEnsemble generate_diffevol_population(int num_members, ParameterEnsemble& _dp);
	ParameterEnsemble generate_sbx_population(int num_members, ParameterEnsemble& _dp);
	ParameterEnsemble generate_pm_population(int num_members, ParameterEnsemble& _dp);
	ParameterEnsemble generate_pso_population(int num_members, ParameterEnsemble& _dp, bool bgomode = false);
	ParameterEnsemble simplex_cceua_kn(ParameterEnsemble s, int k, int optbounds);																																		
    ParameterEnsemble generate_simplex_population(int num_members, ParameterEnsemble& _dp, ObservationEnsemble& _op);

	ParameterEnsemble get_updated_pso_velocty(ParameterEnsemble& _dp, vector<string>& gbest_solutions, bool bgomode = false);

	vector<string> get_pso_gbest_solutions(int num_reals, ParameterEnsemble& _dp, ObservationEnsemble& _op, bool bgomode = false);
	void update_pso_pbest(ParameterEnsemble& _dp, ObservationEnsemble& _op, bool bgomode = false);

	map<string, string> current_pso_lineage_map;

	vector<int> selection(int num_to_select, ParameterEnsemble& _dp, MouMateType& matetype);

	string get_new_member_name(string tag = string());

	void save_populations(ParameterEnsemble& _dp, ObservationEnsemble& _op, string tag = string());
	void save_training_dataset(ParameterEnsemble& _dp, ObservationEnsemble& _op, string tag = string());
	void save_inner_population(ParameterEnsemble& _dp, ObservationEnsemble& _op, string tag = string());
	void select_infills();

	void gauss_mutation_ip(ParameterEnsemble& _dp);
	pair<Eigen::VectorXd, Eigen::VectorXd> sbx(double probability, double eta_m, int idx1, int idx2);
	pair<Eigen::VectorXd, Eigen::VectorXd> sbx_new(double crossover_probability, double di, Eigen::VectorXd& parent1,
		Eigen::VectorXd parent2, vector<string>& _dv_names, Parameters& lbnd, Parameters& ubnd);
	Eigen::VectorXd hybrid_pm(Eigen::VectorXd& parent1, double mutation_probability, double disrupt_probabilty, 
		vector<string>& _dv_names, Parameters& lbnd, Parameters& ubnd);
	pair<double, double> get_betas(double v1, double v2, double distribution_index);
	void get_sbx_child_values(const double& p1, const double& p2, const double& lbnd, 
		const double& ubnd, const double& eta, double& rnd, double& c1, double& c2);


	pair<Parameters, Observations> get_optimal_solution(ParameterEnsemble& _dp, ObservationEnsemble& _oe, string& opt_member_name);

	map<string, map<string, double>> obj_func_report(ParameterEnsemble& _dp, ObservationEnsemble& _oe);
	map<string, map<string, double>> get_obj_func_summary_stats(ParameterEnsemble& _dp, ObservationEnsemble& _op);
	map<string, map<string, double>> obj_func_change_report(map<string, map<string, double>>& current_obj_summary);
    map<string, map<string, double>> decvar_report(ParameterEnsemble& _dp);
    map<string, map<string, double>> decvar_change_report(map<string, map<string, double>>& current_dv_summary);


	int get_max_len_obj_name();
	bool should_use_multigen();

	void queue_resample_runs(ParameterEnsemble& _dp); //outer iters

	void get_current_true_solution();

};

#endif //MOEA_H_
