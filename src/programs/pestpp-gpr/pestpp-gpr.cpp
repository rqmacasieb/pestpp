#include "RunManagerPanther.h" //needs to be first because it includes winsock2.h
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>
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
#include "gpr.h"
#include "covariance.h"
#include "sequential_lp.h"
#include "PerformanceLog.h"
#include "debug.h"
#include "logger.h"

using namespace std;
using namespace pest_utils;

int main(int argc, char* argv[])
{
    string version = PESTPP_VERSION;
    cout << endl << endl;
    cout << "             pestpp-gpr: a tool for Gaussian Process Regression" << endl << endl;
    cout << "                         by the PEST++ development team" << endl << endl << endl;
    cout << endl << endl << "version: " << version << endl;
    cout << "binary compiled on " << __DATE__ << " at " << __TIME__ << endl << endl;
    auto start = chrono::steady_clock::now();
    string start_string = get_time_string();
    cout << "started at " << start_string << endl;

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
    int flag = remove(rns_file.c_str());

    if (cmdline.runmanagertype == CmdLine::RunManagerType::EXTERNAL)
    {
        cerr << "External run manager ('/e') not supported by pestpp-gpr, please use panther instead" << endl;
        exit(1);
    }

    debug_initialize(file_manager.build_filename("dbg"));

    file_manager.open_default_files();
    ofstream& fout_rec = file_manager.rec_ofstream();
    PerformanceLog performance_log(file_manager.open_ofile_ext("log"));

    fout_rec << "              pestpp-gpr: a tool for Gaussian Process Regression" << endl;
    fout_rec << "                         by the PEST++ development team" << endl << endl << endl;
    fout_rec << endl;
    fout_rec << endl << endl << "version: " << version << endl;
    fout_rec << "binary compiled on " << __DATE__ << " at " << __TIME__ << endl << endl;
    fout_rec << "using control file: \"" << cmdline.ctl_file_name << "\"" << endl;
    fout_rec << "in directory: \"" << OperSys::getcwd() << "\"" << endl;
    fout_rec << "on host: \"" << w_get_hostname() << "\"" << endl;
    fout_rec << "started at " << start_string << endl << endl;

    cout << endl;
    cout << "using control file: \"" << cmdline.ctl_file_name << "\"" << endl;
    cout << "in directory: \"" << OperSys::getcwd() << "\"" << endl;
    cout << "on host: \"" << w_get_hostname() << "\"" << endl << endl;

    Pest pest_scenario;
    try {
        performance_log.log_event("starting to process control file");
        pest_scenario.process_ctl_file(file_manager.open_ifile_ext("pst"), file_manager.build_filename("pst"), fout_rec);
        file_manager.close_file("pst");
        performance_log.log_event("finished processing control file");
    }
    catch (PestError& e) {
        cerr << "Error processing control file: " << filename << endl << endl;
        cerr << e.what() << endl << endl;
        fout_rec << "Error processing control file: " << filename << endl << endl;
        fout_rec << e.what() << endl << endl;
        fout_rec.close();
        throw(e);
    }

    pest_scenario.check_inputs(fout_rec);

    pest_scenario.get_pestpp_options_ptr()->set_iter_summary_flag(false);
    OutputFileWriter output_file_writer(file_manager, pest_scenario, false);
    output_file_writer.scenario_report(fout_rec, false);

    if (pest_scenario.get_pestpp_options().get_ies_verbose_level() > 1)
    {
        output_file_writer.scenario_pargroup_report(fout_rec);
        output_file_writer.scenario_par_report(fout_rec);
        output_file_writer.scenario_obs_report(fout_rec);
    }

    if (pest_scenario.get_pestpp_options().get_debug_parse_only())
    {
        cout << endl << endl << "DEBUG_PARSE_ONLY is true, exiting..." << endl << endl;
        exit(0);
    }

    RunManagerAbstract* run_manager_ptr;

    if (cmdline.runmanagertype == CmdLine::RunManagerType::PANTHER_MASTER)
    {
        if (pest_scenario.get_control_info().noptmax == 0)
        {
            cout << endl << endl << "WARNING: 'noptmax' = 0 but using parallel run mgr. This prob isn't what you want to happen..." << endl << endl;
        }

        performance_log.log_event("initializing panther run manager");
        run_manager_ptr = new RunManagerPanther(
            rns_file, cmdline.panther_port,
            file_manager.open_ofile_ext("rmr"),
            pest_scenario.get_pestpp_options().get_max_run_fail(),
            pest_scenario.get_pestpp_options().get_overdue_reched_fac(),
            pest_scenario.get_pestpp_options().get_overdue_giveup_fac(),
            pest_scenario.get_pestpp_options().get_overdue_giveup_minutes(),
            pest_scenario.get_pestpp_options().get_panther_echo());
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
    Parameters cur_ctl_parameters = pest_scenario.get_ctl_parameters();
    run_manager_ptr->initialize(base_trans_seq.ctl2model_cp(cur_ctl_parameters), pest_scenario.get_ctl_observations());

    performance_log.log_event("initializing GP object");
    GP gp(pest_scenario, file_manager, output_file_writer, &performance_log, run_manager_ptr);

    try {
        int q = pest_utils::quit_file_found();
        if ((q == 1) || (q == 2))
        {
            cout << "'pest.stp' found, quitting" << endl;
            fout_rec << "'pest.stp' found, quitting" << endl;
        }
        else
        {
            if (q == 4) {
                cout << "...pest.stp found with '4'. run mgr has returned control, removing file." << endl;
                fout_rec << "...pest.stp found with '4'. run mgr has returned control, removing file." << endl;

                if (!pest_utils::try_remove_quit_file()) {
                    cout << "...error removing pest.stp file, bad times ahead..." << endl;
                    fout_rec << "...error removing pest.stp file, bad times ahead..." << endl;
                }
            }

            performance_log.log_event("starting GPR solution");
            gp.iterate_to_solution();

            performance_log.log_event("finalizing GPR solution");
            gp.finalize();
        }
    }
    catch (const exception& e) {
        performance_log.log_event(string("error during GPR analysis: ") + e.what());
        cerr << "Error during GPR analysis: " << e.what() << endl;
        fout_rec << "Error during GPR analysis: " << e.what() << endl;
        delete run_manager_ptr;
        return 1;
    }

    delete run_manager_ptr;

    auto end = chrono::steady_clock::now();
    cout << endl << endl << "pestpp-gpr analysis complete..." << endl;
    fout_rec << endl << endl << "pestpp-gpr analysis complete..." << endl;
    cout << "started at " << start_string << endl;
    cout << "finished at " << get_time_string() << endl;
    cout << "took " << setprecision(6) << (double)chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0 << " minutes" << endl;
    cout << flush;
    fout_rec << "started at " << start_string << endl;
    fout_rec << "finished at " << get_time_string() << endl;
    fout_rec << "took " << setprecision(6) << (double)chrono::duration_cast<chrono::seconds>(end - start).count() / 60.0 << " minutes" << endl;
    fout_rec.close();

    return 0;
}