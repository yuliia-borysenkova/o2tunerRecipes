import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from o2tuner.system import run_command
from o2tuner.utils import annotate_trial
from o2tuner.optimise import optimise, needs_cwd
from o2tuner.io import dump_yaml
from os import makedirs
from os.path import exists


def build_plot(x_arr,y_arr, title, line_label, xlabel, ylabel, savedir, type = "trials") :

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if type == "trials" :
        plt.plot(x_arr, y_arr, color = 'k', label=line_label, ls = '-', lw = 1, marker = '.', mec = 'k', mew = 0.5, mfc = "c", ms = 7)
        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(0, len(x_arr)+1, len(x_arr)*20/100)
        minor_ticks_x = np.arange(0, len(x_arr), len(x_arr)*5/100)

        major_ticks_y = np.arange(min(y_arr)-0.1, max(y_arr)+0.1, len(y_arr)*0.1/100)
        minor_ticks_y = np.arange(min(y_arr)-0.1, max(y_arr)+0.1, len(y_arr)*0.05/100)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)

        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
    
    if type == "variable"  :
        plt.plot(x_arr, y_arr, color = 'k', label=line_label, ls = 'None', lw = 1, marker = '.', mec = 'k', mew = 0.5, mfc = "c", ms = 7)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.savefig(savedir)
    plt.clf()

def best_trial(insp, variables) :
    arg_min = np.argmin(insp.get_losses())
    return [arg_min, insp.get_losses()[arg_min]] + [insp.get_annotation_per_trial(var)[arg_min] for var in variables]

def def_plots(insp, def_plots_dir, param_name) :
    figure, _ = insp.plot_slices()
    figure.savefig(def_plots_dir + param_name + "slices.png")
    print(def_plots_dir + param_name + "slices.png")
    plt.close(figure)

    figure, _ = insp.plot_importance()
    figure.savefig(def_plots_dir + param_name + "importance.png")
    plt.close(figure)

    figure, _ = insp.plot_parallel_coordinates()
    figure.savefig(def_plots_dir + param_name + "parallel_coordinates.png")
    plt.close(figure)

    figure, _ = insp.plot_correlations()
    figure.savefig(def_plots_dir + param_name + "parameter_correlations.png")
    plt.close(figure)

    figure, _ = insp.plot_pairwise_scatter()
    figure.savefig(def_plots_dir + param_name + "pairwise_scatter.png")
    plt.close(figure)

def read_param(path):
    search_strings = ["Efficiency for vertices :",
                      "Persentage of dublicated vertices :", "Persentage of skipped vertices :","Persentage of vertices with purity >= 0.7  :","Persentage of vertices with purity < 0.7 :","Mean Purity :"]
    extract_start = [len(search_string_line.split())
                     for search_string_line in search_strings]
    parameters = []
    with open(path, "r", encoding="utf8") as output_file:
        for line in output_file:
            for i in range(len(search_strings)):
                if search_strings[i] in line:
                    line = line.strip().split()
                    parameters.append(float(line[extract_start[i]]))
    if not parameters:
        print("ERROR: Could not extract parameters")
        sys.exit(1)
    return parameters, search_strings

def loss() :
    files_needed_for_script= ["/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2sim_geometry-aligned.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/sgn_1_Kine.root", "/media/work/software/baseline/optuna_scripts/CheckVertices.C"]
    aliases_scr = ["o2sim_geometry-aligned.root", "o2sim_Kine.root", "CheckVertices.C"]
    aliases_line_scr = [f" ln -s {files_needed_for_script[i]} {aliases_scr[i]} ;" for i in range(len(aliases_scr))]

    cmd_macro = f"{''.join( aliases_line_scr )} root -l -b -q CheckVertices.C++"
    _, log_file = run_command(cmd_macro, log_file="macro_log.log")
    parameters, parameter_names = read_param(log_file)
    
    #eff, purity, pur>0.7, pur<0.7, mixed
    loss_ = [1-parameters[0], 1-parameters[5], 1-parameters[3], parameters[4], (parameters[1] + parameters[2] + (1-parameters[0])+ (1-parameters[5]) + (1-parameters[3]))/5]

    return loss_

@needs_cwd
def objective(trial, config): 

    param_phiCut = trial.suggest_uniform("phiCut", 0, np.pi/4) #default 0.5
    #param_tanLambdaCut = trial.suggest_uniform("tanLambdaCut", 0, np.tan(np.pi/8)) #default 0.2
    param_clusterContributorsCut = trial.suggest_int(
        "clusterContributorsCut", 2, 20) #default 3
    
    annotate_trial(trial, "phiCut", param_phiCut)
    #annotate_trial(trial, "tanLambdaCut", param_tanLambdaCut)
    annotate_trial(trial, "NContr", param_clusterContributorsCut)

    param_tanLambdaCut = 0.2

    files_needed_for_reco = ["/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2clus_its.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/matbud.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/sgn_1_grp.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2simdigitizerworkflow_configuration.ini","/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/itsdigits.root"]
    aliases = ["o2clus_its.root", "matbud.root", "o2sim_grp.root", "o2simdigitizerworkflow_configuration.ini", "itsdigits.root"]
    aliases_line = [f" ln -s {files_needed_for_reco[i]} {aliases[i]} ;" for i in range(len(aliases))]

    cmd = f"{''.join( aliases_line )} o2-its-reco-workflow --trackerCA --tracking-mode async -b --run --configKeyValues \"HBFUtils.orbitFirstSampled=0;HBFUtils.nHBFPerTF=256;HBFUtils.orbitFirst=0;HBFUtils.runNumber=300000;HBFUtils.startTime=1546300800000;ITSVertexerParam.phiCut={param_phiCut};ITSVertexerParam.clusterContributorsCut={param_clusterContributorsCut};ITSVertexerParam.tanLambdaCut={param_tanLambdaCut};NameConf.mDirMatLUT=..\""
    run_command(cmd)

    if "loss_name" not in config or config["loss_name"] == "efficiency":
        return loss()[0]
    if config["loss_name"] == "purity":
        return loss()[1]
    if config["loss_name"] == "purity>0.7":
        return loss()[2]
    if config["loss_name"] == "purity<0.7":
        return loss()[3]
    if config["loss_name"] == "mixed":
        return loss()[4]

def objective_Phi(trial, config): 

    param_phiCut = trial.suggest_uniform("phiCut", 0, np.pi/4) #default 0.5
    #param_tanLambdaCut = trial.suggest_uniform("tanLambdaCut", 0, np.tan(np.pi/8)) #default 0.2
    #param_clusterContributorsCut = trial.suggest_int("clusterContributorsCut", 2, 20) #default 3
    
    annotate_trial(trial, "phiCut", param_phiCut)
    #annotate_trial(trial, "tanLambdaCut", param_tanLambdaCut)
    #annotate_trial(trial, "NContr", param_clusterContributorsCut)

    param_tanLambdaCut = 0.2
    param_clusterContributorsCut = 3

    files_needed_for_reco = ["/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2clus_its.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/matbud.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/sgn_1_grp.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2simdigitizerworkflow_configuration.ini","/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/itsdigits.root"]
    aliases = ["o2clus_its.root", "matbud.root", "o2sim_grp.root", "o2simdigitizerworkflow_configuration.ini", "itsdigits.root"]
    aliases_line = [f" ln -s {files_needed_for_reco[i]} {aliases[i]} ;" for i in range(len(aliases))]

    cmd = f"{''.join( aliases_line )} o2-its-reco-workflow --trackerCA --tracking-mode async -b --run --configKeyValues \"HBFUtils.orbitFirstSampled=0;HBFUtils.nHBFPerTF=256;HBFUtils.orbitFirst=0;HBFUtils.runNumber=300000;HBFUtils.startTime=1546300800000;ITSVertexerParam.phiCut={param_phiCut};ITSVertexerParam.clusterContributorsCut={param_clusterContributorsCut};ITSVertexerParam.tanLambdaCut={param_tanLambdaCut};NameConf.mDirMatLUT=..\""
    run_command(cmd)

    if "loss_name" not in config or config["loss_name"] == "efficiency":
        return loss()[0]
    if config["loss_name"] == "purity":
        return loss()[1]
    if config["loss_name"] == "purity>0.7":
        return loss()[2]
    if config["loss_name"] == "purity<0.7":
        return loss()[3]
    if config["loss_name"] == "mixed":
        return loss()[4]


def objective_NContr(trial, config): 

    #param_phiCut = trial.suggest_uniform("phiCut", 0, np.pi/4) #default 0.5
    #param_tanLambdaCut = trial.suggest_uniform("tanLambdaCut", 0, np.tan(np.pi/8)) #default 0.2
    param_clusterContributorsCut = trial.suggest_int("clusterContributorsCut", 2, 10) #default 3
    
    #annotate_trial(trial, "phiCut", param_phiCut)
    #annotate_trial(trial, "tanLambdaCut", param_tanLambdaCut)
    annotate_trial(trial, "NContr", param_clusterContributorsCut)

    param_tanLambdaCut = 0.2
    #param_clusterContributorsCut = 3
    param_phiCut = 0.5

    files_needed_for_reco = ["/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2clus_its.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/matbud.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/sgn_1_grp.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2simdigitizerworkflow_configuration.ini","/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/itsdigits.root"]
    aliases = ["o2clus_its.root", "matbud.root", "o2sim_grp.root", "o2simdigitizerworkflow_configuration.ini", "itsdigits.root"]
    aliases_line = [f" ln -s {files_needed_for_reco[i]} {aliases[i]} ;" for i in range(len(aliases))]

    cmd = f"{''.join( aliases_line )} o2-its-reco-workflow --trackerCA --tracking-mode async -b --run --configKeyValues \"HBFUtils.orbitFirstSampled=0;HBFUtils.nHBFPerTF=256;HBFUtils.orbitFirst=0;HBFUtils.runNumber=300000;HBFUtils.startTime=1546300800000;ITSVertexerParam.phiCut={param_phiCut};ITSVertexerParam.clusterContributorsCut={param_clusterContributorsCut};ITSVertexerParam.tanLambdaCut={param_tanLambdaCut};NameConf.mDirMatLUT=..\""
    run_command(cmd)

    if "loss_name" not in config or config["loss_name"] == "efficiency":
        return loss()[0]
    if config["loss_name"] == "purity":
        return loss()[1]
    if config["loss_name"] == "purity>0.7":
        return loss()[2]
    if config["loss_name"] == "purity<0.7":
        return loss()[3]
    if config["loss_name"] == "mixed":
        return loss()[4]


def evaluate(inspectors, config):
    insp_arr = []
    for insp in inspectors:
        insp_arr.append(insp)

    p_list = ["Efficiency","Purity","Purity>0.7","Mixed"]
    with open('/media/work/software/baseline/optuna_scripts/Data/param_Phi_NContr.txt', 'w') as fp:
        for i in range (len(p_list)) :
            fp.write(f"_______________________________________{p_list[i]}_______________________________________\n")
            fp.write("Losses \t phiCut \t NContr\n")
            for j in range (len(insp_arr[i].get_losses())) :
                temp_ = insp_arr[i].get_annotation_per_trial("phiCut")[j]
                temp__ = insp_arr[i].get_annotation_per_trial("NContr")[j]
                fp.write(f"{round((insp_arr[i].get_losses())[j],6)} \t {round(temp_,6)} \t {temp__}\n")

    #Plots
    build_plot(np.arange(len(insp_arr[0].get_losses())), insp_arr[0].get_losses(),"Loss (eff) to N_trial", "1-Efficiency", "N_Trial", "Loss = 1 - Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Eff_NumTrial.png")
    build_plot(np.arange(len(insp_arr[1].get_losses())), insp_arr[1].get_losses(),"Loss (pur) to N_trial", "1-Purity", "N_Trial", "Loss = 1 - Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur_NumTrial.png")
    build_plot(np.arange(len(insp_arr[2].get_losses())), insp_arr[2].get_losses(),"Loss (pur>0.7) to N_trial", "1-Purity(>0.7)", "N_Trial", "Loss = 1 - Purity(>0.7)","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur(greater_than_0.7)_NumTrial.png")
    build_plot(np.arange(len(insp_arr[3].get_losses())), insp_arr[3].get_losses(),"Loss (mix) to N_trial", "((1-Pur) + (1-Eff) + (1-Per>0.7) + Pers_skip + Pers_dub)/5", "N_Trial", "Loss = ((1-Pur) + (1-Eff) + (1-Per>0.7) + Pers_skip + Pers_dub)/5","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Mix_NumTrial.png")
    
    build_plot(np.arange(len(insp_arr[0].get_losses())), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to N_trial", "Efficiency", "N_Trial", "Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_NumTrial.png")
    build_plot(np.arange(len(insp_arr[1].get_losses())), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to N_trial", "Purity", "N_Trial", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_NumTrial.png")

    build_plot(insp_arr[0].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to Phi", "Efficiency", "Phi", "Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_Phi.png","variable")
    build_plot(insp_arr[1].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to Phi", "Purity", "Phi", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_Phi.png","variable")
    build_plot(insp_arr[0].get_annotation_per_trial("NContr"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to NContr", "Efficiency", "NContr", "Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_NContr.png","variable")
    build_plot(insp_arr[1].get_annotation_per_trial("NContr"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to NContr", "Purity", "NContr", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_NContr.png","variable")


    #Analitics
    print("_________________________2_parameters (phi, NContr)_________________________")
    print("Name,\tnumber of trial,\tbest min value,\tphi,\tNContr")
    print("Efficiency", *best_trial(insp_arr[0], ["phiCut", "NContr"]), sep = ",\t")
    print("Purity", *best_trial(insp_arr[1], ["phiCut", "NContr"]), sep = ",\t")
    print("Mixed", *best_trial(insp_arr[3], ["phiCut", "NContr"]), sep = ",\t")

    #Default plots
    def_plots_dir = "/home/yborysen/work/software/baseline/optuna_scripts/Plots_inspector/"
    param_name = ["eff_","pur_","pur>0.7_","mix_"]
    
    print(insp_arr[0].get_most_important())
    def_plots(insp_arr[0], def_plots_dir, param_name[0])
    def_plots(insp_arr[1], def_plots_dir, param_name[1])

    return True

def evaluate_Phi(inspectors, config):
    insp_arr = []
    for insp in inspectors:
        insp_arr.append(insp)

    p_list = ["Efficiency","Purity"]
    with open('/media/work/software/baseline/optuna_scripts/Data/only_Phi.txt', 'w') as fp:
        for i in range (len(p_list)) :
            fp.write(f"_______________________________________{p_list[i]}_______________________________________\n")
            fp.write("Losses \t phiCut\n")
            for j in range (len(insp_arr[i].get_losses())) :
                temp_ = insp_arr[i].get_annotation_per_trial("phiCut")[j]
                fp.write(f"{round((insp_arr[i].get_losses())[j],6)} \t {round(temp_,6)}\n")

    #Plots
    build_plot(np.arange(len(insp_arr[0].get_losses())), insp_arr[0].get_losses(),"Loss (eff) to N_trial", "1-Efficiency", "N_Trial", "Loss = 1 - Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Eff_NumTrial_Phi_only.png")
    build_plot(np.arange(len(insp_arr[1].get_losses())), insp_arr[1].get_losses(),"Loss (pur) to N_trial", "1-Purity", "N_Trial", "Loss = 1 - Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur_NumTrial_Phi_only.png")

    build_plot(insp_arr[0].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to Phi (phi only)", "Efficiency", "Phi", "Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_Phi_Phi_only.png","variable")
    build_plot(insp_arr[1].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to Phi (phi only)", "Purity", "Phi", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_Phi_Phi_only.png","variable")

    #Analitics
    print("_________________________1_parameter (phi)_________________________")
    print("Name,\tnumber of trial,\tbest min value,\t phi")
    print("Efficiency", *best_trial(insp_arr[0], ["phiCut"]), sep = ",\t")
    print("Purity", *best_trial(insp_arr[1], ["phiCut"]), sep = ",\t")

    return True

def evaluate_NContr(inspectors, config):
    insp_arr = []
    for insp in inspectors:
        insp_arr.append(insp)

    p_list = ["Efficiency","Purity"]
    with open('/media/work/software/baseline/optuna_scripts/Data/only_NContr.txt', 'w') as fp:
        for i in range (len(p_list)) :
            fp.write(f"_______________________________________{p_list[i]}_______________________________________\n")
            fp.write("Losses \t NContr\n")
            for j in range (len(insp_arr[i].get_losses())) :
                temp_ = insp_arr[i].get_annotation_per_trial("NContr")[j]
                fp.write(f"{round((insp_arr[i].get_losses())[j],6)} \t {temp_}\n")


    build_plot(np.arange(len(insp_arr[0].get_losses())), insp_arr[0].get_losses(),"Loss (eff) to N_trial", "1-Efficiency", "N_Trial", "Loss = 1 - Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Eff_NumTrial_NContr_only.png")
    build_plot(np.arange(len(insp_arr[1].get_losses())), insp_arr[1].get_losses(),"Loss (pur) to N_trial", "1-Purity", "N_Trial", "Loss = 1 - Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur_NumTrial_NContr_only.png")

    build_plot(insp_arr[0].get_annotation_per_trial("NContr"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to NContr (NContr only)", "Efficiency", "NContr", "Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_NContr_NContr_only.png","variable")
    build_plot(insp_arr[1].get_annotation_per_trial("NContr"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to NContr (NContr only)", "Purity", "NContr", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_NContr_NContr_only.png","variable")

    #Analitics
    print("_________________________1_parameter (NContr)_________________________")
    print("Name,\tnumber of trial,\tbest min value,\t NContr")
    print("Efficiency", *best_trial(insp_arr[0], ["NContr"]), sep = ",\t")
    print("Purity", *best_trial(insp_arr[1], ["NContr"]), sep = ",\t")

    return True