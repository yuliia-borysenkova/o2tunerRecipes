import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

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

def buid_plot_quadrupole_axes(Ntrial_x_axe, loss_y_axe, phi_y_axe, NContr_y_axe, lambda_y_axe, xlabel, filename):
    fig, ax1 = plt.subplots()

    #ms = marker size, mfc = marker face color, mew = marker edge width, mec = marker edge color, lw = line width
    # ls = '-', lw = 1, marker = '.', mec = 'k', mew = 0.5, mfc = "c", ms = 7
    ax1.plot(Ntrial_x_axe, loss_y_axe, color="blue", ls = '-', lw = 0.5, marker = '.', mec = 'blue', mew = 0.5, mfc = "royalblue", ms = 7)
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(Ntrial_x_axe, phi_y_axe, color="green", ls = '-', lw = 0.5, marker = '.', mec = 'green', mew = 0.5, mfc = "springgreen", ms = 7)

    ax3 = ax1.twinx()
    ax3.plot(Ntrial_x_axe, NContr_y_axe, color="red", ls = '-', lw = 0.5, marker = '.', mec = 'red', mew = 0.5, mfc = "orangered", ms = 7)

    ax4 = ax1.twinx()
    ax4.plot(Ntrial_x_axe, lambda_y_axe, color="purple", ls = '-', lw = 0.5, marker = '.', mec = 'purple', mew = 0.5, mfc = "deeppink", ms = 7)

    #ax3.spines['right'].set_position(('outward',60))
    ax3.spines['right'].set_position(('axes',1.15))
    ax4.spines['right'].set_position(('axes',1.3))

    ax1.set_xlabel("Number of trial", color="black")
    ax1.set_ylabel(xlabel,color="blue")
    ax2.set_ylabel(r'$\Delta \bar \varphi$',color="green")
    ax3.set_ylabel(r'$N_{contrib}$',color="red")
    ax4.set_ylabel(r'$\Delta \tan \bar \lambda$',color="purple")

    ax1.tick_params(axis='y',colors="blue")
    ax2.tick_params(axis='y',colors="green")
    ax3.tick_params(axis='y',colors="red")
    ax4.tick_params(axis='y',colors="purple")

    ax2.spines['right'].set_color("green")
    ax3.spines['right'].set_color("red")
    ax3.spines['left'].set_color("blue")
    ax4.spines['right'].set_color("purple")

    fig.savefig(filename, bbox_inches='tight')
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
    search_strings = ["N_total_rec =","N_total_sim =","N_successful =","N_duplicated(counted one time) =","N_duplicated(total) =","N_miss =","N_fake =","N_good =","N_duplicated =","N_vertices_with_purity>=0.7 =","N_vertices_with_purity<0.7 =" ,"Efficiency for vertices =","Percentage of vertices with purity (>= 0.7)  =","Percentage of vertices with purity (< 0.7) =","Mean Purity ="]
    extract_start = [len(search_string_line.split())
                     for search_string_line in search_strings]
    parameters = []
    with open(path, "r", encoding="utf8") as output_file:
        for line in output_file:
            for i in range(len(search_strings)):
                if search_strings[i] in line:
                    line = line.strip().split()
                    parameters.append(float(line[extract_start[i]]))
    # print(parameters)
    # print(len(parameters))
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
    
    loss_ = [1-(parameters[7]/parameters[1]), 1-parameters[14], 1-parameters[0]/parameters[1], (1-(parameters[7]/parameters[1])+(1-parameters[14]))/2]

    return loss_

#define the loss functions and optimise them
@needs_cwd
def objective(trial, config): 

    param_phiCut = trial.suggest_uniform("phiCut", 0.000001, np.pi/4) #default 0.5
    param_tanLambdaCut = trial.suggest_uniform("tanLambdaCut", 0.000001, np.tan(np.pi/8)) #default 0.2
    param_clusterContributorsCut = trial.suggest_int("clusterContributorsCut", 2, 10) #default 3
    
    annotate_trial(trial, "phiCut", param_phiCut)
    annotate_trial(trial, "tanLambdaCut", param_tanLambdaCut)
    annotate_trial(trial, "NContr", param_clusterContributorsCut)

    files_needed_for_reco = ["/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2clus_its.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/matbud.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/sgn_1_grp.root", "/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/o2simdigitizerworkflow_configuration.ini","/media/work/software/baseline/optuna_scripts/test_run/simulation_1/tf1/itsdigits.root"]
    aliases = ["o2clus_its.root", "matbud.root", "o2sim_grp.root", "o2simdigitizerworkflow_configuration.ini", "itsdigits.root"]
    aliases_line = [f" ln -s {files_needed_for_reco[i]} {aliases[i]} ;" for i in range(len(aliases))]

    cmd = f"{''.join( aliases_line )} o2-its-reco-workflow --trackerCA --tracking-mode async -b --run --configKeyValues \"HBFUtils.orbitFirstSampled=0;HBFUtils.nHBFPerTF=256;HBFUtils.orbitFirst=0;HBFUtils.runNumber=300000;HBFUtils.startTime=1546300800000;ITSVertexerParam.phiCut={param_phiCut};ITSVertexerParam.clusterContributorsCut={param_clusterContributorsCut};ITSVertexerParam.tanLambdaCut={param_tanLambdaCut};NameConf.mDirMatLUT=..\""
    run_command(cmd)

    if "loss_name" not in config or config["loss_name"] == "efficiency_good":
        return loss()[0]
    if config["loss_name"] == "purity":
        return loss()[1]
    if config["loss_name"] == "efficiency_total":
        return loss()[2]
    if config["loss_name"] == "mixed":
        return loss()[3]

#Crearing plots and analyse the optimisation
def evaluate(inspectors, config):
    insp_arr = []
    for insp in inspectors:
        insp_arr.append(insp)

    #Printing the optimisation process for every trial into a file
    p_list = ["Efficiency_good","Purity","Efficiency_total","Mixed"]
    with open('/media/work/software/baseline/optuna_scripts/Data/param_Phi_NContr_lambda.txt', 'w') as fp:
        for i in range (len(p_list)) :
            fp.write(f"_______________________________________{p_list[i]}_______________________________________\n")
            fp.write("Losses \t phiCut \t NContr\t tanLambda\n")
            for j in range (len(insp_arr[i].get_losses())) :
                temp_ = insp_arr[i].get_annotation_per_trial("phiCut")[j]
                temp__ = insp_arr[i].get_annotation_per_trial("NContr")[j]
                temp___ = insp_arr[i].get_annotation_per_trial("tanLambdaCut")[j]
                fp.write(f"{round((insp_arr[i].get_losses())[j],6)} \t {round(temp_,6)} \t {temp__} \t {temp___}\n")

    #Plotting the dependence of parameters on losses and losses on Ntrial
    #build_plot(np.arange(len(insp_arr[0].get_losses())), insp_arr[0].get_losses(),"Loss (eff) to N_trial", "1-Efficiency", "N_Trial", "Loss = 1 - Efficiency","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Eff_NumTrial.png")
    #build_plot(np.arange(len(insp_arr[1].get_losses())), insp_arr[1].get_losses(),"Loss (pur) to N_trial", "1-Purity", "N_Trial", "Loss = 1 - Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur_NumTrial.png")
    #build_plot(np.arange(len(insp_arr[2].get_losses())), insp_arr[2].get_losses(),"Loss (pur>0.7) to N_trial", "1-Purity(>0.7)", "N_Trial", "Loss = 1 - Purity(>0.7)","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Pur(greater_than_0.7)_NumTrial.png")
    #build_plot(np.arange(len(insp_arr[3].get_losses())), insp_arr[3].get_losses(),"Loss (mix) to N_trial", "((1-Pur) + (1-Eff) + (1-Per>0.7) + Pers_skip + Pers_dub)/5", "N_Trial", "Loss = ((1-Pur) + (1-Eff) + (1-Per>0.7) + Pers_skip + Pers_dub)/5","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Loss_Mix_NumTrial.png")
    
    # build_plot(np.arange(len(insp_arr[0].get_losses())), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff_good to N_trial", "Efficiency_good", "N_Trial", "Efficiency_good","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_good_NumTrial.png")
    # build_plot(np.arange(len(insp_arr[1].get_losses())), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to N_trial", "Purity", "N_Trial", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_NumTrial.png")
    # build_plot(np.arange(len(insp_arr[2].get_losses())), np.full(len(insp_arr[2].get_losses()), 1)/insp_arr[2].get_losses(),"Eff_total to N_trial", "Efficiency_total", "N_Trial", "Efficiency_total","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_total_NumTrial.png")

    # build_plot(insp_arr[0].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff_good to Phi", "Efficiency_good", "Phi", "Efficiency_good","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_good_Phi.png","variable")
    # build_plot(insp_arr[1].get_annotation_per_trial("phiCut"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to Phi", "Purity", "Phi", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_Phi.png","variable")
    # build_plot(insp_arr[0].get_annotation_per_trial("NContr"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff_good to NContr", "Efficiency_good", "NContr", "Efficiency_good","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_good_NContr.png","variable")
    # build_plot(insp_arr[1].get_annotation_per_trial("NContr"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to NContr", "Purity", "NContr", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_NContr.png","variable")
    # build_plot(insp_arr[0].get_annotation_per_trial("tanLambdaCut"), np.full(len(insp_arr[0].get_losses()), 1)-insp_arr[0].get_losses(),"Eff to tanLambda", "Efficiency_good", "tanLambda", "Efficiency_good","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Eff_good_tanLambda.png","variable")
    # build_plot(insp_arr[1].get_annotation_per_trial("tanLambdaCut"), np.full(len(insp_arr[1].get_losses()), 1)-insp_arr[1].get_losses(),"Pur to tanLambda", "Purity", "tanLambda", "Purity","/home/yborysen/work/software/baseline/optuna_scripts/Plots/Pur_tanlambda.png","variable")
    
    #Plotting the optimisation process Ntrial -> Loss, {parameters}
    #plot with 4 y axes
    #efficiency
    loss_y_axe = np.full(len(insp_arr[0].get_losses()),1) - insp_arr[0].get_losses()
    phi_y_axe = insp_arr[0].get_annotation_per_trial("phiCut")
    NContr_y_axe = insp_arr[0].get_annotation_per_trial("NContr")
    lambda_y_axe = insp_arr[0].get_annotation_per_trial("tanLambdaCut")
    Ntrial_x_axe = np.arange(len(insp_arr[0].get_losses()))

    buid_plot_quadrupole_axes(Ntrial_x_axe, loss_y_axe, phi_y_axe, NContr_y_axe, lambda_y_axe, r"$Efficiency_{good}$", "/home/yborysen/work/software/baseline/optuna_scripts/Plots/Quadrupole_axes_efficiency_good")

    #purity
    loss_y_axe = np.full(len(insp_arr[1].get_losses()),1) - insp_arr[1].get_losses()
    phi_y_axe = insp_arr[1].get_annotation_per_trial("phiCut")
    NContr_y_axe = insp_arr[1].get_annotation_per_trial("NContr")
    lambda_y_axe = insp_arr[1].get_annotation_per_trial("tanLambdaCut")
    Ntrial_x_axe = np.arange(len(insp_arr[1].get_losses()))

    buid_plot_quadrupole_axes(Ntrial_x_axe, loss_y_axe, phi_y_axe, NContr_y_axe, lambda_y_axe, "Purity", "/home/yborysen/work/software/baseline/optuna_scripts/Plots/Quadrupole_axes_purity")

    #Efficiency_total
    loss_y_axe = np.full(len(insp_arr[2].get_losses()),1) - insp_arr[2].get_losses()
    phi_y_axe = insp_arr[2].get_annotation_per_trial("phiCut")
    NContr_y_axe = insp_arr[2].get_annotation_per_trial("NContr")
    lambda_y_axe = insp_arr[2].get_annotation_per_trial("tanLambdaCut")
    Ntrial_x_axe = np.arange(len(insp_arr[2].get_losses()))

    buid_plot_quadrupole_axes(Ntrial_x_axe, loss_y_axe, phi_y_axe, NContr_y_axe, lambda_y_axe, r"$Efficiency_{total}$", "/home/yborysen/work/software/baseline/optuna_scripts/Plots/Quadrupole_axes_efficiency_total")

    #Mixed
    loss_y_axe = np.full(len(insp_arr[3].get_losses()),1) - insp_arr[3].get_losses()
    phi_y_axe = insp_arr[3].get_annotation_per_trial("phiCut")
    NContr_y_axe = insp_arr[3].get_annotation_per_trial("NContr")
    lambda_y_axe = insp_arr[3].get_annotation_per_trial("tanLambdaCut")
    Ntrial_x_axe = np.arange(len(insp_arr[3].get_losses()))

    buid_plot_quadrupole_axes(Ntrial_x_axe, loss_y_axe, phi_y_axe, NContr_y_axe, lambda_y_axe, r"$(Efficiency_{good} + Purity$)/2", "/home/yborysen/work/software/baseline/optuna_scripts/Plots/Quadrupole_axes_mixed")

    #Printing best values
    print("_________________________3_parameters (phi, NContr, tanlambda)_________________________")
    print("Name,\tnumber of trial,\tbest min value,\tphi,\tNContr,\ttanlambda")
    print("Efficiency_good", *best_trial(insp_arr[0], ["phiCut", "NContr", "tanLambdaCut"]), sep = ",\t")
    print("Purity", *best_trial(insp_arr[1], ["phiCut", "NContr", "tanLambdaCut"]), sep = ",\t")
    print("Efficiency_total", *best_trial(insp_arr[2], ["phiCut", "NContr", "tanLambdaCut"]), sep = ",\t")
    print("Mixed", *best_trial(insp_arr[3], ["phiCut", "NContr", "tanLambdaCut"]), sep = ",\t")

    return True