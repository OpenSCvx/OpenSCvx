import sys
import warnings
warnings.filterwarnings("ignore")
import queue
import time

from termcolor import colored

# Define colors for printing
col_main = "blue"
col_pos = "green"
col_neg = "red"

def intro():
    # Silence syntax warnings
    warnings.filterwarnings("ignore")
    # fmt: off
    ascii_art = '''
                             
                            ____                    _____  _____           
                           / __ \                  / ____|/ ____|          
                          | |  | |_ __   ___ _ __ | (___ | |  __   ____  __
                          | |  | | '_ \ / _ \ '_ \ \___ \| |  \ \ / /\ \/ /
                          | |__| | |_) |  __/ | | |____) | |___\ V /  >  < 
                           \____/| .__/ \___|_| |_|_____/ \_____\_/  /_/\_\ 
                                 | |                                       
                                 |_|                                       
---------------------------------------------------------------------------------------------------------
                                Author: Chris Hayner and Griffin Norris
                                    Autonomous Controls Laboratory
                                       University of Washington
---------------------------------------------------------------------------------------------------------
'''
    # fmt: on
    print(ascii_art)

def header():
    print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
        "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))
    print(colored("---------------------------------------------------------------------------------------------------------"))

def intermediate(print_queue, params):
    hz = 30.0
    while True:
        t_start = time.time()
        try:
            data = print_queue.get(timeout=1.0/hz)
            # remove bottom labels and line
            if not data["iter"] == 1:
                sys.stdout.write('\x1b[1A\x1b[2K\x1b[1A\x1b[2K')
            if data["prob_stat"][3] == 'f':
                # Only show the first element of the string
                data["prob_stat"] = data["prob_stat"][0]

            iter_colored = colored("{:4d}".format(data["iter"]))
            J_tot_colored = colored("{:.1e}".format(data["J_total"]))
            J_tr_colored = colored("{:.1e}".format(data["J_tr"]), col_pos if data["J_tr"] <= params.scp.ep_tr else col_neg)
            J_vb_colored = colored("{:.1e}".format(data["J_vb"]), col_pos if data["J_vb"] <= params.scp.ep_vb else col_neg)
            J_vc_colored = colored("{:.1e}".format(data["J_vc"]), col_pos if data["J_vc"] <= params.scp.ep_vc else col_neg)
            cost_colored = colored("{:.1e}".format(data["cost"]))
            prob_stat_colored = colored(data["prob_stat"], col_pos if data["prob_stat"] == 'optimal' else col_neg)

            print("{:^4} |     {:^6.2f}    |      {:^6.2F}     | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
                iter_colored, data["dis_time"], data["subprop_time"], J_tot_colored, J_tr_colored, J_vb_colored, J_vc_colored, cost_colored, prob_stat_colored))

            print(colored("---------------------------------------------------------------------------------------------------------"))
            print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
                "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))
        except queue.Empty:
            pass
        time.sleep(max(0.0, 1.0/hz - (time.time() - t_start)))

def footer(computation_time):
    print(colored("---------------------------------------------------------------------------------------------------------"))
    # Define ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print with bold text
    print("------------------------------------------------ " + BOLD + "RESULTS" + RESET + " ------------------------------------------------")
    print("Total Computation Time: ", computation_time)