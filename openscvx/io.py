import warnings
warnings.filterwarnings("ignore")

from termcolor import colored

def intro():
    # Silence syntax warnings
    warnings.filterwarnings("ignore")
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
    print(ascii_art)

def header():
    print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7} | {:^14}".format(
        "Iter", "Dis Time (ms)", "Solve Time (ms)", "J_total", "J_tr", "J_vb", "J_vc", "Cost", "Solver Status"))
    print(colored("---------------------------------------------------------------------------------------------------------"))

def footer(computation_time):
    print(colored("---------------------------------------------------------------------------------------------------------"))
    # Define ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print with bold text
    print("------------------------------------------------ " + BOLD + "RESULTS" + RESET + " ------------------------------------------------")
    print("Total Computation Time: ", computation_time)