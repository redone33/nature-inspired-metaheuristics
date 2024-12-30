import argparse
import algorithms.TGA
import algorithms.WCA
import visualizer
import benchmark
import algorithms

def parse_args():
    parser = argparse.ArgumentParser(description="A script to visualize benchmark functions.")

    # Add the --run flag to execute the algorithm
    parser.add_argument(
        "--run", "-run",
        action="store_true",
        help="Run the specified algorithm on the specified function."
    )

    # Add the --visualize (-vis) flag
    parser.add_argument(
        "--visualize", "-vis",
        action="store_true",
        help="Enable visualization of the specified function."
    )
    
    # Add the --function_name (-fname) argument
    parser.add_argument(
        "--function_name", "-fname",
        type=str,
        choices=["sphere_func", "elliptic_func", "elliptic_rot_func", 
                 "schwefel_func", "rosenbrock_func", "rastrigin_func", 
                 "rastrigin_rot_func", "ackley_func", "ackley_rot_func"],  # Add more function names if needed
        help="Name of the function to visualize."
    )
    
    # Add the --algorithm (-alg) argument
    parser.add_argument(
        "--algorithm", "-alg",
        type=str,
        choices=["WCA", "TGA"],  # Add other algorithm names if needed
        help="Name of the algorithm to run on the specified function."
    )

    # Add the --real_time (-rt) argument
    parser.add_argument(
        "--real_time", "-rt",
        action="store_true",
        help="Run real time animation of algorithm running on function."
    )

    # Add the -gif argument
    parser.add_argument(
        "--gif", "-gif",
        action="store_true",
        help="Create gif of algorithm running on function."
    )

    return parser.parse_args()


def run(args):
    if args.visualize and args.run:
        print("Cannot use --visualize and --run together. Please choose one.")
        return

    if args.real_time and args.gif:
        print("Cannot use --real_time and --gif together. Please choose one.")
        return

    if args.run:
        if not args.algorithm:
            print("Please specify an algorithm with --algorithm or -alg when using --run.")
        else:
            run_algorithm(args.function_name, args.algorithm)
    
    if args.visualize:
        # Dynamically get the function from benchmark module
        func = getattr(benchmark, args.function_name, None)

        if func:
            if not args.algorithm:
                print(f"Visualizing {args.function_name}...")
                visualizer.visualize(func, None)
            else:
                if args.algorithm:
                    if args.gif:
                        visualizer.visualize(func, args.algorithm, gif=1)
                    elif args.real_time:
                        visualizer.visualize(func, args.algorithm, real_time=1)
                    else:
                        print(f"Please use --real_time or --gif ")             
        else:
            print(f"Function '{args.function_name}' is not recognized in the benchmark module.")


def run_algorithm(function_name, algorithm):
    # Dynamically get the function from benchmark module
    func = getattr(benchmark, function_name, None)
    if not func:
        print(f"Function '{function_name}' is not recognized in the benchmark module.")
        return
    
    algorithm_name = getattr(algorithms, algorithm, None)
    if not algorithm_name:
        print(f"Algorithm '{algorithm_name}' is not recognized in the algorithms module.")
        return

    if algorithm == "WCA":
        print(f"Running WCA on {function_name}...")
        config = {"LB": -5, "UB": 5, "nvars": 2, "npop": 50, "nsr": 4, "dmax": 1e-16, "max_it": 100}
        algorithms.WCA.wca(func, config, gif=False, real_time=False)
    elif algorithm == "TGA":
        print(f"Running TGA on {function_name}...")
        config = {"LB": -5, "UB": 5, "nvars": 2, "npop": 100, "N1": 40, "N2": 40, "N3": 20, "N4": 30, "lambda": 0.5, "theta": 1.1, "max_it": 100}
        algorithms.TGA.tga(func, config, gif=False, real_time=False) 
    else:
        print(f"Algorithm '{algorithm}' is not recognized.")

def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()