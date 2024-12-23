import argparse
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
        choices=["sphere_func", "elliptic_func", "elliptic_rot_func", "schwefel_func", 
                 "rosenbrock_func", "rastrigin_func", "rastrigin_rot_func", "ackley_func", "ackley_rot_func"],  # Add more function names if needed
        help="Name of the function to visualize."
    )
    
    # Add the --algorithm (-alg) argument
    parser.add_argument(
        "--algorithm", "-alg",
        type=str,
        choices=["WCA"],  # Add other algorithm names if needed
        help="Name of the algorithm to run on the specified function."
    )

    return parser.parse_args()

def run_algorithm(function_name, algorithm):
    # Dynamically get the function from benchmark module
    func = getattr(benchmark, function_name, None)
    if not func:
        print(f"Function '{function_name}' is not recognized in the benchmark module.")
        return
    
    if algorithm == "WCA":
        print(f"Running WCA on {function_name}...")
        algorithms.WCA.wca(func, -5, 5, nvars=2)  # Adjust parameters as needed for WCA
    else:
        print(f"Algorithm '{algorithm}' is not recognized.")

def run(args):

    if args.visualize and args.run:
        print("Cannot use --visualize and --run together. Please choose one.")
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
            print(f"Visualizing {args.function_name}...")
            visualizer.visualize(func, None)
        else:
            print(f"Function '{args.function_name}' is not recognized in the benchmark module.")

    '''if args.visualize:
        # Dynamically get the function from benchmark module
        func = getattr(benchmark, args.function_name, None)
        
        if func:
            print(f"Visualizing {args.function_name}...")
            visualizer.visualize(func, None)
        else:
            print(f"Function '{args.function_name}' is not recognized in the benchmark module.")
    else:
        print("Visualization flag is not set. Use --visualize or -vis to enable it.")'''

def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()