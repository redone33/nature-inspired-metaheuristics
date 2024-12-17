import argparse
import visualizer
import benchmark

def parse_args():
    parser = argparse.ArgumentParser(description="A script to visualize benchmark functions.")

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
    
    return parser.parse_args()

def run(args):
    if args.visualize:
        # Dynamically get the function from benchmark module
        func = getattr(benchmark, args.function_name, None)
        
        if func:
            print(f"Visualizing {args.function_name}...")
            visualizer.visualize(func, None)
        else:
            print(f"Function '{args.function_name}' is not recognized in the benchmark module.")
    else:
        print("Visualization flag is not set. Use --visualize or -vis to enable it.")

def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()