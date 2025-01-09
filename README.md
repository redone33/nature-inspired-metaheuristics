# Nature-inspired Metaheuristics

- Just a playground repo for nature inspired metaheuristics

> [!WARNING]
> This software is unfinished.

## Idea

- Write a simple program to run and visualize some of nature inspired algorithms in 3D.

## Quick start

```console
$ python main.py -run -fname sphere_func -alg WCA
$ python main.py --visualize --function_name sphere_func
$ python main.py --visualize --function_name sphere_func -alg WCA -gif
$ python main.py --visualize --function_name sphere_func -alg WCA -rt
```

## Demo

- You can generate .gif file using -gif arg

```console
$ python main.py --visualize --function_name spehere_func -alg WCA -gif
```

![alt-text](gifs/WCA.gif)


- Feel free to add more benchmark functions and algorithms

## Benchmark functions

- sphere_func, elliptic_func, schwefel_func, rosenbrock_func, rastrigin_func, ackley_rot_func

## Implemented algorithms:

- WCA (Water Cycle Algorithm)
- TGA (Tree Growth Algorithm)
- MBO (Monarch Butterfly Optimization)

## References

- https://www.sciencedirect.com/science/article/pii/S2352711016300024 - WCA (Water Cycle Algorithm)
- https://www.researchgate.net/publication/320009185_Tree_Growth_Algorithm_TGA_An_Effective_Metaheuristic_Algorithm_Inspired_by_trees'_behavior - TGA (Tree Growth Algorithm)
- https://link.springer.com/article/10.1007/s00521-015-1923-y - MBO (Monarch Butterfly Optimization)