# Genetic Algorithm Creature Climbing Simulation

This project implements a Genetic Algorithm (GA) to evolve virtual creatures that can climb a hill. The simulation is designed to optimize the creatures' ability to reach the highest possible point on a 2D landscape using a variety of parameters and fitness functions.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Description

This project simulates an evolutionary process in which virtual creatures attempt to climb a hill. The creatures are controlled by a Proportional-Derivative (PD) motor controller, and their physical traits (such as limb size, movement rate, and mutation rate) evolve over time based on fitness functions. The goal is to maximize the height climbed.

The genetic algorithm optimizes the creatures’ behavior through iterations, using various parameters and mutation rates. Fitness is determined based on how close the creature is to the target and how much of the landscape it traverses. Two sets of parameters are used for comparison to analyze their effectiveness.

## Features

- **Genetic Algorithm:** Customizable parameters, such as population size, mutation rates, and iteration count.
- **PD Motor Control:** Proportional-Derivative controller for smooth movement of creatures.
- **2D Simulation:** Visual representation of creature evolution over time.
- **Dynamic Fitness Function:** Combines distance to target and height climbed to evolve efficient climbers.
- **Real-time Sensory Input:** Allows creatures to adjust behavior dynamically based on environment.
- **Parameter Comparison:** Comparison of multiple sets of parameters to evaluate optimal evolution strategy.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/Qixuan-Codes/Genetic-Algorithm.git
    ```

2. Navigate to the project directory:

    ```
    cd genetic-algorithm
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

To run the simulation, execute the following command:

```
python genetic_algorithm.py
```

This will begin the simulation, and you'll see the creatures evolve over time. Fitness graphs will be generated and saved to compare the performance of different parameter sets.

## Results

Graphs of fitness over time are generated to visualize the optimization process. The fittest creature with the best fitness score will be saved and be available for viewing using:

```
python realtime_from_csv.py
```

### Contributing
Contributions are welcome! Please open an issue or submit a pull request if you would like to improve the project.