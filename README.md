# neat-snake
Integrating Neural Evolution In Augmented Topologies in a snake game to make it a self learner.

![](demo/demo.gif)

It also print topologies

![](demo/net.jpg)

# Requirements
This projects requires pygame python library as well as python-neat library
For running trained models (and print network topologies), you will also need graphviz and matplotlib

# Running
To run this game you need to execute the game file inside the project folder with python.  

```bash
python game.py
```

## Saving knowledge
Whenever you quit the game a new file called "population.dat" gets created. That file contains all the knowledge learnt so far.

## Loading knowledge
In order to load knowledge , open up the game with the population file as a terminal argument.
```bash
python game.py population.dat
```

# Notes
The game is still under development. The snake is still not able to always go to the food, but it can find 50+ foods.

## Academic

There are several challenges when modeling artificial intelligence methods for autonomous players on games (bots). NEAT is one of the models that, combining genetic algorithms and neural networks, seek to describe a bot behavior more intelligently. 

In NEAT, a neural network is used for decision making, taking relevant inputs from the environment and giving real-time decisions. In a more abstract way, a genetic algorithm is applied for the learning step of the neural networks' weights, layers, and parameters.
