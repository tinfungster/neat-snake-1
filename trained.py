import os
from neat import nn, population
import statistics
import visualize
import pygame
import field
import food
import snake
import math
import sys
import pickle
from matplotlib import pyplot as plt
import numpy as np

rendering = True
debuggin = False
renderdelay = 0


blockSize = 16  # size of blocks
width = 16  # size of field width in blocks
height = 16
screenSize = (width * blockSize, height * blockSize)
speed = 1  # milliseconds per step
bg_color = 0x000000
snake_color = 0xFFFFFF
temp_speed = 0

best_foods = 0
best_fitness = 0
loop_punishment = 0.25
near_food_score = 0.2
moved_score = 0.01

# Initialize pygame and open a window
pygame.init()
screen = pygame.display.set_mode(screenSize)


pygame.time.set_timer(pygame.USEREVENT, speed)
clock = pygame.time.Clock()
scr = pygame.surfarray.pixels2d(screen)

dx = 1
dy = 0
generation_number = 0


def get_game_matrix(scr):
    global bg_color
    global snake_color
    res_matrix = []

    for i, x in enumerate(scr):
        res_arr = []
        if (i % blockSize == 0):
            for j, y in enumerate(x):
                if j % blockSize == 0:
                    if scr[i][j] == snake_color:
                        res_arr += [1]
                    elif scr[i][j] == bg_color:
                        res_arr += [0]
                    else:
                        res_arr += [2]
            res_matrix += [res_arr]

    # print res_matrix
    return res_matrix


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def positive(x):
    return x if x > 0 else 0


def left(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dx, dy = 0, 1
    elif (dx, dy) == (0, 1):
        dx, dy = 1, 0
    elif (dx, dy) == (1, 0):
        dx, dy = 0, -1
    elif (dx, dy) == (0, -1):
        (dx, dy) = (-1, 0)
    return (dx, dy)

def semi_left(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dy = 1
    elif (dx, dy) == (0, 1):
        dx = 1
    elif (dx, dy) == (1, 0):
        dy = -1
    elif (dx, dy) == (0, -1):
        dx = -1
    return (dx, dy)

def right(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        (dx, dy) = (0, -1)
    elif (dx, dy) == (0, -1):
        (dx, dy) = (1, 0)
    elif (dx, dy) == (1, 0):
        (dx, dy) = (0, 1)
    elif (dx, dy) == (0, 1):
        (dx, dy) = (-1, 0)
    return (dx, dy)

def semi_right(orientation):
    (dx, dy) = orientation
    if (dx, dy) == (-1, 0):
        dy = -1
    elif (dx, dy) == (0, -1):
        dx = 1
    elif (dx, dy) == (1, 0):
        dy = 1
    elif (dx, dy) == (0, 1):
        dx = -1
    return (dx, dy)

def look_to (orient, pos, game_matrix):
    dx, dy = orient
    px, py = pos

    body_found = False
    food_found = False
    
    dist_food = width
    dist_tail = width
    dist_wall = width

    dist = 0

    while px >= 0 and px < width and py >= 0 and py < height:
        if not body_found and game_matrix[px][py] == 1:
            dist_tail = dist
            body_found = True
        if not food_found and game_matrix[px][py] == 2:
            dist_food = dist
            food_found = True

        dist_wall = dist
        px += dx
        py += dy
        dist += 1
    
    return (dist_wall, dist_food, dist_tail)

def get_inputs(game_matrix, position, orientation):  # (dx,dy)
    
    dx, dy = orientation
    px, py = position
    
    dist_straight_wall, dist_straight_food, dist_straight_tail = look_to((dx, dy), position, game_matrix)
    
    dx, dy = left(orientation)
    dist_left_wall, dist_left_food, dist_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = right(orientation)
    dist_right_wall, dist_right_food, dist_right_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_left(orientation)
    dist_straight_left_wall, dist_straight_left_food, dist_straight_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_left(left(orientation))
    dist_left_left_wall, dist_left_left_food, dist_left_left_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_right(orientation)
    dist_straight_right_wall, dist_straight_right_food, dist_straight_right_tail = look_to((dx, dy), position, game_matrix)

    dx, dy = semi_right(right(orientation))
    dist_right_right_wall, dist_right_right_food, dist_right_right_tail = look_to((dx, dy), position, game_matrix)

    # ## teste caudas
    # if dist_straight_wall > dist_straight_tail:
    #      dist_straight_wall = dist_straight_tail

    # if dist_left_wall > dist_left_tail:
    #      dist_left_wall = dist_left_tail
    
    # if dist_right_wall > dist_right_tail:
    #      dist_right_wall = dist_right_tail

    return [dist_straight_wall, dist_straight_food, dist_straight_tail, dist_left_wall, dist_left_food, dist_left_tail, dist_right_wall, dist_right_food, dist_right_tail, dist_straight_left_wall, dist_straight_left_food, dist_straight_left_tail, dist_left_left_wall, dist_left_left_food, dist_left_left_tail, dist_straight_right_wall, dist_straight_right_food, dist_straight_right_tail, dist_right_right_wall, dist_right_right_food, dist_right_right_tail]


def save_best_generation_instance(instance, filename='best_generation_instances.pickle'):
    instances = []
    if os.path.isfile(filename):
        instances = load_object(filename)
    instances.append(instance)
    save_object(instances, filename)

def run():
    global best_fitness
    global best_foods
    global screen
    global width
    global height
    global blockSize
    global scr
    global generation_number
    global pop
    global bg_color
    global snake_color
    
    instances = load_object("best_generation_instances.pickle")
    
    for maior_obj in instances:
        

        g = maior_obj['genome']
        if(g.num_inputs < 21):
            continue

        if(maior_obj['fitness']< 0.4):
            continue
            
        net = maior_obj['net']
        
        visualize.draw_net(g, view=True, filename="xor2-all.gv")
        # net = nn.create_feed_forward_phenotype(g)
        dx = 1
        dy = 0
        score = 0.0
        hunger = 100
        # Create the field, the snake and a bit of food
        theField = field.Field(screen, width, height, blockSize, bg_color)
        theFood = food.Food(theField)
        theSnake = snake.Snake(theField, snake_color, 5)
        snake_head_x, snake_head_y = theSnake.body[0]
        dist = math.sqrt((snake_head_x - theFood.x) ** 2 + (snake_head_y - theFood.y) ** 2)
        error = 0
        countFrames = 0

        pastPoints = set()

        foods = 0

        while True:
            countFrames += 1

            event = pygame.event.wait()

            if event.type == pygame.QUIT:  # window closed
                print("Quittin")
                save_object(pop, 'population.dat')  ## export population
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:  # timer elapsed
                matrix = get_game_matrix(scr)
                # print matrix
                head_x, head_y = theSnake.body[0]
                head_x += dx
                head_y += dy
                inputs = get_inputs(matrix, (head_x, head_y), (dx, dy))
                if debuggin:
                    print(inputs)
                
                outputs = net.serial_activate(inputs)
                direction = outputs.index(max(outputs))
                if direction == 0:  # dont turn
                    # print "Straight"
                    pass

                if direction == 1:  # turn left
                    # print "Left"
                    (dx, dy) = left((dx, dy))
                if direction == 2:  # turn right
                    # print "Right"
                    (dx, dy) = right((dx, dy))

                hunger -= 1
                if not theSnake.move(dx, dy) or hunger <= 0:
                    break
                else:
                    inputs = get_inputs(matrix, (head_x, head_y), (dx, dy))
                    
                    # current_state = inputs[1] < inputs[0]
                    #
                    # wall, bread, wall_left, bread_left, wall_right, bread_right = (inputs)
                    ##score += math.sqrt((theFood.x - theSnake.body[0][0]) ** 2 + (theFood.y - theSnake.body[0][1]) ** 2)
                    score += moved_score
                    pass

            # loop punishment
            if theSnake.body[0] in pastPoints:
                score -= loop_punishment
            pastPoints.add(theSnake.body[0])

            # food
            if theSnake.body[0] == (theFood.x, theFood.y):
                pastPoints = set()
                theSnake.grow()
                theFood = food.Food(theField)  # make a new piece of food
                score += 5
                hunger += 100
                foods += 1
            else:
                # near food score
                if abs(theSnake.body[0][0] - theFood.x + theSnake.body[0][1] - theFood.y) <= 1:
                    score += near_food_score

            if rendering:
                theField.draw()
                theFood.draw()
                theSnake.draw()
                pygame.display.update()
                pygame.time.wait(renderdelay)

            if event.type == pygame.KEYDOWN:  # key pressed
                if event.key == pygame.K_LEFT:
                    temp_speed = 200
                    pygame.time.set_timer(pygame.USEREVENT, temp_speed)
                elif event.key == pygame.K_RIGHT:
                    temp_speed = speed
                    pygame.time.set_timer(pygame.USEREVENT, temp_speed)

        # Game over!
        if rendering:
            for i in range(0, 10):
                theField.draw()
                theFood.draw()
                theSnake.draw(damage=(i % 2 == 0))
                pygame.display.update()

        # pygame.time.wait(100)
        # score = positive(score)
        g.fitness = score/100

        best_foods = max(best_foods, foods)
        best_fitness = max(best_fitness, g.fitness)
        print("foods",foods, "\tscore","%.2f"%score, "\toriginal fit", "%.2f"%maior_obj['fitness'], "\tgeneration" "%.2f"%maior_obj['num_generation'])
        # if debuggin:
        # print(f"Generation {generation_number} \tGenome {genome_number} \tFoods {foods} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
    

   
run()