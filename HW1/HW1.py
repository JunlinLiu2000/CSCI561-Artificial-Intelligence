import sys
import time
import csv
import math
import random
import numpy as np


def read_file(input_file):
    with open(input_file,encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for i in range(1,len(lines)):
        a = lines[i].rstrip('\n').split()
        tuple1 = (int(a[0]),int(a[1]),int(a[2]))
        data.append(tuple1)
    return lines[0], data

def distance_matrix(city_size, data):
    city_dist = np.zeros([city_size,city_size])
    city_dic = {}
    for i in range(city_size):
        city_dic[data[i]] = i
        for j in range(city_size):
            city_dist[i,j] = math.dist(data[i],data[j])
    return city_dist, city_dic

def CreateInitialPopulation(city_size, data, pop_size, city_dic):
    initial_population = []
    list1 = list(range(city_size))
    for i in range(pop_size):
        path = random.sample(list1, city_size)
        # path.append(path[0])
        initial_population.append(path)
    return initial_population

def count_fitness(city_dist, city_dic, population):
    fitness_list = []
    distance_list =[]
    count = 0
    fitness_sum = 0
    for path in population:
        distance = 0
        for i in range(len(path)):
            if i == len(path)-1:
                distance += city_dist[path[i],path[0]]
            else:
                distance += city_dist[path[i],path[i+1]]
        fitness = 1/distance
        fitness_list.append((fitness,count))
        distance_list.append(distance)
        fitness_sum += fitness
        count += 1
    return sorted(fitness_list), distance_list, fitness_sum

def CreateMatingPool(population, RankList, fitness_sum, pop_size, distance_list):
    distance_list = np.array(distance_list)
    prob_list = distance_list/fitness_sum
    q = prob_list.cumsum()
    matingPool = []
    for i in range(pop_size):
        r = np.random.rand()
        for j in range(pop_size):
            if r < q[0]:
                matingPool.append(0)
                break
            elif q[j] < r <= q[j+1]:
                matingPool.append(j+1)
                break
    population = np.array(population)
    next_gen = population[matingPool, :]
    return next_gen

def select_index(city_size):
    index1 = random.randint(1, city_size)
    index2 = random.randint(1, city_size)
    while index1 == index2:
        index1 = random.randint(1, city_size)
    Start_index = min(index1, index2)
    End_index = max(index1, index2)
    return Start_index, End_index

def intercross(Parent1, Parent2):
    Start_index, End_index = select_index(city_size)
    son1 = []
    son2 = []
    son = []
    for i in range(Start_index, End_index):
        son1.append(Parent1[i])
    for item in Parent2:
        if item not in son1:
            son2.append(item)
    son = son1+son2
    return son

def Crossover(next_gen, city_size, cross_prob, pop_size):
    Child = []
    for i in range(pop_size-1):
        if cross_prob >= np.random.rand():
            Child.append(intercross(next_gen[i], next_gen[i+1]))
    return Child

def mutation_sub(city_num, pop_num, next_gen, mut_prob):
    for i in range(pop_num):
        if mut_prob >= np.random.rand():
            r1 = np.random.randint(city_num)
            r2 = np.random.randint(city_num)
            while r2 == r1:
                r2 = np.random.randint(city_num)
            if r1 > r2:
                temp = r1
                r1 = r2 
                r2 = temp
            next_gen[i, r1:r2] = next_gen[i, r1:r2][::-1]

def comp_dis(city_num, matrix_distance, one_path):
    res = 0
    for i in range(city_size - 1):
        res += matrix_distance[one_path[i], one_path[i + 1]]
    return res

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    pop_size = 5
    cross_prob = 0.5
    mut_prob = 0.5
    iteration = 5
    city_size, data = read_file(input_file)
    city_size = int(city_size)
    city_dist,city_dic = distance_matrix(city_size, data)
    population = CreateInitialPopulation(city_size, data, pop_size, city_dic)
    RankList, distance, fitness_sum= count_fitness(city_dist, city_dic, population)
    print(population)
    print(fitness_sum)
    print(distance)
    evbest_path = population[0]
    best_path_list = []
    best_distance_list = []
    # for i in range(iteration):
    #     next_gen = CreateMatingPool(population, RankList, fitness_sum, pop_size, distance)
    #     cross_sub(city_size, pop_size, next_gen, cross_prob, evbest_path)
    #     mutation_sub(city_size, pop_size, next_gen, mut_prob)

    #     for j in range(pop_size):
    #         distance[j] = comp_dis(city_size, city_dic, next_gen[j, :])
    #     index = distance.argmin()  # index 记录最小总路程

    #     if distance[index] <= evbest_distance:
    #         evbest_distance = distance[index]
    #         evbest_path = next_gen[index, :]
    #     else:
    #         distance[index] = evbest_distance
    #         next_gen[index, :] = evbest_path
        
    #     best_path_list.append(evbest_path)
    #     best_distance_list.append(evbest_distance)

    # best_path = evbest_path
    # best_distance = evbest_distance
    # print(best_distance)


