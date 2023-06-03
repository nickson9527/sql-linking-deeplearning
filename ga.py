from tqdm import trange
import numpy as np
# import pandas as pd
import os

class GA():
    def __init__(self,num_chrom,num_iter,rate_cross,rate_mutate,max_node,bit_range,raw_x,raw_y,fitness_f,args):
        self.num_iter = num_iter
        self.num_chrom = num_chrom
        self.rate_cross = rate_cross
        self.rate_mutate = rate_mutate
        self.population = []
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.fitness_f = fitness_f
        self.args = args
        self.max_node = max_node
        self.bit_range = bit_range
        self.lookup = {}
        self.best = {'model':None,'code':None,'v-acc':-5e10,'t-acc':-5e10,'v-loss':-5e10,'t-loss':-5e10,'time':0}
        self.history = []
    
    def init_chrom(self):
        for i in range(self.num_chrom):
            chrom = Chromosome(np.random.randint(self.bit_range, size=self.max_node))
            self.population.append(chrom)
    
    def genetic_algo(self):
        self.init_chrom()
        self.evaluation()
        progress = trange(self.num_iter,desc = "GA")
        for iter in progress:
            new_population = []
            weight = np.array([self.population[i].fitness for i in range(self.num_chrom)])
            # weight = np.log(weight)
            weight = weight - np.mean(weight)
            std_weight = np.exp(weight) / np.sum(np.exp(weight))
            for j in range(0,self.num_chrom,2):
                p1,p2 = np.random.choice(self.population,size=2,p=std_weight, replace=False)
                if np.random.rand() < self.rate_cross:
                    p1,p2 = self.cross_over(p1,p2)
                if np.random.rand() < self.rate_mutate:
                    p1 = self.mutation(p1)
                if np.random.rand() < self.rate_mutate:
                    p2 = self.mutation(p2)

                new_population.append(p1)
                new_population.append(p2)
            self.population = new_population
            self.evaluation()
        return self.best

    def cross_over(self,c1,c2):
        cut = np.random.randint(1,self.max_node-1)
        n1 = Chromosome(np.concatenate((c1.chrom[:cut],c2.chrom[cut:])))
        n2 = Chromosome(np.concatenate((c2.chrom[:cut],c1.chrom[cut:])))
        return n1,n2

    def mutation(self,chrom):
        cut = np.random.randint(0,self.max_node)
        c = chrom.chrom
        c[cut] = np.random.choice([i for i in range(self.bit_range) if i != c[cut] ])
        n = Chromosome(c)
        return n

    def evaluation(self):
        for i in range(self.num_chrom):
            chrom = self.population[i].chrom
            if str(chrom) in self.lookup:
                score = self.lookup[str(chrom)]
            else:
                score = self.fitness_f(chrom,self.raw_x,self.raw_y,self.args)
                self.lookup[str(chrom)] = score
            if score['v-acc'] > self.best['v-acc']:
                self.best['code'] = score['code']
                self.best['model'] = score['model']
                self.best['v-acc'] = score['v-acc']
                self.best['t-acc'] = score['t-acc']
                self.best['v-loss'] = score['v-loss']
                self.best['t-loss'] = score['t-loss']
                self.best['time'] = score['time']
                self.best['best epoch']:score['best epoch']

            self.population[i].fitness = score['v-acc']

class Chromosome():
    def __init__(self,chrom=np.zeros(2000)):
        self.chrom = chrom
        self.fitness = 0