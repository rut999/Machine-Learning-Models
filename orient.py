#!/usr/local/bin/python3
"""
Code by: nakopa,pvajja,rparvat
Authors: Naga Anjaneyulu , Prudhvi Vajja , Rutvik Parvataneni
"""
import sys
from Neural_Net import*
from KNN import *
import numpy as np
import pandas as pd
import pickle
import random
import math



class Node:
    
    def __init__(self,input_data,f1,f2,value,max_depth):
        self.input_data = input_data
        self.f1 = f1
        self.f2 = f2
        self.value = value
        self.purity = 0
        self.orient = None
        self.left = None
        self.right = None
        self.curr_depth = 0
        self.max_depth = max_depth
        
    def get_curr_depth(self):
        return self.curr_depth
        
    def get_max_depth(self):
        return self.max_depth
        
    def set_max_depth(self,max_depth):
        self.max_depth = max_depth
            
    def set_curr_depth(self,curr_depth):
        self.curr_depth = curr_depth
    
    def get_f1(self):
        return self.f1
    
    def get_f2(self):
        return self.f2

    def set_f1(self,f1):
        self.f1 = f1
        
    def set_f2(self,f2):
        self.f2 = f2
        
    def get_value(self):
        return self.value
    
    def set_value(self,value):
        self.value= value
    
    def get_input_data(self):
        return self.input_data
    
    def get_orient(self):
        return self.orient
    
    def get_purity(self):
        return self.orient
    
    def get_left(self):
        return self.left
        
    def get_right(self):
        return self.right
    
    def set_left(self,left_node):
        self.left = left_node

    def set_right(self,right_node):
        self.right = right_node
        
    def set_purity(self,purity):
        self.purity = purity
        
    def set_orient(self,orient):
        self.orient = orient
    
    def set_input_data(self,input_data):
        self.input_data = input_data
    
        
def get_orient_count(image_data):
    
    orientations = {}  
    orients = [0,90,180,270]
    for orient in orients:
        count = image_data.loc[image_data.iloc[:, 0] == orient].shape[0]
        orientations[orient] = count
    return orientations
     

def find_purity(image_data):
    
    orientations = get_orient_count(image_data)
    orient_sum = sum([ value for key,value in orientations.items()])
    max_purity = 0
    pure_orient = ""
    if orient_sum > 0 :
        for orient,value in orientations.items():
            if (value/orient_sum) > max_purity:
                max_purity = (value/orient_sum)
                pure_orient = orient
            
    return max_purity*100,pure_orient
    
   
        
def caluclate_entropy(image_data):
    
    orientations = get_orient_count(image_data)
    impurity = 1
    if image_data.shape[0] > 0 :
        for orientation in orientations:
            or_prob = orientations[orientation] / image_data.shape[0]
            impurity -= or_prob*math.log(or_prob+1,2)
    return impurity
    
    
def get_node(image_data):
    
    f1 = random.randint(0,image_data.shape[1]-1)
    f2 = random.randint(0,image_data.shape[1]-1)
    value = random.randint(0,255)
    node = Node(image_data,f1,f2,value,7)
        
    return node
        
        
       
def best_split_features(node):
    info_gain = -math.inf
    image_data = node.get_input_data()
    parent_entropy = caluclate_entropy(image_data)
    left_split = image_data.loc[image_data.iloc[:,node.get_f1()] < image_data.iloc[:,node.get_f2()]]
    left_entropy = caluclate_entropy(left_split)
         
    right_split = image_data.loc[image_data.iloc[:,node.get_f1()] >= image_data.iloc[:,node.get_f2()]]
    right_entropy = caluclate_entropy(right_split)
         
    if image_data.shape[0] > 0 :
        info_gain = parent_entropy - ((left_split.shape[0]/image_data.shape[0])*left_entropy 
                                             + (right_split.shape[0]/image_data.shape[0])*right_entropy)
             
    return info_gain


def build_tree(node):
        
      
        best_node = None
        max_gain = 0
        input_data = node.get_input_data()
        for i in range(0,5):
            node1 = get_node(input_data)
            gain = best_split_features(node1)
            if gain > max_gain:
                best_node = node1
            elif gain == 0:
                best_node = node1
        
        if best_node != None :
            node.set_f1(best_node.get_f1())
            node.set_f2(best_node.get_f2())
            node.set_value(best_node.get_value())
        
            left_split = input_data.loc[input_data.iloc[:,node.get_f1()] < input_data.iloc[:,node.get_f2()]]
            left_child = get_node(left_split)
            l_purity ,l_orient = find_purity(left_split)
            if l_purity >= 80 or node.get_curr_depth() >= node.get_max_depth():
                left_child.set_purity(l_purity)
                left_child.set_orient(l_orient)
                left_child.set_left(None)
                left_child.set_right(None)
                node.set_left(left_child)
            elif node.get_curr_depth() <= node.get_max_depth() :
                left_child.set_curr_depth(node.get_curr_depth() + 1)
                node.set_left(left_child)
                build_tree(left_child)
        
            right_split = input_data.loc[input_data.iloc[:,node.get_f1()] >= input_data.iloc[:,node.get_f2()]]
            right_child = get_node(right_split)
            r_purity ,r_orient = find_purity(right_split)
 
            if r_purity >= 80 or node.get_curr_depth() >= node.get_max_depth():
                right_child.set_purity(r_purity)
                right_child.set_orient(r_orient)
                right_child.set_left(None)
                right_child.set_right(None)
                node.set_right(right_child)
            elif node.get_curr_depth() <= node.get_max_depth() :
                right_child.set_curr_depth(node.get_curr_depth() + 1)
                node.set_right(right_child)
                build_tree(right_child)
  

          
def build_test_tree(model_tree):
    if model_tree != None :
        input_data = model_tree.get_input_data()
        print(input_data.shape[0])
        left_split = input_data.loc[input_data.iloc[:,model_tree.get_f1()] < input_data.iloc[:,model_tree.get_f2()]]
        right_split = input_data.loc[input_data.iloc[:,model_tree.get_f1()] >= input_data.iloc[:,model_tree.get_f2()]]
        if model_tree.get_left():
           model_tree.get_left().set_input_data(left_split)
           left = model_tree.get_left()
           build_test_tree(left)
        elif model_tree.get_right():
           model_tree.get_right().set_input_data(right_split)
           right = model_tree.get_right()
           build_test_tree(right)


def clean_tree(model_tree):
     if model_tree :
         if model_tree.get_input_data() is not None:
             model_tree.set_input_data(None)
    
         clean_tree(model_tree.get_left())
         clean_tree(model_tree.get_right())
 

def traverse_tree(tree,orient):
    
    if tree != None :
        left_child = tree.get_left()
        right_child = tree.get_right()
    
        if left_child != None:
            traverse_tree(left_child,orient)
    
        if right_child !=None:
            traverse_tree(right_child,orient)
        
        if left_child == None and right_child == None and tree.get_orient() != None:
            print("node_orient : node_purity" )
            print(str(tree.get_orient()) )
            orient[tree] = tree.get_input_data()
            

def traverse(node):
    if node:
        print("node:", (node.get_orient(), node.get_f1(), node.get_f2(), node.get_purity() ,node.get_curr_depth() ,node.get_input_data().shape[0]))
        traverse(node.get_left())
        traverse(node.get_right())

        
def get_accuracy(tree):
    orientations ={}
    traverse_tree(tree,orientations)
    class_count = 0
    total_count = 0
    for orient,image_data in orientations.items():
        if image_data is not None :
            class_count += image_data.loc[image_data.iloc[:, 0] == orient.get_orient()].shape[0]   
            total_count += image_data.shape[0]
    print(total_count)
    if total_count > 0 :
        return (class_count/total_count)*100
    else:
        return 0



def get_decision_tree_data(file_name):
    data ={}
    image_list =[]
    with open(file_name) as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            line_list = line.split()
            if len(line_list) > 2 :
                data[(line_list[0],line_list[1])] = line_list[1:]
    for key,value in data.items():
        value = [ int(val) for val in value]
        image_list.append(value)
    image_frame = np.asarray(image_list)
    final_data = pd.DataFrame(image_frame)
    return final_data



"""
NN and KNN code from here 

"""
def read_file(filename):
    f = open(filename, "r")
    train_data = []
    image_name = []
    lines = f.readlines()
    for x in lines:
        l = x.split()
        train_data.append(l[1:])
        image_name.append(l[0])
    f.close()
    return np.array(train_data),image_name

def output_file(image_id, y_pred):
    c = [[a, b]
         for a, b in zip(image_id, y_pred)]
    with open("output.txt", "w") as file:
        for i in range(len(c)):
            file.write(str(c[i][0]) + ' ' + str(int(c[i][1])))
            if i != len(c) - 1:
                file.write("\n")

if __name__== "__main__":
    if len(sys.argv) != 5:
        raise Exception("Error: expected 4 arguments")
    if sys.argv[1] == 'train':
        train_data,w = read_file(sys.argv[2])
        if(sys.argv[4] == 'nnet'or sys.argv[4] == 'best'):
            train(train_data,sys.argv[3])
        elif(sys.argv[4] =='tree'):
             train_data = get_decision_tree_data(sys.argv[2])
             decision_tree = Node(train_data,0,0,0,3)
             build_tree(decision_tree)
             accuracy = get_accuracy(decision_tree)
             print(accuracy)
    
             with open(sys.argv[3], 'wb') as file:
               pickle.dump(decision_tree, file)
            
        elif(sys.argv[4] =='nearest'):
            train_model(sys.argv[2],sys.argv[3])

    elif sys.argv[1] == 'test':
        test_data,test_image_id = read_file(sys.argv[2])
        if(sys.argv[4] == 'nnet' or sys.argv[4] == 'best'):
            y_pred = test(test_data,sys.argv[3])
            output_file(test_image_id,y_pred)
        elif (sys.argv[4] == 'tree'):
            
            test_data = get_decision_tree_data(sys.argv[2])
            model_tree = None
            with open(sys.argv[3], 'rb') as file:
                model_tree = pickle.load(file)
            clean_tree(model_tree)
            model_tree.set_input_data(test_data)
            build_test_tree(model_tree)
            accuracy = get_accuracy(model_tree)
            print(accuracy)
  
        elif (sys.argv[4] == 'nearest'):
            y_pred = test_model(sys.argv[2], sys.argv[3])
            output_file(test_image_id, y_pred)









