"""
Program to train the decision tree
Identifies best split threshold and best attributes
Uses Information gain for attribute selection
Performs chi-squared pruning after DT is created
Write the threshold onto two files - normal_threshold, chi_squared_threshold
Author - Ritvik Joshi, Rahul Dashora, Amruta Deshpande
"""

from csv import reader
import numpy as np
import math
import copy
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import cm

class node:
    '''
    Decision tree nodes
    parent- parent of current node
    left - left child
    right- right child
    Attr_index - if Attribute node, then indicates attribute index else None
    threshold - if Attribute node, then indicates Threshold for split else None
    value - if leaf node, then indicates value of the result, else None
    '''
    __slots__='parent','left','right','Attr_index','threshold','value'

    def __init__(self,parent,left,right,index,threshold,value):
        self.left=left
        self.right=right
        self.Attr_index=index
        self.threshold=threshold
        self.value=value
        self.parent=parent

    def __str__(self):

        if self.parent!=None or self.left!=None:
            return str(self.Attr_index)
        else:
            return "None"

class dec_tree:
    '''
    Decision tree class
    root -> Root node of the tree
    display -> Displays the tree attribute
    in_order -> return inorder traversl of the tree
    pre_order -> return pre_order traversal of the tree
    create_tree -> return leafs path depth, Number of attribute node, Number of leaf nodes
    '''
    __slots__ ="root"

    def __init__(self):
        self.root=None


    def display(self,root):

        print('Attribute::',root.Attr_index,' Threshold::',root.threshold,' Value::',root.value)
        if(root.left!=None):
            print("Attribute:: ",root.Attr_index,' going left')
            self.display(root.left)

        if(root.right!=None):
            print("Attribute:: ",root.Attr_index,'going right')
            self.display(root.right)

        print("Attribute:: ",root.Attr_index,' going back')


    def in_tree(self,root,depth,result):
        if(root.Attr_index!=None):
            result+=str(root.Attr_index)+" "+str(root.threshold)+" "+str(root.value)+" "
        else:
            result+=str(root.parent)+" "+str(root.threshold)+" "+str(root.value)+" "
        if(root.left!=None):
            result=self.in_tree(root.left,depth+1,result)
        if(root.right!=None):
            result=self.in_tree(root.right,depth+1,result)

        return result

    def pre_tree(self,root,depth,result):
        if(root.left!=None):
            result=self.pre_tree(root.left,depth+1,result)
        if(root.Attr_index!=None):
            result+=str(root.Attr_index)+" "+str(root.threshold)+" "+str(root.value)+" "
        else:
            result+=str(root.parent)+" "+str(root.threshold)+" "+str(root.value)+" "

        if(root.right!=None):
            result=self.pre_tree(root.right,depth+1,result)

        return result

    def create_tree(self,root,depth,result,node_count,leaf_count,lt_count):
        if(root.Attr_index!=None):
            #print("\t"*depth,"if data[",root.Attr_index,"]<=",root.threshold," :",'parent',root.parent)
            result+="\t"*depth+"if data["+str(root.Attr_index)+"]<="+str(root.threshold)+" :"+"\n"
            node_count+=1
        else:
            #print("\t"*(depth+1),"label=",root.value)
            result+="\t"*(depth+1)+"label="+str(root.value)+"\n"
            leaf_count+=1
            lt_count[root.value-1].append(depth-1)
        if(root.left!=None):
            result,node_count,leaf_count,lt_count=self.create_tree(root.left,depth+1,result,node_count,leaf_count,lt_count)
        if(root.right!=None):
            #print("\t"*depth,"else:",'parent',root.parent)
            result+="\t"*depth+"else:"+"\n"
            result,node_count,leaf_count,lt_count=self.create_tree(root.right,depth+1,result,node_count,leaf_count,lt_count)
        return result,node_count,leaf_count,lt_count

    def prediction(self, root, data_record):

        if root.Attr_index != None:
            if root.threshold >= data_record[root.Attr_index]:

                val = self.prediction(root.left, data_record)
                return val
            else:

                val = self.prediction(root.right, data_record)
                return val
        else:

            return root.value

def calc_distance(point1, point2):
    '''
    calculate euclidean distance
    :param point1: first point
    :param point2: second point
    :return: distance b/w the points
    '''
    dist = (np.sum((point1-point2)**2))**(1/2)
    return dist



def decision_tree(data):
    '''
    Decision tree Function
    Create decision tree by performing attribute selection and checking for purity of the splits
    :param data: Input data
    :return: decision tree nodes
    '''

    #Stores attributes information gain
    Attribute_Entropy=[]

    #To calculate Information gain of all the attributes present in the data set expect the target variable
    for attribute in range(0,len(data[0])-1):
        data.sort(key=lambda x:x[attribute])
        new_data = convert_data_colwise(data)
        I_parent = calc_parent_entropy(new_data[len(new_data)-1]) #information gain of the parent
        Attribute_Entropy.append((attribute,Entropy(new_data[attribute],new_data[len(new_data)-1],I_parent)))

    #Identify maximum information gain
    Attribute_Entropy.sort(key=lambda x: x[1][1],reverse=True)

    #create node for best attribute
    current_node=node(None,None,None,Attribute_Entropy[0][0],Attribute_Entropy[0][1][0],None)

    #splitiing data based on best attribute
    left_data,right_data=splitdata(data,Attribute_Entropy[0][0],Attribute_Entropy[0][1][0])

    #checking purity of left split
    prob_one,prob_two,prob_three,prob_four =check_purity(left_data)

    #if pure
    if(prob_one>=1 or prob_two>=1 or prob_three>=1 or prob_four>=1):
        if (prob_one>=1):
            #print('Terminating left with one ')
            current_node.left = node(None,None,None,None,None,1)
        elif(prob_two>=1):
            #print('Terminating left with two ')
            current_node.left = node(None,None,None,None,None,2)
        elif(prob_three>=1):
            #print('Terminating left with three ')
            current_node.left = node(None,None,None,None,None,3)
        elif(prob_four>=1):
            #print('Terminating left with four ')
            current_node.left = node(None,None,None,None,None,4)
    else:
        #if not pure
        current_node.left=decision_tree(left_data)
        current_node.left.parent = current_node

    #checking purity of right split
    prob_one,prob_two,prob_three,prob_four =check_purity(right_data)
    #if pure
    if(prob_one>=1 or prob_two>=1 or prob_three>=1 or prob_four>=1):
        if (prob_one>=1):
            #print('Terminating right with one ')
            current_node.right = node(None,None,None,None,None,1)
        elif(prob_two>=1):
            #print('Terminating right with two ')
            current_node.right = node(None,None,None,None,None,2)
        elif(prob_three>=1):
            #print('Terminating right with three ')
            current_node.right = node(None,None,None,None,None,3)
        elif(prob_four>=1):
            #print('Terminating right with four ')
            current_node.right = node(None,None,None,None,None,4)
    else:
        #if not pure
        current_node.right=decision_tree(right_data)
        current_node.right.parent = current_node

    #return node of the tree
    return current_node




def check_purity(data):
    '''
    Purity checker
    :param data: splitted data
    :return: probability of the count of classes in the split
    '''
    new_data = convert_data_colwise(data)
    prob_one,prob_two,prob_three,prob_four=cal_prob(new_data[len(new_data)-1])

    return prob_one,prob_two,prob_three,prob_four

def splitdata(data,index,Threshold):
    '''
    Split the data based on the threshold
    :param data: Input data
    :param index: Attribute Index
    :param Threshold: Threshold value
    :return: splitted data list
    '''
    data.sort(key=lambda x:x[index])
    left=[]
    right=[]
    for attribute in data:
        #print(attribute,Threshold)
        if attribute[index]<=Threshold:
            left.append(attribute)
        else:
            right.append(attribute)

    return left,right


def Entropy(data,Target_variable,Entropy_parent):
    """
    CAlucate information gain
    :param data: Input data of attribute
    :param Target_variable: Class attribute list
    :param Entropy_parent: Entropy or parent
    :return: Best information gain of the attribute and threshold value
    """
    Entropy=[]
    Threshold=[]
    for i in range(len(data)-1):
        left=[]
        right=[]
        if i+1!=len(data):
            threshold = (data[i]+data[i+1])/2
        for j in range(len(data)):
            if(threshold>float(data[j])):
                left.append(Target_variable[j])
            else:
                right.append(Target_variable[j])
        #print(len(left),len(right))
        Entropy.append(Entropy_parent-calc_Entropy(left,right))
        Threshold.append(threshold)

    max_info_gain = max(Entropy)
    max_index = Entropy.index(max_info_gain)
    best_threshold = Threshold[max_index]

    return (best_threshold,max_info_gain)


def cal_prob(list):
    """
    Calculate probability of the number of classes present in the list
    :param list: Class list
    :return: probability of the classes
    """

    count_list_one=0
    count_list_two=0
    count_list_three=0
    count_list_four=0
    for i in list:
        if i==1 :
            count_list_one+=1
        elif i==2:
            count_list_two+=1
        elif i==3:
            count_list_three+=1
        else:
            count_list_four+=1
    if(len(list)>0):
        pl_one=(count_list_one/len(list))
        pl_two= (count_list_two/len(list))
        pl_three= (count_list_three/len(list))
        pl_four= (count_list_four/len(list))
    else:
        pl_one=0
        pl_two=0
        pl_three=0
        pl_four=0
    #print("C1:",count_list_one,"C0:",count_list_zero)
    return pl_one,pl_two,pl_three,pl_four

def calc_parent_entropy(data):
    """
    Calculate parents entropy
    :param data: Input data
    :return: parent Entropy
    """

    pl=list(cal_prob(data))
    for index in range(len(pl)):
        if pl[index]==0:
            pl[index]=0.001

    Entropy_parent = (-(pl[0])* math.log2(pl[0]))+(-(pl[1])* math.log2(pl[1]))+(-(pl[2])* math.log2(pl[2]))+(-(pl[3])* math.log2(pl[3]))

    return Entropy_parent

def calc_Entropy(left,right):
    """
    Calculate Entropy of Left and right child
    :param left: Left split
    :param right: Right split
    :return: Weighted Entropy
    """

    pl=list(cal_prob(left))
    pr=list(cal_prob(right))


    for index in range(len(pl)):
        if pl[index]==0:
            pl[index]=0.001
        if pr[index]==0:
            pr[index]=0.001



    Entropy_left = (-(pl[0])* math.log2(pl[0]))+(-(pl[1])* math.log2(pl[1]))+(-(pl[2])* math.log2(pl[2]))+(-(pl[3])* math.log2(pl[3]))

    Entropy_right = (-(pr[0])* math.log2(pr[0]))+(-(pr[1])* math.log2(pr[1]))+(-(pr[2])* math.log2(pr[2]))+(-(pr[3])* math.log2(pr[3]))


    #print("E::",Entropy_left,Entropy_right)
    prob_left = len(left)/(len(left)+len(right))
    prob_right = len(right)/(len(left)+len(right))

    weighted_Entropy = prob_left*Entropy_left + prob_right*Entropy_right


    return weighted_Entropy

def convert_data_colwise(new_data):
    """
    To convert rowise data into column wise
    :param new_data: rowwise data
    :return: colwise data
    """
    data2=[np.zeros((0)) for _ in range(len(new_data[0]))]
    for r_index in range(len(new_data)):
        for c_index in range(len(new_data[r_index])):
            data2[c_index]=np.append(data2[c_index],np.float(new_data[r_index][c_index]))

    return data2


def chi_square_pruning(data,node):
    """
    Perform Chi-Squared Pruning
    :param data: Input data
    :param node: Root node of the Decision tree
    :return: Root node of pruned tree
    """
    left,right = splitdata(data,node.Attr_index,node.threshold)
    new_data = convert_data_colwise(data)
    new_left = convert_data_colwise(left)
    new_right = convert_data_colwise(right)
    #print(len(left),len(right))
    pruning_flag=False


    if (node.left.value==None or node.right.value==None):
        if(node.left.value==None):
            node.left = chi_square_pruning(left,node.left)
        if(node.right.value==None):
            node.right =chi_square_pruning(right,node.right)

    if(node.left.value!=None and node.right.value!=None):

        class_count_data=list(get_count(new_data[len(new_data)-1]))
        class_count_left = list(get_count(new_left[len(new_left)-1]))
        class_count_right=list(get_count(new_right[len(new_right)-1]))
        left_prob = len(left)/len(data)
        right_prob = len(right)/len(data)

        left_cap=[]
        right_cap=[]
        for index in range(len(class_count_data)):
            left_cap.append(class_count_data[index]*left_prob)
            right_cap.append(class_count_data[index]*right_prob)

        chi_square = 0
        for index in range(len(left_cap)):
            if left_cap[index]!=0:
                chi_square+=((class_count_left[index]-left_cap[index])**2)/left_cap[index]
            if right_cap[index]!=0:
                chi_square+=((class_count_right[index]-right_cap[index])**2)/right_cap[index]


        if(chi_square<=7.145):
            #print("***********************************Pruning True***********************************")
            pruning_flag=True

    if(pruning_flag):
        left_index=class_count_left.index(max(class_count_left))
        right_index=class_count_right.index(max(class_count_right))
        if(class_count_left[left_index]< class_count_right[right_index]):
            node.left.value=(left_index+1)

            return node.left
        else:
            node.right.value=(right_index+1)

            return node.right

    return node

def get_count(list):
    """
    Count of Classes present in the tree
    :param list:  Class list
    :return: class counts
    """
    count_list_one=0
    count_list_two=0
    count_list_three=0
    count_list_four=0
    for i in list:
        if i==1 :
            count_list_one+=1
        elif i==2:
            count_list_two+=1
        elif i==3:
            count_list_three+=1
        else:
            count_list_four+=1

    return count_list_one,count_list_two,count_list_three,count_list_four

def plot_data(data,contour):
    '''
    Plot data distribution and Decision boundary across data
    :param data: Input data
    :param weights: Final weights
    :return:None
    '''

    col=('blue','yellow','red','black')
    #Plotting data points onto the graph
    fig1= plt.subplot(1,2,1)
    plt.subplot(1,2,1),plt.title("Before pruning")
    fig2= plt.subplot(1,2,2)
    plt.title("After pruning")

    cmap = cm.PRGn
    fig1.contourf(contour[0][0],contour[0][1],contour[0][2],cmap= cm.get_cmap(cmap,4))
    fig2.contourf(contour[1][0],contour[1][1],contour[1][2],cmap= cm.get_cmap(cmap,4))
    for index in range(len(data[0])):
        fig1.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')
        fig2.scatter(data[0][index], data[1][index],c=col[int(data[2][index])-1],marker='x')




    blue_patch = mpatches.Patch(color='blue', label='Class 1')
    yellow_patch = mpatches.Patch(color='yellow', label='Class 2')
    red_patch = mpatches.Patch(color='red', label='Class 3')
    black_patch = mpatches.Patch(color='black', label='Class 4')


    #fig1.xlabel=("Attribute1"),fig1.ylabel("Attribute2"),fig1.title("Attribute and Class Distribution")
    fig1.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    fig2.legend(handles=[blue_patch,yellow_patch,red_patch,black_patch],loc="upper right")
    plt.show()

def generatePlot(tree):
    """
    Generate the decision boundary of the Decision tree split
    :param tree:
    :return:
    """
    xl = []
    yl = []
    data =[]
    dim = 100
    result = []
    for i in range(0, dim):
        for j in range(0, dim):
            xl.append(i / dim)
            yl.append(j / dim)
            data.append(np.array(((i/dim), (j/dim))))


    result = test(data,tree)

    x = np.array(xl)
    y = np.array(yl)
    res = np.array(result)
    #print(res)
    xx = x.reshape((dim, dim))

    yy = y.reshape((dim, dim))
    r2d = res.reshape((dim, dim))
    #print(r2d)
    return  xx,yy,r2d

def test(data,tree):
    """
    Test function
    :param data: Test data
    :param tree: Decision tree object
    :return: results
    """
    result=[]
    for i in range(len(data)):
        result.append(tree.prediction(tree.root,data[i]))

    return result

def main():
    """
    MAin function
    Reads input data filename from user
    Converts the data into numpy array format
    Calls Decision tree function
    Calls Pruning function
    Prints Results
    Create Output threshold files
    :return:
    """
    filename=input("Enter the filename")

    #filename='train_data.csv'
    file = open(filename, "r")
    attr_list = list(reader(file))


    data_list=attr_list #create the final list for passing
    data=[np.zeros((0)) for _ in range(len(data_list))]
    data2 = [np.zeros((0)) for _ in range(len(data_list[0]))]
    for r_index in range(len(data_list)):

        for c_index in range(len(data_list[r_index])):
            data2[c_index] = np.append(data2[c_index], np.float(data_list[r_index][c_index]))
            data[r_index]=np.append(data[r_index],np.float(data_list[r_index][c_index]))

    dtree = dec_tree()
    dtree.root=decision_tree(data)
    pruned_tree =copy.deepcopy(dtree)

    print('***********************Decision Tree(Not pruned)*******************************')
    result,node_count,leaf_count,lp_count=dtree.create_tree(dtree.root,1,"",0,0,[[] for _ in range(4)])

    print('Number of node',node_count)
    print('Number of leaf node',leaf_count)
    #print(lp_count)
    for index in range(len(lp_count)):
        max_val=max(lp_count[index])
        min_val=min(lp_count[index])
        print('Maximum path to leaf node',(index+1),'::',max_val)
        print('Min path to leaf node',(index+1),'::',min_val)
        print('Average path to leaf node',(index+1),"::",((max_val+min_val)//2))

    pruned_tree.root=chi_square_pruning(data,pruned_tree.root)
    print('***********************Decision Tree(Chi-Squared pruned(0.05))*******************************')
    final_result,node_count,leaf_count,lp_count=pruned_tree.create_tree(pruned_tree.root,1,"",0,0,[[] for _ in range(4)])

    print('Number of node ::',node_count)
    print('Number of leaf node ::',leaf_count)
    #print(lp_count)
    for index in range(len(lp_count)):
        max_val=max(lp_count[index])
        min_val=min(lp_count[index])
        print('Maximum path to leaf node',(index+1),'::',max_val)
        print('Min path to leaf node',(index+1),'::',min_val)
        print('Average path to leaf node',(index+1),"::",((max_val+min_val)//2))


    print('Decision tree threshold are written in file orig_threshold.csv and chi_sqaured_threshold.csv')

    target=open('normal_threshold.csv','w')
    write_data = dtree.pre_tree(dtree.root,1,"")
    target.write(write_data+'\n')
    write_data = dtree.in_tree(dtree.root,1,"")
    target.write(write_data+'\n')

    target=open('chi_squared_threshold.csv','w')
    write_data = dtree.pre_tree(pruned_tree.root,1,"")
    target.write(write_data+'\n')
    write_data = dtree.in_tree(pruned_tree.root,1,"")
    target.write(write_data+'\n')

    region=[]
    xx, yy, finalres = generatePlot(dtree)
    region.append((xx, yy, finalres))
    xx, yy, finalres = generatePlot(pruned_tree)
    region.append((xx, yy, finalres))
    plot_data(data2, region)


main()
