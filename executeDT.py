"""
Program to Execute Decision tree
Takes test data as input
Takes Decision tree threshold files as input
Displays the confusion matrix for the result
Plots the Decision boundary for test result
Author : Rahul Dashora
"""


import numpy as np
from csv import reader
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import  cm
ct_index=0
r=0

class confusion_matrix:
    """
    Confusion matrix class
    """
    __slots__='mat'

    def __init__(self):
        self.mat=[[0 for _ in range(4)] for _ in range(4)]
        #print(self.mat)

class node:
    '''
    Decision tree class
    root -> Root node of the tree
    display -> Displays the tree attribute
    in_order -> return inorder traversl of the tree
    pre_order -> return pre_order traversal of the tree
    create_tree -> return leafs path depth, Number of attribute node, Number of leaf nodes
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

    def __repr__(self):

            return str(self.threshold)



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


    def create_tree(self,root,depth,result):
        if(root.Attr_index!=None):
            print("\t"*depth,"if data[",root.Attr_index,"]<=",root.threshold," parent:",root.parent)
            result+="\t"*depth+"if data["+str(root.Attr_index)+"]<="+str(root.threshold)+" :"+"\n"
        else:
            print("\t"*(depth+1),"label=",root.value)
            result+="\t"*(depth+1)+"label="+str(root.value)+"\n"
        if(root.left!=None):
            result=self.create_tree(root.left,depth+1,result)
        if(root.right!=None):
            print("\t"*depth,"else:"," parent:",root.parent)
            result+="\t"*depth+"else:"+"\n"
            result=self.create_tree(root.right,depth+1,result)
        return result

    def prediction(self,root,data_record):

        if root.Attr_index!=None:
            if root.threshold>= data_record[root.Attr_index]:

                val=self.prediction(root.left,data_record)
                return val
            else:

                val=self.prediction(root.right,data_record)
                return val
        else:

            return root.value


def accuracy(actual_class,prediction):
    '''
    Comparing the result to actual class and displaying the confusion matrix
    :param actual_class: Original classes
    :param prediction: Predicted classes
    :return:None
    '''


    #Comparing classes
    conf_mat= confusion_matrix()
    for index in range(len(actual_class)):
        conf_mat.mat[int(actual_class[index]-1)][int(prediction[index]-1)]+=1

    #print(conf_mat.mat)
    print("********Confusion Matrix********\n")
    print(" \t1\t2\t3\t4")
    ind=1
    result=""
    for row in conf_mat.mat:
        result+=str(ind)+'\t'
        ind+=1
        for col in row:
            result+=str(col)+'\t'
        result+='\n'

    print(result)

    recognition=0

    for r_index in range(len(conf_mat.mat)):
        recognition+=conf_mat.mat[r_index][r_index]

    print("Recognition rate (%) :: ",(((recognition)/len(actual_class))*100))

    Cost_mat = [[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]]
    gain=0
    for r_index in range(4):
        for c_index in range(4):
            gain+=Cost_mat[r_index][c_index]* conf_mat.mat[r_index][c_index]

    print("Profit earned = ",gain, "cents")

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



def main():
    """
    Main function
    Reads Input test file
    Reads Threshold files
    Recreate Decision tree
    Calls Test function
    :return:
    """
    filename=input("Enter data file name")
    file = open(filename, "r")
    attr_list = list(reader(file))
    global ct_index

    data_list=attr_list #create the final list for passing
    data=[np.zeros((0)) for _ in range(len(data_list))]
    actual_class=np.zeros((0))
    data2=[np.zeros((0)) for _ in range(len(data_list[0]))]
    for r_index in range(len(data_list)):
        for c_index in range(len(data_list[r_index])):
            data2[c_index]=np.append(data2[c_index],np.float(data_list[r_index][c_index]))
            if(c_index==len(data_list[r_index])-1):
                #Storing class in separate list
                 actual_class=np.append(actual_class,np.float(data_list[r_index][c_index]))
            else:
                data[r_index]=np.append(data[r_index],np.float(data_list[r_index][c_index]))


    region=[]
    #read threshold file
    filename=input("Enter normal tree threshold file::")
    file=open(filename,"r")
    thres_list=[]
    for line in file:
        line=line.strip()
        line=line.split(' ')
        node_list=[]

        for index in range(0,len(line)-1,3):
            if line[index+2]!='None':
                node_list.append(node(None,None,None,None,None,float(line[index+2])))
            else:
                node_list.append(node(None,None,None,int(line[index]),float(line[index+1]),None))
        thres_list.append(node_list)


    dtree = dec_tree()

    dtree.root = recreate_tree(thres_list[0],thres_list[1],0,len(thres_list[0])-1)
    result_normal=test(data,dtree)

    accuracy(actual_class,result_normal)
    xx,yy,finalres=generatePlot(dtree)
    region.append((xx,yy,finalres))


    ct_index=0
    filename=input("Enter pruned tree threshold file::")
    file=open(filename,"r")
    thres_list=[]
    for line in file:
        line=line.strip()
        line=line.split(' ')

        node_list=[]

        for index in range(0,len(line)-1,3):
            if line[index+2]!='None':
                node_list.append(node(None,None,None,None,None,float(line[index+2])))
            else:
                node_list.append(node(None,None,None,int(line[index]),float(line[index+1]),None))
        thres_list.append(node_list)

    pruned_tree= dec_tree()
    pruned_tree.root = recreate_tree(thres_list[0],thres_list[1],0,len(thres_list[0])-1)
    pruned_result=test(data,pruned_tree)
    accuracy(actual_class,pruned_result)
    xx,yy,finalres=generatePlot(pruned_tree)
    region.append((xx,yy,finalres))

    plot_data(data2,region)


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



def recreate_tree(in_ord, pre_ord,st, ed):
    """
    Recreate Decision tree from in order and pre order traversal of the decision tree nodes
    :param in_ord: Inorder traversal of the data
    :param pre_ord: PreOrder traversal of the data
    :param st: Starting pointer
    :param ed: End Pointer
    :return: Root node of the decision tree
    """
    global ct_index
    if (st > ed):
        return None

    curr_node = pre_ord[ct_index]

    ct_index+=1


    if st == ed :
        return curr_node
    in_position = lookup(in_ord,st,ed+1,curr_node)
    curr_node.left = recreate_tree(in_ord, pre_ord, st, in_position-1)
    curr_node.right = recreate_tree(in_ord, pre_ord, in_position+1, ed)

    return curr_node


def lookup(in_ord,st,ed,target):
    """
    Look up function to find the node in In order List
    :param in_ord: Inorder node list
    :param st: Starting pointer
    :param ed: End Pointer
    :param target: Target node
    :return: Location of the Target node
    """
    for i in range(st, ed):
            if in_ord[i].threshold == target.threshold and target.Attr_index == in_ord[i].Attr_index:
                return i




main()
