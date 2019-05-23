import numpy
import scipy.special

"""
    以下代码可用于创建,训练和查询3层神经网络
"""
class neuralNetwork:

    #初始化:输入层节点数量,隐藏层节点数量,输出层节点数量
    def __init__(self,inputnodes,hiddennnodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennnodes
        self.onodes = outputnodes

        #权重 input与hidden  hidden与output  在-0.5~+0.5
        """
        此处是较为简单的权重设计
        self.wih = (numpy.random.rand(self.hnodes,self.inodes) - 0.5)
        self.wih = (numpy.random.rand(self.hnodes,self.onodes) - 0.5)
        """
        #神经网络的心脏:链接权重矩阵
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes,self.hnodes)))

        #学习效率
        self.lr = learningrate

        #激活函数
        self.activation_function = lambda x : scipy.special.expit(x)
        pass;


    #训练 (将计算得到的输出与需要的输出进行对比，使用差值来指导网络权重的更新)
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        #信号进入隐藏层节点(对信号进行权重调节, 并抑制)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #信号最终输出(对信号进行权重调节, 并抑制)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #计算差值(即训练样本提供的预期目标值:targets_list与实际计算:final_outputs的差值)
        output_errors = targets - final_outputs

        #计算隐藏层节点的反向传播误差
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #实现隐藏层和输出层的权重优化
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),numpy.transpose(hidden_outputs))

        #实现输入层和隐藏层的权重优化
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),numpy.transpose(inputs))
        pass;


    #查询
    def query(self,inputs_list):

        #二维数组
        inputs = numpy.array(inputs_list,ndmin=2).T

        #信号进入隐藏层节点(对信号进行权重调节,并抑制)
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #信号最终输出(对信号进行权重调节,并抑制)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #返回数据
        return  final_outputs

