import numpy
import scipy.special


# 定义神经网络
class NeuralNetwork:

    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 链接权重矩阵，wih（w_input_hidden） 和 who（w_hidden_output）
        # 数组内的权重为wij，其中链接是从节点i到下一层的节点j，比如
        # w11 w21
        # w12 w22
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate

        # sigmoid 函数作为激活函数, scipy.special 中调用sigmoid函数：expit()。
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 把输入列表和目标列表转换成2D矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算输入到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算来自隐藏层的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算进入最终输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算来自最终输出层的信号
        final_outputs = self.activation_function(final_inputs)

        # 输出层错误是（目标值-实际值）
        outputs_errors = targets - final_outputs

        # 隐藏层错误是 output_errors，通过权重分离，在隐藏节点重组
        hidden_errors = numpy.dot(self.who.T, outputs_errors)

        # 给输入层和隐藏层的链接更新权重
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(
            inputs)

        pass

    # 查询网络
    def query(self, inputs_list):
        # 把输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算进入隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 计算来自隐藏层的信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算进入最后输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算来自最后输出层的信号
        final_outputs = self.activation_function(final_inputs)

        return final_outputs