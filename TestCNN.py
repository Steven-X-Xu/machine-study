import Read_Pic
import numpy
import matplotlib.pyplot

input_nodes = 784
hiddent_nodes = 100
output_nodes = 10

learning_rate = 0.09

n = Read_Pic.neuralNetwork(input_nodes, hiddent_nodes, output_nodes, learning_rate)
scorecard = []
epochs = 5
def test():
    training_data_file = open("data/mnist_train.csv","r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    #for e in range(epochs):
    for record in  training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:])/225.0*0.99)+0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)



def  querytest():
    training_data_file = open("data/mnist_test.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    for record in training_data_list:
        all_values = record.split(",")
        correct_label = int(all_values[0])
        # print(correct_label,"correct_label")
        inputs = (numpy.asfarray(all_values[1:]) / 225.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        # print(label,"network's answer")
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

test();
querytest();
scorecard_array = numpy.asarray(scorecard)
print(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)