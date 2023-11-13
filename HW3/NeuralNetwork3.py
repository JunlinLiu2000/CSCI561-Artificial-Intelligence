import numpy as np
import sys
import time

def load_data(data_path):
    train_data_path = data_path[1]
    train_label_path = data_path[2]
    test_data_path = data_path[3]

    train_data = np.loadtxt(train_data_path,delimiter=",")
    train_label = np.loadtxt(train_label_path, delimiter=",", dtype=int)
    test_data = np.loadtxt(test_data_path,delimiter=",")
    return train_data, train_label, test_data

def relu(mat):
    flag = (mat <= 0) 
    mat[flag] = 0
    return mat
def sigmoid(mat):
    return 1/(1+np.exp(-mat))
def softmax(mat):
    max_s = np.max(mat, axis=-1, keepdims=True)
    tmp_value = mat-max_s
    return np.exp(tmp_value)/ np.sum(np.exp(tmp_value), axis=-1, keepdims=True)


class Model():
    def __init__(self, lr:float, batch_size:int, hidden_layers) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.input_shape = 2
        self.output_shape = 2

        self.layers = (self.input_shape,) + self.hidden_layers + (self.output_shape,)
        self.num_layers = len(self.layers)
        self.weight_list = [np.random.rand(self.layers[i], self.layers[i+1]) for i in range(self.num_layers-1)]
        self.bias_list = [np.random.rand(self.layers[i+1]) for i in range(self.num_layers-1)]

        self.val_accs = []
        self.train_accs = []

    def train(self, train_data, train_label):
        data_length = train_data.shape[0]
        data_index = np.array([i for i in range(data_length)])
        np.random.shuffle(data_index)
        X = train_data[data_index]
        Y = train_label[data_index]
        loss = []
        corrects = 0
        for i in range(data_length//self.batch_size):
            batch_data = X[i:i+self.batch_size]
            batch_label = Y[i:i+self.batch_size]

            logits = self.forward(batch_data)
            
            loss_ = self.softmax_cross_entropy(logits, batch_label)
            loss.append(loss_)
            dws, dbs = self.backwards(logits, batch_label)

            self.weight_list = [weight - self.lr * dw for weight, dw in zip(self.weight_list, dws)]
            self.bias_list = [bias - self.lr * db for bias, db in zip(self.bias_list, dbs)]

            correct = np.sum(np.argmax(logits, axis=-1) == batch_label)
            corrects += correct
        acc = corrects / (data_length//self.batch_size * self.batch_size)
        self.train_accs.append(acc)
        return np.mean(loss), acc


    def evaluate(self, validation_data):
        logits = self.forward(validation_data)
        predict_label =np.argmax(logits, axis=-1)
        return predict_label

    def forward(self, train_data):
        self.inputs_ = []
        self.zs = []

        x = train_data
        for i in range(len(self.weight_list)):
            self.inputs_.append(x)
            x = np.matmul(x, self.weight_list[i]) + self.bias_list[i]
            self.zs.append(x)
            if i < len(self.weight_list)-1:
                x = relu(x)
        
        return x

    def softmax_cross_entropy(self, logits, label):
        label_predict = softmax(logits)
        scores = label_predict[range(self.batch_size), label] + 1e-8
        loss = -np.sum(np.log(scores))/self.batch_size
        return loss

    def devitation_softmax_cross_entropy(self, logits, y):
        label_predict = softmax(logits)
        label_predict[range(self.batch_size), y] -= 1
        return label_predict

    def derivation_relu(self, mat):
        flag = (mat <= 0)
        mat[flag] = 0
        mat[~flag] = 1
        return mat

    def backwards(self, logits, label):

        # initialize weight and bias loss
        dws = [np.zeros((self.layers[i], self.layers[i+1])) for i in range(self.num_layers-1)]
        dbs = [np.zeros((self.layers[i+1])) for i in range(self.num_layers-1)]

        # output layer
        dl = self.devitation_softmax_cross_entropy(logits, label)
        dws[-1] = np.dot(self.inputs_[-1].T, dl) / self.batch_size
        dbs[-1] = np.sum(dl, axis=0, keepdims=True) / self.batch_size
        # propagation
        for i in range(2, self.num_layers):
            dl = np.dot(dl, self.weight_list[-i+1].T) * self.derivation_relu(self.zs[-i])
            dws[-i] = np.dot(self.inputs_[-i].T, dl) / self.batch_size
            dbs[-i] = np.sum(dl, axis=0, keepdims=True) / self.batch_size

        return dws, dbs

if __name__ == '__main__':
    start_time = time.time()
    train_data, train_label, test_data = load_data(data_path=sys.argv)

    epochs = 1000
    lrs = [3e-2, 3e-4]*5
    for lr in lrs:
        model = Model(lr, 100, (16, 32, 8))
        for epoch in range(epochs):
            loss, train_acc = model.train(train_data, train_label)
            # print("Epoch {}, train acc {}, loss {}".format(epoch, train_acc, loss))
        if np.mean(model.train_accs[-10:]) > 0.9:
            predict_label = model.evaluate(test_data)
            np.savetxt("test_predictions.csv", predict_label, fmt="%d")
            break
    
    end_time = time.time()
    time = end_time - start_time
    print(f"Duration:", time)