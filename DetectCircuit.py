import UtilFunctions
import numpy as np
import cv2
import math
import sys
import tensorflow as tf
np.set_printoptions(threshold=np.nan)
####[1,0,0,0] - resistor
####[0,1,0,0] - inductor
####[0,0,1,0] - capacitor
####[0,0,0,1] - diode

restore = 1
np.set_printoptions(threshold=np.nan)
X_test = []
Y_test = []
s="newcir1.png"
input_nodes=100
img = cv2.imread(s,0)

X_test,connections,components = UtilFunctions.findcomponents(img)
for i in range (0,len(X_test)):
    Y_test.append([0,0,0,0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)
graph = tf.Graph()

with graph.as_default():
    n_nodes_hl1 = 100
    n_nodes_hl2 = 10
    n_nodes_hl3 = 10
    n_classes = 4
    x = tf.placeholder('float', [None,100])
    y = tf.placeholder('float', [None, 4])

    def neural_network_model(data):
        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_nodes, n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                          'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

        output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes])),}


        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
        return output
    
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    if (restore==1):
        with tf.Session(graph=graph) as session:
            ckpt = tf.train.get_checkpoint_state('./')
            saver.restore(session, ckpt.model_checkpoint_path)
            _, c, p = session.run([optimizer, cost, prediction], feed_dict={x: X_test,y: Y_test})
            for i in range (0,len(X_test)):
                max1=max(p[i])
                for j in range (0,4):
                    if (p[i][j]==max1):
                        p[i][j]=1
                    else:
                        p[i][j]=0
            for i in range (0,len(p)):
                x=components[i][0]
                y=components[i][1]
                w=components[i][2]
                h=components[i][3]
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                width=1
                size=0.5
                if (p[i][0]==1):
                    cv2.putText(img,str(i)+". Resistor", (components[i][0],components[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, size, 0,width)
                if (p[i][1]==1):
                    cv2.putText(img,str(i)+". Inductor", (components[i][0],components[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, size, 0,width)
                if (p[i][2]==1):
                    cv2.putText(img,str(i)+". Capacitor", (components[i][0],components[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, size, 0,width)
                if (p[i][3]==1):
                    cv2.putText(img,str(i)+". Diode", (components[i][0],components[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, size, 0,width)
                cv2.imshow("img",img)
                cv2.waitKey(0)
            print("connections")
            for i in range (0,len(connections)):
                print("component ",connections[i][0]," is connected to: component",connections[i][1])
                
