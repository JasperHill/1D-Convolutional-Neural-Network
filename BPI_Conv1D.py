#########################################################################
## a rudimentary program to train a single convolutional 1D layer on JSON-formatted Bitcoin data
#########################################################################
import argparse
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf
import tensorflow.keras  as keras
import math

parser           = argparse.ArgumentParser(description='train a 1D convolutional neural network')
parser.add_argument('filename',help='the datafile on which the neural network is to be trained')
args             = parser.parse_args()
datafile         = args.filename

train_type       =                                      'log'
#only works for properly-formatted JSON objects
#########################################################################
##  import and format price data
#########################################################################

rawdata       = pd.read_json(path_or_buf=datafile, orient='records', typ='frame')
rawtdata      =                      rawdata.drop(columns=['average','high','low','open'])
rawpdata      =                                              rawdata.drop(columns=['time'])

tdata         =                                           pd.to_datetime(rawtdata['time'])
UTC_tdata     =                 (tdata - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

np_UTC_tdata  =                                              np.flip(UTC_tdata.to_numpy())
np_pdata      =                               np.transpose(np.fliplr(rawpdata.to_numpy()))

#########################################################################
##  prepare prices, derivatives, and times
#########################################################################


dt            =          np_UTC_tdata[1] - np_UTC_tdata[0]##the time increment
t             = (np_UTC_tdata[2:] - np_UTC_tdata[2]) // dt##an array of times for plotting
length        =                                     len(t)##the total size of the data

B                =                          np_pdata[:,2:]
B_prime          =             np.diff(np_pdata,n=1)[:,1:]
B_doubleprime    =                   np.diff(np_pdata,n=2)

logB             =                  np.log10(np_pdata) + 2###add 2 so that all prices are normalized to the starting BTC price ($0.01)
logB_prime       =                 np.diff(logB,n=1)[:,1:]
logB_doubleprime =                       np.diff(logB,n=2)

logB             =                               logB[:,2:]

#########################################################################
##  create graphs
#########################################################################

plt.figure(figsize=(18,6))
plt.subplot(131)
plt.title('Value of Bitcoin over time')
plt.xlabel('time (h)')
plt.ylabel('value ($)')
plt.plot(t,B[0])

plt.subplot(132)
plt.title('First Derivative')
plt.xlabel('time (h)')
plt.ylabel('change ($)')
plt.plot(t,B_prime[0])

plt.subplot(133)
plt.title('Second Derivative')
plt.xlabel('time (h)')
plt.ylabel('change ($)')
plt.plot(t,B_doubleprime[0])

plt.savefig('Plot.pdf')
plt.close()

#
## log plots ##
#

plt.figure(figsize=(18,6))
plt.subplot(131)
plt.title('Value of Bitcoin over time')
plt.xlabel('time (h)')
plt.ylabel('value ($)')
plt.plot(t,logB[0])

plt.subplot(132)
plt.title('First Derivative')
plt.xlabel('time (h)')
plt.ylabel('change ($)')
plt.plot(t,logB_prime[0])

plt.subplot(133)
plt.title('Second Derivative')
plt.xlabel('time (h)')
plt.ylabel('change ($)')
plt.plot(t,logB_doubleprime[0])

plt.savefig('logPlot.pdf')
plt.close()

#########################################################################
##  create training and testing data
#########################################################################

frac             =                                      1/50 ##fraction of total data set in each sample
target_frac      =                                      1/50 ##size of the prediction window as a fraction of the entire data window
num_of_xpoints   =                                         3 ##for 0th, 1st, and 2nd derivatives
num_of_samples   =                                       100 ##number of training samples
window           =                   math.floor(frac*length) ##time span of each sample
target_window    =            math.floor(target_frac*length) ##time span of target output                    

avgB                =                                      B[0]
avgB_prime          =                                B_prime[0]
avgB_doubleprime    =                          B_doubleprime[0]

logavgB             =                                   logB[0]
logavgB_prime       =                             logB_prime[0]
logavgB_doubleprime =                       logB_doubleprime[0]

x_train           = np.empty([num_of_samples,num_of_xpoints,window])
y_train           = np.empty([num_of_samples,1,target_window])
full_y_train      = np.empty([num_of_samples,num_of_xpoints,target_window])

x_test            = np.empty([num_of_samples,num_of_xpoints,window])
y_test            = np.empty([num_of_samples,1,target_window])
full_y_test       = np.empty([num_of_samples,num_of_xpoints,target_window])

temp1 = np.empty([window,num_of_xpoints])
temp2 = np.empty([window,1])

print('Training Summary:')
print('Number of samples: ',num_of_samples, '|| Window: ',window, '|| Target window: ', target_window)

if (train_type == 'log'):
    for i in range(num_of_samples):
        x_train[i]      = [logavgB[i:(window+i)],
                           logavgB_prime[i:(window+i)],
                           logavgB_doubleprime[i:(window+i)]
                           ]

        full_y_train[i] = [logavgB[(window+i+1):(window+target_window+i+1)],
                           logavgB_prime[(window+i+1):(window+target_window+i+1)],
                           logavgB_doubleprime[(window+i+1):(window+target_window+i+1)]
                           ]

        y_train[i]      = logavgB[(window+i+1):(window+target_window+i+1)]


##
####  prepare log training data
##

        
        x_test[i]       = [logavgB[(window+i):(2*window+i)],
                           logavgB_prime[(window+i):(2*window+i)],
                           logavgB_doubleprime[(window+i):(2*window+i)]
                           ]

        y_test[i]       = logavgB[(2*window+i+1):(2*window+target_window+i+1)]

        full_y_test[i]  = [logavgB[(2*window+i+1):(2*window+target_window+i+1)],
                           logavgB_prime[(2*window+i+1):(2*window+target_window+i+1)],
                           logavgB_doubleprime[(2*window+i+1):(2*window+target_window+i+1)]
                           ]

##
####  prepare log testing data
##



if (train_type == 'linear'):
    for i in range(num_of_samples):
        x_train[i]      = [avgB[i:(window+i)],
                           avgB_prime[i:(window+i)],
                           avgB_doubleprime[i:(window+i)]
                           ]

        full_y_train[i] = [avgB[(window+i+1):(window+target_window+i+1)],
                           avgB_prime[(window+i+1):(window+target_window+i+1)],
                           avgB_doubleprime[(window+i+1):(window+target_window+i+1)]
                           ]

        y_train[i]      = avgB[(window+i+1):(window+target_window+i+1)]

##
####  prepare linear testing data  ## large price values are intractable without normalization ## log data is more economical
##

        x_test[i]       = [avgB[(window+i):(2*window+i)],
                           avgB_prime[(window+i):(2*window+i)],
                           avgB_doubleprime[(window+i):(2*window+i)]
                           ]

        y_test[i]       = avgB[(2*window+i+1):(2*window+target_window+i+1)]

        full_y_test[i]  = [avgB[(2*window+i+1):(2*window+target_window+i+1)],
                           avgB_prime[(2*window+i+1):(2*window+target_window+i+1)],
                           avgB_doubleprime[(2*window+i+1):(2*window+target_window+i+1)]
                           ]
            
##
####  prepare linear testing data
##


#########################################################################
##  create neural network
#########################################################################

EPOCHS = 1000
epoch_minorstep = EPOCHS/100
epoch_majorstep = EPOCHS/10

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if (epoch != 0):
            if (epoch % epoch_minorstep == 0):
                print('.',end='')
                if (epoch % epoch_majorstep == 0):
                    print('')

## custom class (defunct)
#########################################################################
class MemoryLayer(keras.layers.Layer):
    def __init__(self, num_of_outputs):
        super(MemoryLayer,self).__init__(name='')

        #self.flatten0 = keras.layers.Flatten(input_shape=(num_of_xpoints,window))
        self.short_dense = keras.layers.Dense(1,activation='tanh',input_shape=(x_train.shape))
        self.med_dense   = keras.layers.Dense(1,activation='tanh',input_shape=(1,))
        self.long_dense  = keras.layers.Dense(1,activation='tanh',input_shape=(1,))

        self.mem_output  = tf.keras.layers.Dense(num_of_outputs,activation='tanh',input_shape=(3,))
        self.flatten1 = keras.layers.Flatten(input_shape=(num_of_xpoints,window))
        #self.flatten1 = keras.layers.Flatten(input_shape=(num_of_outputs,))


    def call(self, input_tensor):
        #temp = self.flatten0(input_tensor)
        x_sd = self.short_dense(input_tensor)
        x_md = self.med_dense(x_sd)
        x_ld = self.long_dense(x_md)

        temp = tf.convert_to_tensor([x_sd,x_md,x_ld])
        mem = self.mem_output(temp)
        out = self.flatten1(mem)

        return out
    #return temp

#########################################################################

Model = keras.models.Sequential([keras.layers.Conv1D(filters=target_window,kernel_size=[num_of_xpoints],input_shape=(num_of_xpoints,window))
                                 ])
Model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])
Model.fit(x_train[:1],y_train[:1],epochs=1, steps_per_epoch=1, callbacks=[PrintDot()], verbose=0)
Model.summary()

Model.fit(x_train,y_train,epochs=EPOCHS, steps_per_epoch=num_of_samples, callbacks=[PrintDot()], verbose=0)
print('')

prediction_window = target_window
y_predicted = Model.predict(x_test,steps=num_of_samples)

#########################################################################
##  create graphs
#########################################################################

length = y_test.shape[0]
y_test = np.power(10,(y_test - 2))
y_predicted = np.power(10,(y_predicted - 2))

plt.figure(figsize=(18,6))
plt.title('Value of Bitcoin over time')
plt.xlabel('time (h)')
plt.ylabel('value ($)')

plt.plot(t[0:target_window],np.transpose(np.asarray(y_test[length-1])), label='actual')
plt.plot(t[0:target_window],np.transpose(np.asarray(y_predicted[num_of_samples-1])), label='prediction')

plt.legend()
plt.savefig('NetworkData.pdf')
plt.close()


