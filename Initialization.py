from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import math

diabetes = datasets.load_diabetes(as_frame=True)
diadf = diabetes['frame']
ddf = diadf.drop(['sex'],axis=1)
ddf=ddf[1:6]
X=ddf.drop(['target'],axis=1)
Y=ddf['target']
Xnp=np.array(X)
Ynp=np.array(Y)
Ynp=Ynp.reshape((1,5))
# print(Ynp.shape)
# print(Ynp)
# print("Basic Input taken into training")
# print(Xnp)


n_hd_l = 5   # Number of hidden layers
n_nue_hd = 4 # Number of neurons in each hidden layer & all hidden layers have same number of neurons
n_imput_dim = 9

def wbinitialise(hd,hdn,inp) :
    weights_i = np.random.random((hdn,inp))
    weights = [np.random.random((hdn,hdn)) for x in range(hd-1)]
    biases = [np.random.random((hdn,1)) for x in range(hd)]
    weights_o = np.random.random((1,hdn))
    weights.insert(0,weights_i)
    weights.append(weights_o)
    biases.append(np.random.random((1,1)))
    # print(biases[-1])

    return weights,biases

Wmat,Bmat=wbinitialise(n_hd_l,n_nue_hd,n_imput_dim)

def sigmoid(input):
    print("Input:")
    print(input)
    print("Sigmoid out :")
    print(1/(1+np.exp(-input)))
    return 1/(1+np.exp(-input))


def forwardpp(input,WgtM,Biasv,nhd,truout):                            # Simple case assuming all layers have
                                                                       # sigmoid activation fn
      a=[]
      h=[]
      inc=np.transpose(input)                  # This transformation sets the rows to be features and columns to be different
                                               # datapoints

      for x in range(nhd):
          # print("Weight matrix in layer ",x+1)
          # print(WgtM[x].shape)
          # print("Input vector in layer ", x + 1)
          # print(inc)
          # print(inc.shape)

          # print(x)
          a_ins = np.matmul(WgtM[x], inc) + Biasv[x]
          h_ins = sigmoid(a_ins)
          a.append(a_ins)
          h.append(h_ins)
          inc = h_ins

      a_ins = np.matmul(WgtM[-1],h_ins)+Biasv[-1]
      a.append(a_ins)
      yt=1*a_ins                                                     # Output layer activation fn is 1 (Linear)
      loss = lossfn(truout, yt)

      return h,a,yt,loss             # We get the dimension of output matrix to be
                              # (number of datapoints, number of neurons)

def lossfn(Yt,Yp):
    # print(Yt.shape)
    # print(Yp.shape)
    sqerror = np.square(Yt-Yp)/2
    loss = np.mean(sqerror)

    return loss

def sigderv(input):
    return np.multiply(sigmoid(input),(1-sigmoid(input)))


def backpropagation(loss,hlist,alist,Yt,Yp,hd,wgtlist,Xt):
    # print("Back propagation fn is run")
    outlaygrad_a = -1*(np.subtract(Yt,Yp))

    presentlaygrad_a = outlaygrad_a
    grad_w_list = []
    grad_b_list = []
    grad_h_list = []
    grad_a_list = []
    for x in range(hd,0,-1):
        # print(x)
        # print(presentlaygrad_a.shape)
        # print(np.transpose(hlist[x-1]).shape)
        presentlaygrad_w = np.matmul((presentlaygrad_a),np.transpose(hlist[x-1]))
        presentlaygrad_b = presentlaygrad_a
        prevlaygrad_h = np.matmul(np.transpose(wgtlist[x]),presentlaygrad_a)
        prevlaygrad_a = np.multiply(prevlaygrad_h,sigderv(alist[x-1]))
        grad_w_list.append(presentlaygrad_w)
        grad_b_list.append(presentlaygrad_b)
        grad_h_list.append(prevlaygrad_h)
        grad_a_list.append(presentlaygrad_a)
        presentlaygrad_a = prevlaygrad_a
        # print(presentlaygrad_w.shape)

    # print(presentlaygrad_a.shape)
    # print(Xt.shape)
    grad_w_lay1 = np.matmul((presentlaygrad_a),Xt)
    # print(grad_w_lay1.shape)

    grad_b_lay1 = presentlaygrad_a

    grad_w_list.append(grad_w_lay1)
    grad_b_list.append(grad_b_lay1)
    grad_a_list.append(presentlaygrad_a)

    grad_w_list.reverse()
    grad_b_list.reverse()
    grad_h_list.reverse()
    grad_a_list.reverse()


    return grad_w_list,grad_b_list,grad_h_list,grad_a_list




def gd(wgtlist,biaslist,gradWlist,gradblist,nhl):       # Gradient Descent
    upWlist=[]
    upblist=[]

    for x in range(nhl+1):
        # print(wgtlist[x].shape)
        # print(gradWlist[x].shape)
        upWlist.append(np.subtract(wgtlist[x],gradWlist[x]))
        upblist.append(np.subtract(biaslist[x],gradblist[x]))

    return upWlist,upblist



def MLFNN(Trainingdata,Weightsl,Biasl,num_hiddn_l,target):
    H,A,yout,loss = forwardpp(Trainingdata,Weightsl,Biasl,num_hiddn_l,target)
    count = 1
    newW=Weightsl
    # while(loss > 0.01):

    for x in range(7):
        print("Loss value after ",count , "epochs :")
        print(loss)
        gWlist,gblist,ghlist,galist = backpropagation(loss,H,A,target,yout,n_hd_l,newW,Trainingdata)
        newW,newB = gd(Weightsl,Biasl,gWlist,gblist,num_hiddn_l)
        H,A,yout,loss = forwardpp(Xnp,newW,newB,n_hd_l,Ynp)

        # plt.plot(count,math.log(loss,10),'r')
        count = count+1

    print('Final Loss values after ',count,' epochs :')
    print(loss)

    # plt.title("Loss vs Epoch")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss values")
    # plt.show()

MLFNN(Xnp,Wmat,Bmat,n_hd_l,Ynp)