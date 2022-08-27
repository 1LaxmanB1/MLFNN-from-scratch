from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import math

diabetes = datasets.load_diabetes(as_frame=True)
diadf = diabetes['frame']
ddf = diadf.drop(['sex'],axis=1)
ddf=ddf[1:400]
X=ddf.drop(['target'],axis=1)
Y=ddf['target']
Xnp=np.array(X)
Ynp=np.array(Y)
# print(Ynp)
Ynp=Ynp.reshape((1,399))
# print(Ynp.shape)
# print(Ynp)
# print("Basic Input taken into training")
# print(Xnp)


n_hd_l = 8   # Number of hidden layers
n_nue_hd = 5 # Number of neurons in each hidden layer & all hidden layers have same number of neurons
n_imput_dim = 9

def wbinitialise(hd,hdn,inp) :
    weights_i = np.random.random((hdn,inp))
    weights = [np.random.random((hdn,hdn)) for x in range(hd-1)]
    biases = [np.random.random((hdn,1)) for x in range(hd)]
    weights_o = np.random.random((1,hdn))
    weights.insert(0,weights_i)
    weights.append(weights_o)
    biases.append(np.random.random((1,1)))
    # print(biases[1])

    return weights,biases


def sigmoid(input):
    # print("Input:")
    # print(input)
    # print("Sigmoid out :")
    # print(1/(1+np.exp(-input)))
    return 1/(1+np.exp(-input))


def forwardpp(input,WgtM,Biasv,nhd,truout):                            # Simple case assuming all layers have
                                                                       # sigmoid activation fn
      a=[]
      h=[]
      inc=np.transpose(input)                  # This transformation sets the rows to be features and columns to be different
      # print(inc.shape)                                         # datapoints

      for x in range(nhd):
          # print("Weight matrix in layer ",x+1)
          # print(WgtM[x].shape)
          # print("Input vector in layer ", x + 1)
          # print(inc)
          # print(inc.shape)

          # print(x)
          # print(WgtM[x].shape)
          # print(inc.shape)
          a_ins = WgtM[x] @ inc + Biasv[x]               # @ signifies matrix multiplication
          h_ins = sigmoid(a_ins)
          a.append(a_ins)
          h.append(h_ins)
          inc = h_ins

      a_ins = np.matmul(WgtM[-1],h_ins)+Biasv[-1]
      a.append(a_ins)
      yp=1*a_ins                                                     # Output layer activation fn is 1 (Linear)
      # print(yp.shape)
      # print(truout.shape)
      loss = lossfn(truout, yp)

      return h,a,yp,loss             # We get the dimension of output matrix to be
                              # (number of datapoints, number of neurons)

def lossfn(Yt,Yp):
    # print(Yt.shape)
    # print(Yp.shape)
    sqerror = np.square(Yt-Yp)/2
    loss = np.mean(sqerror)

    return loss

def sigderv(input):
    return np.multiply(sigmoid(input),(1-sigmoid(input)))


def backpropagation(hlist,alist,Yt,Yp,hd,wgtlist,Xt):
    # print("Back propagation fn is run")
    outlaygrad_a = -1*(Yt - Yp)

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
        presentlaygrad_b = np.sum(presentlaygrad_a,axis=1)
        if x == hd:
           size = (1,1)
        else:
            size = (5,1)

        presentlaygrad_b = presentlaygrad_b.reshape(size)
        # print(presentlaygrad_b)
        prevlaygrad_h = np.matmul(np.transpose(wgtlist[x]),presentlaygrad_a)
        prevlaygrad_a = np.multiply(prevlaygrad_h,sigderv(alist[x-1]))
        grad_w_list.append(presentlaygrad_w)
        grad_b_list.append(presentlaygrad_b)
        grad_h_list.append(prevlaygrad_h)
        grad_a_list.append(presentlaygrad_a)
        presentlaygrad_a = prevlaygrad_a
        # print(presentlaygrad_w.shape)

    # print(presentlaygrad_a.shape)
    print(Xt.shape)
    grad_w_lay1 = np.matmul((presentlaygrad_a),Xt)
    # print(grad_w_lay1.shape)

    grad_b_lay1 = np.sum(presentlaygrad_a,axis=1)
    grad_b_lay1 = grad_b_lay1.reshape(size)

    grad_w_list.append(grad_w_lay1)
    grad_b_list.append(grad_b_lay1)
    grad_a_list.append(presentlaygrad_a)

    grad_w_list.reverse()
    grad_b_list.reverse()
    grad_h_list.reverse()
    grad_a_list.reverse()


    return grad_w_list,grad_b_list,grad_h_list,grad_a_list




def gd(wgtlist,biaslist,gradWlist,gradblist,nhl):       # Gradient Descent
    upWlist = []
    upblist = []
    n = 0.0001
    for x in range(nhl+1):
        # print("Weight matrix shape in layer ",x+1)
        # print(wgtlist[x].shape)
        # print("Grad Weight matrix shape in layer ",x+1)
        # print(gradWlist[x].shape)
        upWlist.append(wgtlist[x]-n*gradWlist[x])
        # print("Bias vector shape in layer ",x+1)
        # print(biaslist[x].shape)
        # print("Grad Bias vector shape in layer ",x+1)
        # print(gradblist[x].shape)
        upblist.append(biaslist[x]-n*gradblist[x])

    return upWlist,upblist


def adam (wgtlist,biaslist,gradWlist,gradblist,nhl,mw,vw,mb,vb,count):

    beta1 = 0.5
    beta2 = 0.5
    eps = 0.01
    n = 0.1

    mhatw = []
    vhatw = []
    mhatb = []
    vhatb = []
    upWlist = []
    upblist = []

    for x in range(nhl+1):
        # print(x)
        mw[x] = beta1 * mw[x] + (1-beta1)*gradWlist[x]
        vw[x] = beta2 * vw[x] + (1 - beta2) * np.square(gradWlist[x])
        mhatw.append(mw[x] / (1-beta1**count))
        vhatw.append(vw[x] / (1 - beta2**count))

        upWlist.append(wgtlist[x] - n*np.divide(mhatw[x],np.sqrt(vhatw[x]+eps)))



        mb[x] = beta1 * mb[x] + (1 - beta1) * gradblist[x]
        vb[x] = beta2 * vb[x] + (1 - beta2) * np.square(gradblist[x])
        mhatb.append(mb[x] / (1-beta1**count))
        vhatb.append(vb[x] / (1 - beta2**count))

        upblist.append(biaslist[x] - n * np.divide(mhatb[x], np.sqrt(vhatb[x] + eps)))

    return upWlist,upblist,mw,vw,mb,vb





def MLFNN(Trainingdata,Weightsl,Biasl,num_hiddn_l,target):
    H,A,yout,loss = forwardpp(Trainingdata,Weightsl,Biasl,num_hiddn_l,target)
    count = 1
    newW=Weightsl
    update = 'gd'

    mw = [np.zeros(Weightsl[x].shape) for x in range(num_hiddn_l+1)]
    vw = [np.zeros(Weightsl[x].shape) for x in range(num_hiddn_l+1)]
    mb = [np.zeros(Biasl[x].shape) for x in range(num_hiddn_l+1)]
    vb = [np.zeros(Biasl[x].shape) for x in range(num_hiddn_l+1)]


    # while(loss > 0.01):

    for x in range(10):
        print("Loss value after ",count , "epochs :")
        print(loss)
        gWlist,gblist,ghlist,galist = backpropagation(H,A,target,yout,n_hd_l,newW,Trainingdata)
        if (update == 'gd'):
            newW,newB = gd(Weightsl,Biasl,gWlist,gblist,num_hiddn_l)
        elif (update == 'adam'):
            newW, newB,mw,vw,mb,vb = adam(Weightsl,Biasl,gWlist,gblist,num_hiddn_l,mw,vw,mb,vb,count)

        H,A,yout,loss = forwardpp(Xnp,newW,newB,n_hd_l,Ynp)

        # plt.plot(count,math.log(loss,10),'r')
        count = count+1

    print('Final Loss values after ',count-1,' epochs :')
    print(loss)

    # plt.title("Loss vs Epoch")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss values")
    # plt.show()

Wmat,Bmat=wbinitialise(n_hd_l,n_nue_hd,n_imput_dim)
MLFNN(Xnp,Wmat,Bmat,n_hd_l,Ynp)