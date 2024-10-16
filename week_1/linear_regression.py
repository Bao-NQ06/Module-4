import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data , index):
    result = [round(row[index],1)for row in data]

    return result

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset , delimiter =',', skip_header =1).tolist()
    N = len(data)

    tv_data = get_column(data , 0)
    radio_data = get_column(data , 1)
    newspaper_data = get_column(data , 2)
    sales_data = get_column(data , 3)

    X = [ tv_data , radio_data , newspaper_data ]
    y = sales_data
    return X , y

X , y = prepare_data('week_1/advertising.csv')
list = [ sum(X [0][:5]), sum(X [1][:5]), sum(X [2][:5]), sum(y [:5])]
# print(list)

def initialize_params():
    w1 , w2 , w3 , b =(0.016992259082509283 , 0.0070783670518262355 ,-0.002307860847821344 , 0)
    return w1 , w2 , w3 , b

def predict(x1 , x2 , x3 , w1 , w2 , w3 , b):
    result = x1 * w1 + x2 * w2 + x3 * w3 + b

    return result

def compute_loss_mse(y_hat , y):

    loss =round((y_hat - y)**2, 2)
    return loss

def compute_loss_mae(y_hat , y):

    loss =abs(y_hat - y)
    return loss

def compute_gradient_wi(xi , y , y_hat):
    dl_dwi = 2 * xi *(y_hat - y)
    return dl_dwi

def compute_gradient_b(y , y_hat):
    dl_db = 2 *(y_hat - y)
    return dl_db

def update_weight_wi(wi , dl_dwi , lr):
    wi = wi - lr * dl_dwi
    return wi

def update_weight_b(b , dl_db , lr):
    b = b - lr * dl_db
    return b

def implement_linear_regression(X_data , y_data , epoch_max = 50 , lr = 1e-5):
    losses = []

    w1 , w2 , w3 , b = initialize_params()

    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
        # get a sample
            x1 = X_data [0][ i ]
            x2 = X_data [1][ i ]
            x3 = X_data [2][ i ]

            y = y_data [ i ]

            # compute output
            y_hat = predict(x1 , x2 , x3 , w1 , w2 , w3 , b)

            # compute loss
            loss = compute_loss_mse(y_hat, y)

            # compute gradient w1 , w2 , w3 , b
            dl_dw1 = compute_gradient_wi(x1 , y , y_hat)
            dl_dw2 = compute_gradient_wi(x2 , y , y_hat)
            dl_dw3 = compute_gradient_wi(x3 , y , y_hat)
            dl_db = compute_gradient_b(y , y_hat)
            w1 = update_weight_wi(w1 , dl_dw1 , lr)
            w2 = update_weight_wi(w2 , dl_dw2 , lr)
            w3 = update_weight_wi(w3 , dl_dw3 , lr)
            b = update_weight_b(b , dl_db , lr)
            
            # logging
            losses.append(loss)
    return(w1 , w2 , w3 ,b , losses)
    


def implement_linear_regression_nsamples(X_data , y_data , epoch_max =50 , lr=1e-5):
    losses = []
    w1 , w2 , w3 , b = initialize_params ()
    N = len( y_data )

    for epoch in range ( epoch_max ) :

        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0
        
        for i in range ( N ) :
        # get a sample
            x1 = X_data [0][ i ]
            x2 = X_data [1][ i ]
            x3 = X_data [2][ i ]
            
            y = y_data [ i ]
            
            # compute output
            y_hat = predict ( x1 , x2 , x3 , w1 , w2 , w3 , b )
            
            # compute loss
            loss = compute_loss_mse(y_hat , y)
            
            # accumulate loss
            loss_total += loss
            # compute gradient w1 , w2 , w3 , b
            
            dl_dw1 = compute_gradient_wi ( x1 , y , y_hat )
            dl_dw2 = compute_gradient_wi ( x2 , y , y_hat )
            dl_dw3 = compute_gradient_wi ( x3 , y , y_hat )
            dl_db = compute_gradient_b (y , y_hat )
            
            # accumulate gradient w1 , w2 , w3 , b
            
            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db
            
            # ( after processing N samples ) - update parameters
            w1 = update_weight_wi ( w1 , dw1_total , lr )
            w2 = update_weight_wi ( w2 , dw2_total , lr )
            w3 = update_weight_wi ( w3 , dw3_total , lr )
            b = update_weight_b ( b , db_total , lr )
            
            
            
        # logging
        losses.append(loss_total/N)
    return ( w1 , w2 , w3 ,b , losses )


def prepare_data ( file_name_dataset ) :
    data = np.genfromtxt(file_name_dataset , delimiter =',', skip_header =1).tolist()
    
    tv_data = get_column ( data , 0)
    radio_data = get_column ( data , 1)
    newspaper_data = get_column ( data , 2)
    sales_data = get_column ( data , 3)
    X = [[1 , x1 , x2 , x3 ] for x1 , x2 , x3 in zip( tv_data , radio_data , newspaper_data ) ]
    y = sales_data
    return X , y

def initialize_params () :
    bias = 0
    w1 = random.gauss ( mu =0.0 , sigma =0.01)
    w2 = random.gauss ( mu =0.0 , sigma =0.01)
    w3 = random.gauss ( mu =0.0 , sigma =0.01)

    # comment this line for real application
    return [0 , -0.01268850433497871 , 0.004752496982185252 , 0.0073796171538643845]

def predict ( X_features , weights ) :
    result = sum([ x * w for x , w in zip( X_features , weights )])
    return result

def compute_loss ( y_hat , y ) :
    return ( y_hat - y ) **2

def compute_gradient_w ( X_features , y , y_hat ) :
    dl_dweights = [2 * x *( y_hat - y ) for x in X_features ]
    return dl_dweights

def update_weight ( weights , dl_dweights , lr ) :
    weights = [ w - lr * dl_dw for w , dl_dw in zip( weights , dl_dweights )]
    return weights

def implement_linear_regression ( X_feature , y_output , epoch_max =50 , lr =1e-5) :

    losses = []
    weights = initialize_params ()
    N = len( y_output )
    for epoch in range ( epoch_max ) :
        print (" epoch ", epoch )
        for i in range ( N ) :
        # get a sample - row i
            features_i = X_feature [ i ]
            y = y_output[ i ]
            
            # compute output
            y_hat = predict ( features_i , weights )
            
            # compute loss
            loss = compute_loss (y , y_hat )
            
            # compute gradient w1 , w2 , w3 , b
            dl_dweights = compute_gradient_w ( features_i , y , y_hat )
            
            # update parameters
            weights = update_weight ( weights , dl_dweights , lr )
            
            # logging
            losses . append ( loss )
    return weights , losses
            

X , y = prepare_data('week_1/advertising.csv')
W , L = implement_linear_regression(X , y , epoch_max =50 , lr =1e-5)
# Print loss value at iteration 9999
print ( L [9999])