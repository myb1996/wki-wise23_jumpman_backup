import numpy as np

    
def CSPMF(EEG_Signals, label, channels : int, count_feature_vector):
    Channel_count = channels
    Sample_count = len(EEG_Signals)
    class_labels = [1, 0]
    # EEG_Signals = EEG_train
    # label = np.array(label_train)
    m = count_feature_vector
    

    cov0 = EEG_Signals[label[:, 0] == 0]
    cov1 = EEG_Signals[label[:, 0] == 1]
    
    MatrixCov0 = np.zeros((Channel_count, Channel_count, len(cov0)))
    MatrixCov1 = np.zeros((Channel_count, Channel_count, len(cov1)))
    for i in range(len(cov0)):
            E = cov0[i].T
        #     print(E.shape)
            EE = np.dot(E.T,E)
            MatrixCov0[:,:,i] = EE/np.trace(EE);
    # print(MatrixCov0.shape)
    # print(MatrixCov0)
    
    for i in range(len(cov1)):
            E = cov1[i].T
        #     print(E.shape)
            EE = np.dot(E.T,E)
            MatrixCov1[:,:,i] = EE/np.trace(EE);
    # print(MatrixCov1.shape)
    
    np.set_printoptions(precision=8)
    cov0 = np.mean(MatrixCov0,2)
    cov1 = np.mean(MatrixCov1,2)
    covTotal = cov0 + cov1
#     print(cov0.shape)
#     print(cov1.shape)
#     print(covTotal)

# Calculate common eigenvector matrix and eigenvalues
    [E_val,E_vec] = np.linalg.eigh(covTotal)
    # Sort descending
    E_vec = E_vec[:,E_val.argsort()[::-1]]
    E_val = sorted(E_val,reverse=True)

    # matrix whitening
    E_val = np.diag(E_val)
    E_val = np.sqrt(1./E_val)

#     # Remove infinite values
    c = np.isinf(E_val)
    E_val[c] = 0
#     print('E_val:', E_val)
    P = np.dot(E_val,E_vec.T)
    
    transformedCov0 = np.dot(np.dot(P,cov0),P.T)
    transformedCov1 = np.dot(np.dot(P,cov1),P.T)
#     print(transformedCov0)
#     print(transformedCov1)

    # Decompose transformedCov1 into principal components to obtain the public eigenvector matrix E_val0
    [E_val0,E_vec0] = np.linalg.eig(transformedCov0)
    # ascending order
    E_vec0 = E_vec0[:,E_val0.argsort()]
    E_val0 = np.sort(E_val0)
    # print(E_vec0)
    # print(E_val0)


    # Decompose transformedCov1 into principal components to obtain the public eigenvector matrix E_val1
    [E_val1,E_vec1] = np.linalg.eig(transformedCov1)
    # descending order
    E_vec1 = E_vec1[:,E_val1.argsort()]
    E_val1 = sorted(E_val1,reverse=True)
    # print(E_vec1)
    # print(E_val1)
    
    selected_vec0 = np.hstack((E_vec0[:, :m], E_vec0[:, -m:]))  # select vec from E_vec0
    selected_vec1 = np.hstack((E_vec1[:, :m], E_vec1[:, -m:]))  # select vec from E_vec1

    # build matrix B
    B = np.hstack((selected_vec0, selected_vec1))
#     print(B)
    CSPMatrix = np.dot(B.T,P)
#     print(CSPMatrix.shape)
    return CSPMatrix
