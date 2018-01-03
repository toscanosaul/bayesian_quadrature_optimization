
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from scipy.misc import comb, logsumexp
import math


train_samples = 30000
test_size = 10000
number_classes = 10
fit_intercept = True

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])

X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

number_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=test_size, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y_train = np.zeros((len(y_train), number_classes))
for i,j in enumerate(y_train):
    Y_train[i, int(j)] = 1

Y_test = np.zeros((len(y_test), number_classes))
for i,j in enumerate(y_test):
    Y_test[i, int(j)] = 1


def loss_function(X, Y, w, alpha=0):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    w = w.reshape(n_classes, -1)
    fit_intercept = w.size == (n_classes * (n_features + 1))
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w

def grad_loss(X, Y, w, alpha=0):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = loss_function(X, Y, w,alpha)
    diff = (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def hessian_loss(X, Y, w, alpha=0):
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))

    loss, grad, p = grad_loss(X, Y, w, alpha)

    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()

    w_copy = w.copy()
    w_copy[-1] = 0.0
    return grad, hessp, w_copy


def train_logistic(
        momentum=0.9, lr=0.01, batch_size=1000, alpha=0.1, maxepoch=50, adam=False, betas=None,
        eps=1e-8):

    pairs_tr = X_train.shape[0]
    if betas is None:
        betas = (0.9, 0.999)

    w0 = np.zeros((number_classes, number_features + int(fit_intercept)),
                  order='F', dtype=X.dtype)
    np.random.seed(1)
    w0 = 0.1 * np.random.randn(number_classes, number_features + int(fit_intercept))
    w0 = w0.ravel()
    v = np.zeros(len(w0))
    num_batches = int(math.ceil(pairs_tr / float(batch_size)))
    m0 = np.zeros(len(w0))
    v0 = np.zeros(len(w0))
    t_ = 0

    n_hyperparam = 1
    Z = np.zeros((2.0 * len(w0), n_hyperparam))

    for epoch in range(0, maxepoch):
        shuffled_order = np.arange(pairs_tr)
        np.random.shuffle(shuffled_order)
        previous = w0.copy()
        for batch in range(num_batches):
            t_ += 1
            next_ = min(batch_size * (batch + 1), pairs_tr)
            batch_idx = np.arange(batch_size * batch, next_)
            res = batch_size * (batch + 1) - next_
            if res > 0:
                extra = np.arange(0, res)
                batch_idx = np.concatenate((extra, batch_idx))
            length_batch = len(batch_idx)
            batch_x = np.array(X_train[shuffled_order[batch_idx],:])
            batch_y = np.array(Y_train[shuffled_order[batch_idx],:])
            g, hess, hess_hyper = hessian_loss(batch_x,batch_y,w0,alpha)
            old_w = w0.copy()
            g = g / float(length_batch)
            if not adam:
                old_v = v.copy()
                v = momentum * v + g
                increament = - lr
                w0 = w0 - lr * v
            else:
                increament = 0.0
                m0 = betas[0] * m0 + (1 - betas[0]) * g
                v0 = betas[1] * v0 + (1 - betas[1]) * ((g) ** 2)
                m_1 = m0 / (1 - (betas[0]) ** (t_))
                v_1 = v0 / (1 - (betas[1]) ** (t_))
                w0 = w0 - lr * m_1 / (np.sqrt(v_1) + eps)
            ## gradient only respect to regularization parameter
            B = increament * hess_hyper[:,np.newaxis] / float(length_batch)
            C = hess_hyper[:,np.newaxis] / float(length_batch)
            B = np.concatenate((B, C), axis=0)

            Z_list = []
            old_Z = Z.copy()
            for i in range(n_hyperparam):
                R = hess(Z[0:len(w0), i])
                val = (increament / float(length_batch)) * R + Z[0:len(w0), i]
                val += (increament * momentum * Z[len(w0):, i])

                val_2 = R / float(length_batch) + momentum * Z[len(w0):, i]
                val = np.concatenate((val, val_2))

                Z_list.append(val)
            Z = np.array(Z_list).T
            Z += B
        den_norm = (np.sqrt(np.sum(previous ** 2)))
        if den_norm == 0:
            norm = np.sqrt(np.sum((previous - w0) ** 2)) / 1e-2
        else:
            norm = np.sqrt(np.sum((previous - w0) ** 2)) / den_norm
        if norm < 0.01:
            break

    Z = Z[0:len(w0), :]
    loss, grad_l, p = grad_loss(X_test, Y_test, w0, alpha=0)
    grad_l = grad_l[np.newaxis, :] / float(X_test.shape[0])
    grad = np.dot(grad_l, Z)
    grad_2 = np.dot(g, Z)
    return w0, grad, loss / float(X_test.shape[0]), grad_2

