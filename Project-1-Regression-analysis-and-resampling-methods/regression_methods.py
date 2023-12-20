from func import *

class OLS():
    def __init__(self, scaling=None):
        self._scaling = scaling

        self._mse_train = []
        self._mse_test = []
        self._r2_train = []
        self._r2_test = []
        self._beta_list = []
        self._polynomials = []

        self._bias = []
        self._variance = []

        self._called = False

    def fit(self, x, y, z, maxdegree, test_size=0.2):
        self._x = x
        self._y = y
        self._z = z
        self._maxdegree = maxdegree
        
        self._design_matrix(self._x, self._y, maxdegree)

        self._test_size = test_size
        self._X_train_main, self._X_test_main, self._x_train, self._x_test, self._y_train, self._y_test, self._z_train, self._z_test = train_test_split(self._X_main, self._x, self._y, self._z, test_size=self._test_size,random_state=10)

    def _design_matrix(self, x, y, degree, ret=False):
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree+1)*(degree+2)/2)

        X = np.ones((N,l))

        for i in range(1,degree+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

        if ret:
            return X
        else:
            self._X_main = X

    def predict(self, degree, bootstrap=False, n_bootstraps=65, CV=False, k=10):
        if bootstrap:
            if not self._called:
                self._z_pred = np.zeros((len(self._z_test),n_bootstraps))
            self._called = True
            self._bootstrap(degree, n_bootstraps=n_bootstraps)

        elif CV:
            if not self._called:
                self._scores_KFold = np.zeros((self._maxdegree-degree, k))
                self._kfold = KFold(n_splits=k, random_state=1, shuffle=True)

                self._called = True

            self._CV(degree)
            self._polynomials.append(degree)
            
        else:
            # with indercept
            l = int((degree+1)*(degree+2)/2)

            self._X_train, self._X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

            if not self._scaling:
                # Calculating the beta with the training sets
                self._beta = (np.linalg.pinv(self._X_train.T @ self._X_train) @ self._X_train.T) @ self._z_train

                self._z_tilde = self._X_train @ self._beta
                self._z_predict = self._X_test @ self._beta
            
            elif self._scaling=="mean":
                # Scaling
                X_train_mean = np.mean(self._X_train, axis=0)

                X_train_scaled = self._X_train - X_train_mean
                X_test_scaled = self._X_test - X_train_mean

                z_scaler = np.mean(self._z_train)
                z_train_scaled = self._z_train - z_scaler

                # Calculating the beta with the training sets
                self._beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T) @ z_train_scaled

                # Calculating z_tilde and z_predict
                self._z_tilde = X_train_scaled @ self._beta + z_scaler
                self._z_predict = X_test_scaled @ self._beta + z_scaler
            
            mse_train, mse_test = self.mse()
            self._mse_train.append(mse_train)
            self._mse_test.append(mse_test)

            r2_train, r2_test = self.r2()
            self._r2_train.append(r2_train)
            self._r2_test.append(r2_test)

            self._beta_list.append(self._beta)
            self._polynomials.append(degree)

    def mse(self):
        mse_train = mean_squared_error(self._z_train, self._z_tilde)
        mse_test = mean_squared_error(self._z_test, self._z_predict)

        return mse_train, mse_test
    
    def r2(self):
        r2_train = r2_score(self._z_train, self._z_tilde)
        r2_test = r2_score(self._z_test, self._z_predict)

        return r2_train, r2_test
    
    def beta(self):
        return self._beta
    
    def plot_beta(self):
        beta_plot_min = []
        beta_plot_max = []
        for beta in self._beta_list:
            if len(beta) < 2:
                beta_min = beta[0]
                beta_max = beta[0]

            else:
                beta_min = min(beta)
                beta_max = max(beta)

            beta_plot_min.append(beta_min)
            beta_plot_max.append(beta_max)

        plt.plot(self._polynomials, beta_plot_min, label="min beta value")
        plt.plot(self._polynomials, beta_plot_max, label="max beta value")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Beta value")

        plt.xticks(self._polynomials)


    def plot_mse(self):
        if not self._scaling:
            label1 = "MSE train set, no scaling"
            label2 = "MSE test set, no scaling"
        if self._scaling=="mean":
            label1 = "MSE train set, mean scaling"
            label2 = "MSE test set, mean scaling"

        plt.plot(self._polynomials, self._mse_train, "--",label=label1)
        plt.plot(self._polynomials, self._mse_test, label=label2)

        min_mse_train = self._mse_train.index(min(self._mse_train))
        plt.plot(self._polynomials[min_mse_train], self._mse_train[min_mse_train], "y*", 
                label=f"Training minimum: {self._mse_train[min_mse_train]:.4e}")
        min_mse_test =  self._mse_test.index(min( self._mse_test))
        plt.plot(self._polynomials[min_mse_test],  self._mse_test[min_mse_test], "r*", 
                label=f"Test minimum: { self._mse_test[min_mse_test]:.4e}")


        plt.xlabel("Polynomial order")
        plt.ylabel("MSE")
        plt.xticks(self._polynomials)

        print(f"Best MSE for OLS regression is {self._mse_test[min_mse_test]}")

    def plot_r2(self):
        plt.plot(self._polynomials, self._r2_train, "r--", label="R2 train set")
        plt.plot(self._polynomials, self._r2_test, label="R2 test set")

        max_r2_train = self._r2_train.index(max(self._r2_train))
        plt.plot(self._polynomials[max_r2_train], self._r2_train[max_r2_train], "y*", 
                label=f"Training maximum: {self._r2_train[max_r2_train]:.4f}")
        max_r2_test = self._r2_test.index(max(self._r2_test))
        plt.plot(self._polynomials[max_r2_test], self._r2_test[max_r2_test], "r*", 
                label=f"Test maximum: {self._r2_test[max_r2_test]:.4f}")

        plt.xlabel("Polynomial order")
        plt.ylabel("R2 score")
        plt.xticks(self._polynomials)

    def _bootstrap(self, degree, n_bootstraps):
        self._n_bootstraps = n_bootstraps

        l = int((degree+1)*(degree+2)/2)

        X_test = self._X_test_main[:,:l]
        z_test = self._z_test.reshape(-1,1)

        # X_test = self._design_matrix(self._x_test, self._y_test, degree, ret=True)

        for i in range(n_bootstraps):
            # Resampling
            x_train_boot, y_train_boot, z_train_boot = resample(self._x_train, self._y_train, self._z_train, random_state=i)

            # Training design matrix
            X_train = self._design_matrix(x_train_boot, y_train_boot, degree, ret=True)

            # Fit the model
            beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train_boot

            # Calculate predicted value
            self._z_pred[:,i] = X_test @ beta

        self._polynomials.append(degree)

        bias = np.mean((z_test - np.mean(self._z_pred, axis=1,keepdims = True))**2)
        variance = np.mean(np.var(self._z_pred, axis=1, keepdims=True))
        error = np.mean(np.mean((z_test - self._z_pred)**2, axis=1, keepdims=True))

        self._bias.append(bias)
        self._variance.append(variance)
        self._mse_test.append(error)


    def plot_bootstrap(self):
        if len(self._polynomials)==1:
            return self._mse_test, self._bias, self._variance
 
        else:
            plt.plot(self._polynomials, self._mse_test, label='Error')
            plt.plot(self._polynomials, self._bias, label='bias')
            plt.plot(self._polynomials, self._variance, label='Variance')
            plt.legend()
            plt.grid()
            plt.title(f"OLS bias variance tradeoff")
            plt.xlabel("Polynomial degree")
            plt.xticks(self._polynomials)


    def _CV(self, degree):
        l = int((degree+1)*(degree+2)/2)

        X_train, X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

        i = 0
        
        for train_inds, test_inds in self._kfold.split(self._x):
            x_train = self._x[train_inds]
            y_train = self._y[train_inds]
            z_train = self._z[train_inds]

            x_test = self._x[test_inds]
            y_test = self._y[test_inds]
            z_test = self._z[test_inds]
        
            # Training and test design matrix
            X_train = self._design_matrix(x_train, y_train, degree, ret=True)
            X_test = self._design_matrix(x_test, y_test, degree, ret=True)

            #Fitting the model
            beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

            z_pred = X_test @ beta
                
            z_pred = z_pred.reshape(-1,1)
            j = degree - (self._maxdegree - self._scores_KFold.shape[0])
            self._scores_KFold[j, i] = np.sum((z_pred - z_test[:, np.newaxis])**2)/np.size(z_pred)
        
            i += 1

    def calc_CV(self):
        estimated_mse_KFold = np.mean(self._scores_KFold, axis = 1)
        self._mse_test = estimated_mse_KFold

    def plot_CV(self):
        if len(self._polynomials)==1:
            return self._mse_test
        
        else:
            #PLotting the MSE
            plt.plot(self._polynomials, self._mse_test, label='Estimated MSE OLS')
            plt.legend()
            plt.grid()
            plt.title("OLS cross validation")
            plt.xlabel("Polynomial degree")
            plt.ylabel("MSE")
            plt.xticks(self._polynomials)


    def get_mse_test(self):
        return self._mse_test



class Ridge():
    def __init__(self, scaling="mean"):
        self._scaling = scaling

        self._mse_train = []
        self._mse_test = []
        self._r2_train = []
        self._r2_test = []
        self._beta_list = []
        self._polynomials = []

        self._bias = []
        self._variance = []
        self._error = []

        self._called = False

    def fit(self, x, y, z, maxdegree, lmbds, test_size=0.2):
        self._maxdegree = maxdegree
        self._x = x
        self._y = y
        self._z = z
        self._lmbds = lmbds

        self._design_matrix(self._x, self._y, maxdegree)

        self._test_size = test_size
        self._X_train_main, self._X_test_main, self._x_train, self._x_test, self._y_train, self._y_test, self._z_train, self._z_test = train_test_split(self._X_main, self._x, self._y, self._z, test_size=self._test_size,random_state=10)

    def _design_matrix(self, x, y, degree, ret=False):
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree+1)*(degree+2)/2)

        X = np.ones((N,l-1))

        for i in range(1,degree+1):
            q = int((i)*(i+1)/2) - 1
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

        if ret:
            return X
        else:
            self._X_main = X

    def predict(self, degree, bootstrap=False, n_bootstraps=100, CV=False, k=10):
        if bootstrap:
            if not self._called:
                self._z_pred = np.zeros((len(self._z_test),n_bootstraps))
                
            self._called = True
            self._bootstrap(degree, lmbd=self._lmbds[0], n_bootstraps=n_bootstraps)

        elif CV:
            if not self._called:
                self._scores_KFold = np.zeros((len(self._lmbds), self._maxdegree-degree, k))
                self._kfold = KFold(n_splits=k, random_state=1, shuffle=True)

                self._called = True
        
            for i in range(len(self._lmbds)):
                lmbd = 10**float(self._lmbds[i])
                self._CV(degree, lmbd, i)

            self._polynomials.append(degree)
            
        else:
            beta_list = []
            mse_train_list = []
            mse_test_list = []
            r2_train_list = []
            r2_test_list = []

            # without indercept
            l = int((degree+1)*(degree+2)/2) - 1

            self._X_train, self._X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

            for i in range(len(self._lmbds)):
                lmbd = 10**float(self._lmbds[i])

                if not self._scaling:
                    # Calculating the beta with the training sets
                    self._beta = (np.linalg.pinv(self._X_train.T @ self._X_train + lmbd * np.identity(self._X_train.shape[1])) @ self._X_train.T) @ self._z_train

                    self._z_tilde = self._X_train @ self._beta
                    self._z_predict = self._X_test @ self._beta
                
                elif self._scaling=="mean":
                    # Scaling
                    X_train_mean = np.mean(self._X_train, axis=0)

                    X_train_scaled = self._X_train - X_train_mean
                    X_test_scaled = self._X_test - X_train_mean

                    z_scaler = np.mean(self._z_train)
                    z_train_scaled = self._z_train - z_scaler

                    # Calculating the beta with the training sets
                    self._beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled + lmbd * np.identity(X_train_scaled.shape[1])) @ X_train_scaled.T) @ z_train_scaled

                    # Calculating z_tilde and z_predict
                    self._z_tilde = X_train_scaled @ self._beta + z_scaler
                    self._z_predict = X_test_scaled @ self._beta + z_scaler

                elif self._scaling=="minmax":
                    X_train_max = np.max(self._X_train)
                    X_train_min = np.min(self._X_train)
                    X_train_scaled = (self._X_train-X_train_min)/(X_train_max-X_train_min)
                    X_test_scaled = (self._X_test-X_train_min)/(X_train_max-X_train_min)

                    z_train_max = np.max(self._z_train)
                    z_train_min = np.min(self._z_train)
                    z_train_scaled = (self._z_train-z_train_min)/(z_train_max-z_train_min)

                    # Calculating the beta with the training sets
                    self._beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled + lmbd * np.identity(X_train_scaled.shape[1])) @ X_train_scaled.T) @ z_train_scaled

                    # Calculating z_tilde and z_predict
                    self._z_tilde = X_train_scaled @ self._beta
                    self._z_predict = X_test_scaled @ self._beta

                beta_list.append(self._beta)

                mse_train, mse_test = self.mse()
                mse_train_list.append(mse_train)
                mse_test_list.append(mse_test)

                r2_train, r2_test = self.r2()
                r2_train_list.append(r2_train)
                r2_test_list.append(r2_test)
            

            self._mse_train.append(mse_train_list)
            self._mse_test.append(mse_test_list)

            self._r2_train.append(r2_train_list)
            self._r2_test.append(r2_test_list)

            self._beta_list.append(beta_list)
            self._polynomials.append(degree)


    def mse(self):
        mse_train = mean_squared_error(self._z_train, self._z_tilde)
        mse_test = mean_squared_error(self._z_test, self._z_predict)

        return mse_train, mse_test
    
    def r2(self):
        r2_train = r2_score(self._z_train, self._z_tilde)
        r2_test = r2_score(self._z_test, self._z_predict)

        return r2_train, r2_test

    def heatmap_mse(self, data_set="test"):
        # Heatmap with x-axis as lambda and y-axis as polynomial degree
        cmap = "RdYlGn_r"

        if data_set=="train":
            data = np.asarray(self._mse_train)

        elif data_set=="test":
            data = np.asarray(self._mse_test)

        else:
            raise ValueError("data_set must be either train or test.")

        im = sns.heatmap(data, xticklabels=self._lmbds, yticklabels=self._polynomials, cmap=cmap, 
                        cbar_kws={"orientation": "vertical", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05}, 
                        annot=True, annot_kws={"fontsize":9}, linewidth=0.5)

        best_idx = np.unravel_index(np.argmin(data, axis=None), data.shape)
        im.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fc="none", ec="blue", lw=5, clip_on=False))

        plt.title(f"Ridge regression MSE for {data_set} set")
        im.set_ylabel("Polynomial degree", labelpad=10)
        im.set_xlabel("$\lambda}$ ($log_{10}$)", labelpad=10)
        plt.tight_layout()

        print(f"Best MSE for Ridge regression is {data[best_idx[0],best_idx[1]]}")

    def plot_beta(self, degree):

        beta_plot_min = []
        beta_plot_max = []

        i = 0
        for p in self._polynomials:
            if p==degree:
                break
            i += 1

        for beta in self._beta_list[i]:
            if len(beta) < 2:
                beta_min = beta[0]
                beta_max = beta[0]

            else:
                beta_min = min(beta)
                beta_max = max(beta)

            beta_plot_min.append(beta_min)
            beta_plot_max.append(beta_max)

        
        plt.plot(self._lmbds, beta_plot_min)
        plt.plot(self._lmbds, beta_plot_max)
        plt.title(f"Polynomial degree {degree}")
        plt.xlabel("$\lambda$ ($log_{10}$)")
        plt.ylabel("Beta")
        plt.grid()

    def _bootstrap(self, degree, lmbd, n_bootstraps):
        l = int((degree+1)*(degree+2)/2)-1

        X_test = self._X_test_main[:,:l]
        z_test = self._z_test.reshape(-1,1)

        # X_test = self._design_matrix(self._x_test, self._y_test, degree, ret=True)

        for i in range(n_bootstraps):
            # Resampling
            x_train_boot, y_train_boot, z_train_boot = resample(self._x_train, self._y_train, self._z_train, random_state=i)

            # Training design matrix
            X_train = self._design_matrix(x_train_boot, y_train_boot, degree, ret=True)

            if not self._scaling:
                # Calculating the beta with the training sets
                beta = (np.linalg.pinv(X_train.T @ X_train + 10**(lmbd) * np.identity(X_train.shape[1])) @ X_train.T) @ z_train_boot

                z_pred = X_test @ beta
            
            elif self._scaling=="mean":
                # Scaling
                X_train_mean = np.mean(X_train, axis=0)

                X_train_scaled = X_train - X_train_mean
                X_test_scaled = X_test - X_train_mean

                z_scaler = np.mean(z_train_boot)
                z_train_scaled = z_train_boot - z_scaler

                # Calculating the beta with the training sets
                beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled + 10**(lmbd) * np.identity(X_train_scaled.shape[1])) @ X_train_scaled.T) @ z_train_scaled

                # Calculating z_tilde and z_predict
                z_pred = X_test_scaled @ beta + z_scaler

            elif self._scaling=="minmax":
                X_train_max = np.max(X_train)
                X_train_min = np.min(X_train)
                X_train_scaled = (X_train-X_train_min)/(X_train_max-X_train_min)
                X_test_scaled = (X_test-X_train_min)/(X_train_max-X_train_min)

                z_train_max = np.max(z_train_boot)
                z_train_min = np.min(z_train_boot)
                z_train_scaled = (z_train_boot-z_train_min)/(z_train_max-z_train_min)

                # Calculating the beta with the training sets
                beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled + 10**(lmbd) * np.identity(X_train_scaled.shape[1])) @ X_train_scaled.T) @ z_train_scaled

                z_pred = X_test_scaled @ beta

            # Calculate predicted value
            self._z_pred[:,i] = z_pred

        self._polynomials.append(degree)

        bias = np.mean((z_test - np.mean(self._z_pred, axis=1,keepdims = True))**2)
        variance = np.mean(np.var(self._z_pred, axis=1, keepdims=True))
        error = np.mean(np.mean((z_test - self._z_pred)**2, axis=1, keepdims=True))

        self._bias.append(bias)
        self._variance.append(variance)
        self._error.append(error)


    def plot_bootstrap(self):
        plt.plot(self._polynomials, self._error, label='Error')
        plt.plot(self._polynomials, self._bias, label='bias')
        plt.plot(self._polynomials, self._variance, label='Variance')
        plt.legend()
        plt.grid()
        plt.title(f"Ridge bias variance tradeof, $\lambda(log10)$={self._lmbds[0]}")
        plt.xlabel("Polynomial degree")
        plt.xticks(self._polynomials)

    def _CV(self, degree, lmbd, k):
        l = int((degree+1)*(degree+2)/2)-1

        X_train, X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

        i = 0
        
        for train_inds, test_inds in self._kfold.split(self._x):
            x_train = self._x[train_inds]
            y_train = self._y[train_inds]
            z_train = self._z[train_inds]

            x_test = self._x[test_inds]
            y_test = self._y[test_inds]
            z_test = self._z[test_inds]
        
            # Training and test design matrix
            X_train = self._design_matrix(x_train, y_train, degree, ret=True)
            X_test = self._design_matrix(x_test, y_test, degree, ret=True)

            if not self._scaling:
                # Calculating the beta with the training sets
                beta = (np.linalg.pinv(X_train.T @ X_train + lmbd * np.identity(X_train.shape[1])) @ X_train.T) @ z_train

                z_pred = X_test @ beta
            
            elif self._scaling=="mean":
                # Scaling
                X_train_mean = np.mean(X_train, axis=0)

                X_train_scaled = X_train - X_train_mean
                X_test_scaled = X_test - X_train_mean

                z_scaler = np.mean(z_train)
                z_train_scaled = z_train - z_scaler

                # Calculating the beta with the training sets
                beta = (np.linalg.pinv(X_train_scaled.T @ X_train_scaled + lmbd * np.identity(X_train_scaled.shape[1])) @ X_train_scaled.T) @ z_train_scaled

                # Calculating z_tilde and z_predict
                z_pred = X_test_scaled @ beta + z_scaler
                
            z_pred = z_pred.reshape(-1,1)
            j = degree - (self._maxdegree - self._scores_KFold.shape[1])
            self._scores_KFold[k, j, i] = np.sum((z_pred - z_test[:, np.newaxis])**2)/np.size(z_pred)
        
            i += 1

    def plot_CV(self):

        estimated_mse_KFold = np.mean(self._scores_KFold, axis = 1)

        #PLotting the MSE
        plt.plot(self._polynomials, estimated_mse_KFold, label='Estimated MSE Ridge')
        plt.legend()
        plt.grid()
        plt.title("Ridge cross validation")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.xticks(self._polynomials)

    def get_mse_test(self):
        return self._mse_test
    
    def get_mse_train(self):
        return self._mse_train
    
    def heatmap_CV(self):
        N = self._scores_KFold.shape[0]
        data = []
        for i in range(N):
            data.append(np.mean(self._scores_KFold[i,:,:], axis = 1))

        data = np.asarray(data).T
        
        # Heatmap with x-axis as lambda and y-axis as polynomial degree
        cmap = "RdYlGn_r"

        im = sns.heatmap(data, xticklabels=self._lmbds, yticklabels=self._polynomials, cmap=cmap, 
                        cbar_kws={"orientation": "vertical", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05}, 
                        annot=True, annot_kws={"fontsize":9}, linewidth=0.5)

        best_idx = np.unravel_index(np.argmin(data, axis=None), data.shape)
        im.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fc="none", ec="blue", lw=5, clip_on=False))

        plt.title(f"Ridge regression CV for test set")
        im.set_ylabel("Polynomial degree", labelpad=10)
        im.set_xlabel("$\lambda}$ ($log_{10}$)", labelpad=10)
        plt.tight_layout()


class Lasso():
    def __init__(self, scaling=None, max_iter=1000, tol=0.0001, fit_intercept=True):
        self._scaling = scaling
        self._max_iter = max_iter
        self._tol = tol
        self._fit_intercept = fit_intercept

        self._mse_train = []
        self._mse_test = []
        self._r2_train = []
        self._r2_test = []
        self._beta_list = []
        self._polynomials = []

        self._bias = []
        self._variance = []
        self._error = []

        self._called = False

    def fit(self, x, y, z, maxdegree, lmbds, test_size=0.2):
        self._maxdegree = maxdegree
        self._x = x
        self._y = y
        self._z = z
        self._lmbds = lmbds

        self._design_matrix(self._x, self._y, maxdegree)

        self._test_size = test_size
        self._X_train_main, self._X_test_main, self._x_train, self._x_test, self._y_train, self._y_test, self._z_train, self._z_test = train_test_split(self._X_main, self._x, self._y, self._z, test_size=self._test_size,random_state=10)

        

    def _design_matrix(self, x, y, degree, ret=False):
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((degree+1)*(degree+2)/2)

        X = np.ones((N,l-1))

        for i in range(1,degree+1):
            q = int((i)*(i+1)/2) - 1
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

        if ret:
            return X 
        else:
            self._X_main = X

    @ignore_warnings(category=ConvergenceWarning)
    def predict(self, degree, bootstrap=False, n_bootstraps=100, CV=False, k=10):
        if bootstrap:
            if not self._called:
                self._z_pred = np.zeros((len(self._z_test),n_bootstraps))
            self._called = True
            self._bootstrap(degree, lmbd=self._lmbds[0], n_bootstraps=n_bootstraps)
            

        elif CV:
            if not self._called:
                self._scores_KFold = np.zeros((len(self._lmbds), self._maxdegree-degree, k))
                print(self._scores_KFold.shape)
                self._kfold = KFold(n_splits=k, random_state=1, shuffle=True)

                self._called = True
        
            for i in range(len(self._lmbds)):
                lmbd = 10**float(self._lmbds[i])
                self._CV(degree, lmbd, i)

            self._polynomials.append(degree)
            
        else:
            beta_list = []
            mse_train_list = []
            mse_test_list = []
            r2_train_list = []
            r2_test_list = []

            # without indercept
            l = int((degree+1)*(degree+2)/2) - 1

            self._X_train, self._X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

            if self._scaling=="mean":
                scaler = StandardScaler(with_std=False)
                self._X_train = scaler.fit_transform(self._X_train)
                self._X_test = scaler.fit_transform(self._X_test)
           

            for i in range(len(self._lmbds)):
                lmbd = 10**float(self._lmbds[i])

                RegLasso = linear_model.Lasso(alpha=lmbd, fit_intercept=self._fit_intercept, max_iter=self._max_iter, tol=self._tol)
                
                RegLasso.fit(self._X_train, self._z_train)

                self._z_tilde = RegLasso.predict(self._X_train)
                self._z_predict = RegLasso.predict(self._X_test)
                self._beta = RegLasso.coef_

                beta_list.append(self._beta)

                mse_train, mse_test = self.mse()
                mse_train_list.append(mse_train)
                mse_test_list.append(mse_test)

                r2_train, r2_test = self.r2()
                r2_train_list.append(r2_train)
                r2_test_list.append(r2_test)
            
            

            self._mse_train.append(mse_train_list)
            self._mse_test.append(mse_test_list)

            self._r2_train.append(r2_train_list)
            self._r2_test.append(r2_test_list)

            self._beta_list.append(beta_list)
            self._polynomials.append(degree)


    def mse(self):
        mse_train = mean_squared_error(self._z_train, self._z_tilde)
        mse_test = mean_squared_error(self._z_test, self._z_predict)

        return mse_train, mse_test
    
    def r2(self):
        r2_train = r2_score(self._z_train, self._z_tilde)
        r2_test = r2_score(self._z_test, self._z_predict)

        return r2_train, r2_test

    def heatmap_mse(self, data_set="test"):
        # Heatmap with x-axis as lambda and y-axis as polynomial degree
        cmap = "RdYlGn_r"

        if data_set=="train":
            data = np.asarray(self._mse_train)

        elif data_set=="test":
            data = np.asarray(self._mse_test)

        else:
            raise ValueError("data_set must be either train or test.")

        im = sns.heatmap(data, xticklabels=self._lmbds, yticklabels=self._polynomials, cmap=cmap, 
                        cbar_kws={"orientation": "vertical", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05}, 
                        annot=True, annot_kws={"fontsize":9}, linewidth=0.5)

        best_idx = np.unravel_index(np.argmin(data, axis=None), data.shape)
        im.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fc="none", ec="blue", lw=5, clip_on=False))

        plt.title(f"Lasso regression MSE for {data_set} set")
        im.set_ylabel("Polynomial degree", labelpad=10)
        im.set_xlabel("$\lambda}$ ($log_{10}$)", labelpad=10)
        plt.tight_layout()

        print(f"Best MSE for Lasso regression is {data[best_idx[0],best_idx[1]]}")


    def plot_beta(self, degree):

        beta_plot_min = []
        beta_plot_max = []

        i = 0
        for p in self._polynomials:
            if p==degree:
                break
            i += 1

        for beta in self._beta_list[i]:
            if len(beta) < 2:
                beta_min = beta[0]
                beta_max = beta[0]

            else:
                beta_min = min(beta)
                beta_max = max(beta)

            beta_plot_min.append(beta_min)
            beta_plot_max.append(beta_max)

        
        plt.plot(self._lmbds, beta_plot_min)
        plt.plot(self._lmbds, beta_plot_max)
        plt.title(f"Polynomial degree {degree}")
        plt.xlabel("$\lambda$ ($log_{10}$)")
        plt.ylabel("Beta")
        plt.grid()

    @ignore_warnings(category=ConvergenceWarning)
    def _bootstrap(self, degree, lmbd, n_bootstraps):
        self._n_bootstraps = n_bootstraps

        l = int((degree+1)*(degree+2)/2)-1

        X_test = self._X_test_main[:,:l]
        z_test = self._z_test.reshape(-1,1)

        

        for i in range(n_bootstraps):
            # Resampling
            x_train_boot, y_train_boot, z_train_boot = resample(self._x_train, self._y_train, self._z_train)

            # Training design matrix
            X_train = self._design_matrix(x_train_boot, y_train_boot, degree, ret=True)


            RegLasso = linear_model.Lasso(alpha=10**(lmbd), fit_intercept=self._fit_intercept, max_iter=self._max_iter, tol=self._tol)
            RegLasso.fit(X_train, z_train_boot)

            z_pred = RegLasso.predict(X_test)

            # Calculate predicted value
            self._z_pred[:,i] = z_pred

        self._polynomials.append(degree)

        bias = np.mean((z_test - np.mean(self._z_pred, axis=1,keepdims = True))**2)
        variance = np.mean(np.var(self._z_pred, axis=1, keepdims=True))
        error = np.mean(np.mean((z_test - self._z_pred)**2, axis=1, keepdims=True))

        self._bias.append(bias)
        self._variance.append(variance)
        self._error.append(error)


    def plot_bootstrap(self):
        print(len(self._polynomials))
        if len(self._polynomials)==1:
            plt.plot(self._n_bootstraps, self._error, "r.",label='Error')
            plt.plot(self._n_bootstraps, self._bias, "b.", label='bias')
            plt.plot(self._n_bootstraps, self._variance, "g.", label='Variance')

        else:
            plt.plot(self._polynomials, self._error, label='Error')
            plt.plot(self._polynomials, self._bias, label='bias')
            plt.plot(self._polynomials, self._variance, label='Variance')
            plt.legend()
            plt.grid()
            plt.title(f"Lasso bias variance tradeof, $\lambda(log10)$={self._lmbds[0]}")
            plt.xlabel("Polynomial degree")
            plt.xticks(self._polynomials)

    @ignore_warnings(category=ConvergenceWarning)
    def _CV(self, degree, lmbd, k):
        l = int((degree+1)*(degree+2)/2)-1

        X_train, X_test = self._X_train_main[:,:l], self._X_test_main[:,:l]

        i = 0
        
        RegLasso = linear_model.Lasso(alpha=lmbd, fit_intercept=self._fit_intercept, max_iter=self._max_iter, tol=self._tol, copy_X=True)

        for train_inds, test_inds in self._kfold.split(self._x):
            x_train = self._x[train_inds]
            y_train = self._y[train_inds]
            z_train = self._z[train_inds]

            x_test = self._x[test_inds]
            y_test = self._y[test_inds]
            z_test = self._z[test_inds]
        
            # Training and test design matrix
            X_train = self._design_matrix(x_train, y_train, degree, ret=True)
            X_test = self._design_matrix(x_test, y_test, degree, ret=True)
            
            if not self._scaling:
                z_scaler = 0

            if self._scaling=="mean":
                # Scaling
                X_train_mean = np.mean(X_train, axis=0)

                X_train = X_train - X_train_mean
                X_test = X_test - X_train_mean

                z_scaler = np.mean(z_train)
                z_train = z_train - z_scaler

            
            RegLasso.fit(X_train, z_train)

            z_pred = RegLasso.predict(X_test) + z_scaler

                
            z_pred = z_pred.reshape(-1,1)
            j = degree - (self._maxdegree - self._scores_KFold.shape[1])
            self._scores_KFold[k, j, i] = np.sum((z_pred - z_test[:, np.newaxis])**2)/np.size(z_pred)
            i += 1

    def plot_CV(self):

        estimated_mse_KFold = np.mean(self._scores_KFold[0,:,:], axis = 1)

        #PLotting the MSE
        plt.plot(self._polynomials, estimated_mse_KFold, label='Estimated MSE Lasso')
        plt.legend()
        plt.grid()
        plt.title("Lasso cross validation")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.xticks(self._polynomials)

    def get_mse_test(self):
        return self._mse_test
    
    def get_mse_train(self):
        return self._mse_train
    
    def heatmap_CV(self):
        N = self._scores_KFold.shape[0]
        data = []
        for i in range(N):
            data.append(np.mean(self._scores_KFold[i,:,:], axis = 1))

        data = np.asarray(data).T
        
        # Heatmap with x-axis as lambda and y-axis as polynomial degree
        cmap = "RdYlGn_r"

        im = sns.heatmap(data, xticklabels=self._lmbds, yticklabels=self._polynomials, cmap=cmap, 
                        cbar_kws={"orientation": "vertical", "shrink":0.8, "aspect":40, "label": "MSE", "pad":0.05}, 
                        annot=True, annot_kws={"fontsize":9}, linewidth=0.5)

        best_idx = np.unravel_index(np.argmin(data, axis=None), data.shape)
        im.add_patch(plt.Rectangle((best_idx[1], best_idx[0]), 1, 1, fc="none", ec="blue", lw=5, clip_on=False))

        plt.title(f"Lasso regression CV for test set")
        im.set_ylabel("Polynomial degree", labelpad=10)
        im.set_xlabel("$\lambda}$ ($log_{10}$)", labelpad=10)
        plt.tight_layout()