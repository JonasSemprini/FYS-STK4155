import regression_methods as rm
from func import *

# Setting data and variables
np.random.seed(2023)

# Creating detailed 3D plots of the franke function
x, y, z = generate_data(noise=False, step_size=0.005)

plot_surf(x, y, z, save=True, out_name="FrankeFunction_smooth.svg")


print("_____________Franke's function__________________")
print("________________________________________________")

x, y, z = generate_data()

plot_surf(x, y, z, save=True, out_name="FrankeFunction_noise.svg")

maxdegree = 16 # So up to polynomial degree 15
lmbds = range(-8,5) # lmbds given in log10 from -8 up to, but not including 5

# Figure of mse and r2, using OLS of Franke function
m1 = rm.OLS()
m1.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m1.predict(degree)

plt.subplot(2,1,1)
plt.title("MSE for OLS")
m1.plot_mse()
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(2,1,2)
plt.title("R2-score for OLS")
m1.plot_r2()
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("plots/fig1.svg")
plt.clf()

plt.figure(figsize=(15,9))
m1.plot_beta()
plt.title("Max and min beta coefficients using OLS")
plt.grid()
plt.savefig("plots/fig6.svg")
plt.clf()

# Ridge MSE
m2 = rm.Ridge()
m2.fit(x,y,z,maxdegree,lmbds)

for degree in range(1,maxdegree):
    m2.predict(degree)

m2.heatmap_mse()
plt.savefig("plots/fig2.svg")
plt.clf()

# Lasso MSE
m3 = rm.Lasso()
m3.fit(x,y,z,maxdegree,lmbds)

for degree in range(1,maxdegree):
    m3.predict(degree)

m3.heatmap_mse()
plt.savefig("plots/fig3.svg")
plt.clf()

# Lasso MSE comparison
m4 = rm.Lasso(max_iter=10000)
m4.fit(x,y,z,maxdegree,lmbds)

m5 = rm.Lasso(max_iter=100)
m5.fit(x,y,z,maxdegree,lmbds)

for degree in range(1,maxdegree):
    m4.predict(degree)
    m5.predict(degree)

m4.heatmap_mse()
plt.savefig("plots/fig4.svg")
plt.clf()

m5.heatmap_mse()
plt.savefig("plots/fig5.svg")
plt.clf()


# OLS bias variance tradeof
m6 = rm.OLS()
m6.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m6.predict(degree, bootstrap=True)

m6.plot_bootstrap()

mse_list = m6.get_mse_test()
min_mse_bias = np.min(mse_list)

for mse, degree in zip(mse_list, m6._polynomials):
    if mse==min_mse_bias:
        min_degree_boot = degree

print(f"With bootstrap we achieve a minimum MSE of {min_mse_bias:.6f} at polynomial degree {min_degree_boot}.")

plt.savefig("plots/fig7.svg")
plt.clf()

# OLS CV
m7 = rm.OLS()
m7.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m7.predict(degree, CV=True)

m7.calc_CV()
m7.plot_CV()


mse_list = m7.get_mse_test()
min_mse_CV = np.min(mse_list)

for mse, degree in zip(mse_list, m7._polynomials):
    if mse==min_mse_CV:
        min_degree_CV = degree

print(f"With CV we achieve a minimum MSE of {min_mse_CV:.6f} at polynomial degree {min_degree_CV}.")

plt.savefig("plots/fig8.svg")
plt.clf()


# OLS bias variance tradeof
error_list = []

degree = min_degree_boot #Optimal polynomial degree from bootstrap
bootstrap_array = np.arange(5,150,5)

for n_bootstraps in bootstrap_array:
    m8 = rm.OLS()
    m8.fit(x,y,z,degree)

    m8.predict(degree, bootstrap=True, n_bootstraps=n_bootstraps)

    error, _, _ = m8.plot_bootstrap()

    error_list.append(error)

min_error = np.min(error_list)

for error, n_boot in zip(error_list, bootstrap_array):
    if error==min_error:
        min_boot = n_boot

print(f"Converging to optimal MSE of {min_error} at {min_boot} bootstrap samples.")

plt.plot(bootstrap_array, error_list, label="Error")
plt.xlabel("Number of bootstraps")
plt.ylabel("MSE")
plt.title(f"Bootstrap MSE over polynomial degree {degree}")
plt.grid()
plt.legend()

plt.savefig("plots/fig9.svg",)
plt.clf()

# OLS Cross validation
CV_list = []

degree = min_degree_CV+1 #Optimal polynomial degree from CV
CV_array = np.array([3,7,10])

for k in CV_array:
    m9 = rm.OLS()
    m9.fit(x,y,z,degree)
    
    for d in range(1,degree):
        m9.predict(d, CV=True, k=k)

    m9.calc_CV()
    mse = m9.get_mse_test()

    CV_list.append(np.min(mse))

min_mse = np.min(CV_list)

for mse, k in zip(CV_list, CV_array):
    if mse==min_mse:
        min_k = k

print(f"Optimal MSE of {min_mse} at {min_k} folds.")

plt.plot(CV_array, CV_list, label="CV MSE")
plt.xlabel("Number of k folds")
plt.ylabel("MSE")
plt.title(f"CV for polynomial degree {degree-1}")
plt.grid()
plt.legend()

plt.savefig("plots/fig10.svg")
plt.clf()

print("_________________Terrain data___________________")
print("________________________________________________")

# Load the terrain
terrain = imread("data/SRTM_data_Norway_1.tif")

x, y, z = generate_dataReal(terrain, N=85)
maxdegree = 26 # So up to polynomial degree 25

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap=cm.coolwarm)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("plots/terrain_data_2D.svg")
plt.clf()

# Figure of mse and r2, using OLS of terrain data
m10 = rm.OLS()
m10.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree): 
    m10.predict(degree)

plt.subplot(2,1,1)
plt.title("MSE for OLS")
m10.plot_mse()
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(2,1,2)
plt.title("R2-score for OLS")
m10.plot_r2()
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("plots/fig11.svg")
plt.clf()

plt.figure(figsize=(15,9))
m10.plot_beta()
plt.title("Max and min beta coefficients using OLS")
plt.grid()
plt.savefig("plots/fig12.svg")
plt.clf()

# Ridge MSE
m11 = rm.Ridge()
m11.fit(x,y,z,maxdegree,lmbds)

for degree in range(1,maxdegree):
    m11.predict(degree)

m11.heatmap_mse()
plt.savefig("plots/fig13.svg")
plt.clf()

# Lasso MSE
m12 = rm.Lasso()
m12.fit(x,y,z,maxdegree,lmbds)

for degree in range(1,maxdegree):
    m12.predict(degree)

m12.heatmap_mse()
plt.savefig("plots/fig14.svg")
plt.clf()


# OLS bias variance tradeof
m15 = rm.OLS()
m15.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m15.predict(degree, bootstrap=True)

m15.plot_bootstrap()

mse_list = m15.get_mse_test()
min_mse_bias = np.min(mse_list)

for mse, degree in zip(mse_list, m15._polynomials):
    if mse==min_mse_bias:
        min_degree_boot = degree

print(f"With bootstrap we achieve a minimum MSE of {min_mse_bias:.6f} at polynomial degree {min_degree_boot}.")

plt.savefig("plots/fig17.svg")
plt.clf()

# OLS CV
m16 = rm.OLS()
m16.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m16.predict(degree, CV=True)

m16.calc_CV()
m16.plot_CV()


mse_list = m16.get_mse_test()
min_mse_CV = np.min(mse_list)

for mse, degree in zip(mse_list, m16._polynomials):
    if mse==min_mse_CV:
        min_degree_CV = degree

print(f"With CV we achieve a minimum MSE of {min_mse_CV:.6f} at polynomial degree {min_degree_CV}.")

plt.savefig("plots/fig18.svg")
plt.clf()


# OLS bias variance tradeof
error_list = []

degree = min_degree_boot #Optimal polynomial degree from bootstrap
bootstrap_array = np.arange(5,155,5)

for n_bootstraps in bootstrap_array:
    m17 = rm.OLS()
    m17.fit(x,y,z,degree)

    m17.predict(degree, bootstrap=True, n_bootstraps=n_bootstraps)

    error, _, _ = m17.plot_bootstrap()

    error_list.append(error)

min_error = np.min(error_list)

for error, n_boot in zip(error_list, bootstrap_array):
    if error==min_error:
        min_boot = n_boot

print(f"Converging to optimal MSE of {min_error} at {min_boot} bootstrap samples.")

plt.plot(bootstrap_array, error_list, label="MSE of OLS with bootstrap.")
plt.xlabel("Number of bootstraps")
plt.ylabel("MSE")
plt.title(f"Bootstrap MSE over polynomial degree {degree}")
plt.grid()
plt.legend()

plt.savefig("plots/fig19.svg",)
plt.clf()



# OLS Cross validation
CV_list = []

degree = min_degree_CV+1 #Optimal polynomial degree from CV
CV_array = np.array([3,7,10])

for k in CV_array:
    m18 = rm.OLS()
    m18.fit(x,y,z,degree)
    
    for d in range(1,degree):
        m18.predict(d, CV=True, k=k)

    m18.calc_CV()
    mse = m18.get_mse_test()

    CV_list.append(np.min(mse))

min_mse = np.min(CV_list)

for mse, k in zip(CV_list, CV_array):
    if mse==min_mse:
        min_k = k

print(f"Optimal MSE of {min_mse} at {min_k} folds.")

plt.plot(CV_array, CV_list, label="CV MSE")
plt.xlabel("Number of k folds")
plt.ylabel("MSE")
plt.title(f"CV for polynomial degree {degree-1}")
plt.grid()
plt.legend()

plt.savefig("plots/fig20.svg")
plt.clf()
