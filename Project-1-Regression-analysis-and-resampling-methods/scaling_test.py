import regression_methods as rm
from func import *

np.random.seed(2023)
x, y, z = generate_data(step_size=0.05)

# OLS
m1 = rm.OLS()
m2 = rm.OLS(scaling="mean")

maxdegree = 21

m1.fit(x,y,z,maxdegree)
m2.fit(x,y,z,maxdegree)

for degree in range(1,maxdegree):
    m1.predict(degree)
    m2.predict(degree)

m1.plot_mse()
m2.plot_mse()

plt.legend()
plt.title("Comparing no scaling and mean scaling for OLS")
plt.grid()
plt.savefig("plots/OLS_scaling_comparison.svg")
plt.clf()


# Ridge
m1 = rm.Ridge(scaling=None)
m2 = rm.Ridge(scaling="mean")

maxdegree = 21
lmbds = np.arange(-12,4,1)

avg_list_m1 = []
avg_list_m2 = []
min_list_m1 = []
max_list_m1 = []
min_list_m2 = []
max_list_m2 = []


for lmbd in lmbds:
    m1 = rm.Ridge(scaling=None)
    m2 = rm.Ridge(scaling="mean")

    m1.fit(x,y,z,maxdegree,lmbds=[lmbd])
    m2.fit(x,y,z,maxdegree,lmbds=[lmbd])

    for degree in range(1,maxdegree):
        m1.predict(degree)
        m2.predict(degree)

    mse = m1.get_mse_test()
    avg_mse = np.mean(mse)
    avg_list_m1.append(avg_mse)
    min_list_m1.append(np.min(mse))
    max_list_m1.append(np.max(mse))

    mse = m2.get_mse_test()
    avg_mse = np.mean(mse)
    avg_list_m2.append(avg_mse)
    min_list_m2.append(np.min(mse))
    max_list_m2.append(np.max(mse))


plt.plot(lmbds, avg_list_m1, "b",label="Avg. Ridge no scaling")
plt.plot(lmbds, avg_list_m2, "r",label="Avg. Ridge mean scaling")

plt.fill_between(lmbds, min_list_m1, max_list_m1, color="b", alpha=0.1)
plt.fill_between(lmbds, min_list_m2, max_list_m2, color="r", alpha=0.1)

plt.legend()
plt.title("Comparing no scaling and mean scaling for Ridge")
plt.grid()
plt.xlabel("$\lambda$(log10)")
plt.ylabel("MSE")

plt.savefig("plots/Ridge_scaling_comparison.svg")
plt.clf()

# Ridge
m1 = rm.Lasso(scaling=None)
m2 = rm.Lasso(scaling="mean")

maxdegree = 21
lmbds = np.arange(-12,4,1)

avg_list_m1 = []
avg_list_m2 = []
min_list_m1 = []
max_list_m1 = []
min_list_m2 = []
max_list_m2 = []


for lmbd in lmbds:
    m1 = rm.Lasso(scaling=None)
    m2 = rm.Lasso(scaling="mean")

    m1.fit(x,y,z,maxdegree,lmbds=[lmbd])
    m2.fit(x,y,z,maxdegree,lmbds=[lmbd])

    for degree in range(1,maxdegree):
        m1.predict(degree)
        m2.predict(degree)

    mse = m1.get_mse_test()
    avg_mse = np.mean(mse)
    avg_list_m1.append(avg_mse)
    min_list_m1.append(np.min(mse))
    max_list_m1.append(np.max(mse))

    mse = m2.get_mse_test()
    avg_mse = np.mean(mse)
    avg_list_m2.append(avg_mse)
    min_list_m2.append(np.min(mse))
    max_list_m2.append(np.max(mse))


plt.plot(lmbds, avg_list_m1, "b",label="Avg. LASSO no scaling")
plt.plot(lmbds, avg_list_m2, "r",label="Avg. LASSO mean scaling")

plt.fill_between(lmbds, min_list_m1, max_list_m1, color="b", alpha=0.1)
plt.fill_between(lmbds, min_list_m2, max_list_m2, color="r", alpha=0.1)

plt.legend()
plt.title("Comparing no scaling and mean scaling for LASSO")
plt.grid()
plt.xlabel("$\lambda$(log10)")
plt.ylabel("MSE")

plt.savefig("plots/LASSO_scaling_comparison.svg")
plt.clf()