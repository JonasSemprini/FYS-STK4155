
# README for Project 1:
**Project 1:** Regression analysis and resampling in FYS-STK4155 at UiO.

**Authors:** *Rebecca Nguyen, Federico Santona, Jonas Semprini and Mathias Svoren*


### Regression analysis and resampling methods
Following .py files are included in project:
- main.py
- func.py
- regression_methods.py

For testing:
- scaling_test.py

Terrain data in directory:
- /data

Plots in directory:
- /plots


### Output from main.py
```console
(base) ➜  project1 git:(master) ✗ python main.py
_____________Franke's function__________________
________________________________________________
Best MSE for OLS regression is 0.013961731203503086
Best MSE for Ridge regression is 0.014061235826809862
Best MSE for Lasso regression is 0.019225545811630038
Best MSE for Lasso regression is 0.015434604495412393
Best MSE for Lasso regression is 0.02173944647552149
With bootstrap we achieve a minimum MSE of 0.015610 at polynomial degree 7.
With CV we achieve a minimum MSE of 0.012008 at polynomial degree 7.
Converging to optimal MSE of 0.01561035465185916 at 65 bootstrap samples.
Optimal MSE of 0.012008399799772026 at 10 folds.
_________________Terrain data___________________
________________________________________________
Best MSE for OLS regression is 0.03265626688334628
Best MSE for Ridge regression is 0.034899841977495634
Best MSE for Lasso regression is 0.03887034788761086
With bootstrap we achieve a minimum MSE of 0.040989 at polynomial degree 7.
With CV we achieve a minimum MSE of 0.029772 at polynomial degree 14.
Converging to optimal MSE of 0.03992126700670976 at 10 bootstrap samples.
Optimal MSE of 0.029771590915183666 at 10 folds.
``````
