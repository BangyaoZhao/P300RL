# devtools::install_github("kangjian2016/fastBayesReg")
# devtools::install_github("kangjian2016/BayesGPfit")
library(fastBayesReg)
library(BayesGPfit)
library(reshape2)
library(ggplot2)


poly_degree = 30
a = 0.01
b = 0.01
d = 1
x <- GP.generate.grids(d = d,
                       num_grids = 500,
                       grids_lim = c(0, 1))

Psi <- GP.eigen.funcs.fast(x,
                           poly_degree = poly_degree,
                           a = a,
                           b = b)
lambda <- GP.eigen.value(
  poly_degree = poly_degree,
  a = a,
  b = b,
  d = d
)
# save result
library(jsonlite)
result = list(x = as.vector(x), Psi = Psi, lambda = lambda)
# Convert list to JSON
json_data <- toJSON(result, pretty = TRUE)
write(json_data, file = "basis.json")
