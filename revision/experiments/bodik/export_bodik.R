# Export the IHDP-rho and Twins datasets of Bodik (CLeaR 2026, "Retrospective Counterfactual Prediction by
# Conditioning on the Factual Outcome"), exactly as his experiment script builds them.
#   Rscript export_bodik.R <path to a clone of github.com/jurobodik/Counterfactual_prediction> <out dir>
#
# Only his two data functions are evaluated, VERBATIM from his helpers.R (IHDP_with_rho, Twins_upload); the rest
# of helpers.R only loads plotting packages. The seeds are those of Experiments/Simulations 1/Main_script.R:
#   base_seed = 123, n_reps = 50, idx_seed = (setup_index - 1) * n_reps + rep, set.seed(base_seed + idx_seed),
#   then IHDP_with_rho(rho, load_csv_file = TRUE)   (no other random draw between the two).
# His setup order: 1-3 Twins (d = 71, 10, 1); 4-8 IHDP d = 10 with rho = 1, .75, .5, .25, 0;
#                  9-13 IHDP d = 1 with the same rho order.
# Output (plain CSV, converted to npz by make_worlds.py):
#   ihdp_X.csv          the 25 covariates (all 25 enter the outcome; the methods see the first d)
#   ihdp_T.csv          treatment
#   ihdp_Y.csv          columns d, rho, rep, unit, Y0, Y1, mu0, mu1   (d = 10 and d = 1 setups, 50 reps each)
#   twins.csv           his Twins_upload(d = 71) output: X1..X71, treatment, Y_obs, Y0, Y1, Y_cf
args <- commandArgs(trailingOnly = TRUE)
SRC <- normalizePath(args[1]); OUT <- args[2]
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages({ library(MASS); library(readr) })

ex <- parse(file.path(SRC, "helpers.R"))
want <- c("IHDP_with_rho", "Twins_upload")
for (e in ex) {
  if (is.call(e) && as.character(e[[1]]) %in% c("<-", "=") && as.character(e[[2]]) %in% want) eval(e, globalenv())
}
stopifnot(exists("IHDP_with_rho"), exists("Twins_upload"))
# The same function with mu0 / mu1 added to its return value (the PEHE truth). The edit is to the final
# data.frame only, so the random draws (sample, then mvrnorm) are made in the same order; checked below.
src <- paste(deparse(IHDP_with_rho), collapse = "\n")
src2 <- sub("Y_cf = Y_cf\\))\\s*}\\s*$", "Y_cf = Y_cf, mu0 = as.numeric(mu0), mu1 = as.numeric(mu1)))\n}", src)
stopifnot(src2 != src)
IHDP_with_rho_mu <- eval(parse(text = src2))

base_seed <- 123; n_reps <- 50
setups <- list()
for (rho in c(1, 0.75, 0.5, 0.25, 0)) setups[[length(setups) + 1]] <- list(d = 10, rho = rho)
for (rho in c(1, 0.75, 0.5, 0.25, 0)) setups[[length(setups) + 1]] <- list(d = 1, rho = rho)
first_ihdp <- 4L                                   # setup_index of the first IHDP setup in his list

owd <- setwd(file.path(SRC, "Experiments", "Datasets"))   # IHDP_with_rho reads "ihdp_data.csv" from the cwd
rows <- list(); X0 <- NULL; T0 <- NULL
for (s in seq_along(setups)) {
  setup_index <- first_ihdp + s - 1L
  for (rep in seq_len(n_reps)) {
    set.seed(base_seed + (setup_index - 1L) * n_reps + rep)
    dat <- IHDP_with_rho_mu(setups[[s]]$rho, load_csv_file = TRUE)
    if (rep == 1L) {                                # his unpatched function, same seed: identical outcomes
      set.seed(base_seed + (setup_index - 1L) * n_reps + rep)
      ref <- IHDP_with_rho(setups[[s]]$rho, load_csv_file = TRUE)
      stopifnot(identical(ref$Y0, dat$Y0), identical(ref$Y1, dat$Y1), identical(ref$treatment, dat$treatment))
    }
    X <- as.matrix(dat[, paste0("X", 1:25)])
    if (is.null(X0)) { X0 <- X; T0 <- dat$treatment }
    stopifnot(all(X == X0), all(dat$treatment == T0))          # covariates and treatment are fixed across reps
    rows[[length(rows) + 1]] <- data.frame(d = setups[[s]]$d, rho = setups[[s]]$rho, rep = rep,
                                           unit = seq_len(nrow(dat)), Y0 = dat$Y0, Y1 = dat$Y1,
                                           mu0 = dat$mu0, mu1 = dat$mu1)
  }
}
tw <- Twins_upload(path = "twins.csv", d = 71)
setwd(owd)

write.csv(X0, file.path(OUT, "ihdp_X.csv"), row.names = FALSE)
write.csv(data.frame(treatment = T0), file.path(OUT, "ihdp_T.csv"), row.names = FALSE)
Y <- do.call(rbind, rows)
write.csv(Y, file.path(OUT, "ihdp_Y.csv"), row.names = FALSE)
write.csv(tw, file.path(OUT, "twins.csv"), row.names = FALSE)
cat(sprintf("IHDP: %d units, %d covariates, %d datasets; Twins: %d pairs, %d covariates\n",
            nrow(X0), ncol(X0), nrow(Y) / nrow(X0), nrow(tw), sum(grepl("^X[0-9]+$", names(tw)))))
for (s in seq_along(setups)) {                     # sanity: realized noise correlation per setup
  y <- Y[Y$d == setups[[s]]$d & Y$rho == setups[[s]]$rho, ]
  cat(sprintf("  d=%2d rho=%.2f  ATE=%.3f\n", setups[[s]]$d, setups[[s]]$rho, mean(y$Y1 - y$Y0)))
}
