## Diagnose means for a dump_for_plot JSON:
##   - true mean (from p_y0_true / p_y1_true)
##   - model raw-marginal mean (sum(p_marg * bin_centers))
##   - EMemp mean (R's iterative Gaussian-refined estimate)
##   - Beta parameters that would result
##   - Where the R MALC output puts its mean
##
## Usage: Rscript diagnose_means.R <input.json>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript diagnose_means.R <input.json>")

suppressPackageStartupMessages({ library(jsonlite) })
source("/Users/furkandanisman/DensOLog_VS/Algorithms/MALC_Algorithm.R")

b <- fromJSON(args[1], simplifyVector = FALSE)
Y_CENTERS <- unlist(b$Y_CENTERS); Y_BIN <- Y_CENTERS[2] - Y_CENTERS[1]
edges <- unlist(b$edges_scaled)
centers <- 0.5 * (head(edges, -1) + tail(edges, -1))
delta <- edges[2] - edges[1]
grid_left <- head(edges, -1)

report <- function(tag, p_marg, p_true, r_bb) {
  cat(sprintf("── %s ──\n", tag))
  # True mean from truth density on Y_CENTERS
  true_mean <- sum(Y_CENTERS * unlist(p_true)) * Y_BIN
  cat(sprintf("  TRUE mean (from truth density)          : %.4f\n", true_mean))

  # Model raw-marginal mean — what our model spits out (already binned)
  model_mean <- sum(unlist(p_marg) * centers)
  cat(sprintf("  MODEL mean (sum p_marg*centers)         : %.4f\n", model_mean))

  # Same as R's mu_low: mean using left edges
  mu_low <- sum(unlist(p_marg) * grid_left)
  cat(sprintf("  mu_low  (sum p_marg*left_edges)         : %.4f\n", mu_low))

  # Reconstruct EMemp inputs the same way R's get_fhatn does
  pn <- unlist(p_marg)
  n_synth <- 10000
  counts <- round(n_synth * pn)
  y <- rep(grid_left, times = counts)
  sigma <- sd(y)
  cat(sprintf("  sigma (sd of expanded samples at left)  : %.4f\n", sigma))

  # EMemp iteration — three starts side-by-side for comparison
  midpt <- sum(pn * centers)
  mu_n_midpt <- EMemp(y, edges, start = midpt, sigma = sigma)$mu_hat
  mu_n_R2    <- EMemp(y, edges, start = 2,      sigma = sigma)$mu_hat
  mu_n_zero  <- EMemp(y, edges, start = 0,      sigma = sigma)$mu_hat
  cat(sprintf("  mu_n(start=midpt=%.4f)                : %.4f\n", midpt, mu_n_midpt))
  cat(sprintf("  mu_n(start=2)     (R original)         : %.4f\n", mu_n_R2))
  cat(sprintf("  mu_n(start=0)                          : %.4f\n", mu_n_zero))
  mu_n <- mu_n_midpt   # use midpoint start (matches Python) for the rest

  Delta <- mu_n - mu_low
  cat(sprintf("  Delta = mu_n - mu_low                    : %.4f  (delta=%.4f, Delta/delta=%.4f)\n",
              Delta, delta, Delta/delta))

  # Beta params for alpha = 2
  alpha <- 2
  beta_ <- 2 * alpha * (Delta/delta - 0.5)
  cat(sprintf("  beta_ (alpha=%d)                          : %.4f  → Beta(%.3f, %.3f)\n",
              alpha, beta_, alpha + beta_, alpha - beta_))
  # Effective jitter mean
  jitter_mean <- delta * (alpha + beta_) / (2 * alpha)
  cat(sprintf("  E[jitter] within bin (should equal Delta): %.4f\n", jitter_mean))

  # Where does R MALC actually put its mean?
  if (!is.null(r_bb)) {
    r_mean <- sum(Y_CENTERS * unlist(r_bb)) * Y_BIN
    cat(sprintf("  R MALC output mean                        : %.4f\n", r_mean))
  }
  cat("\n")
}

# Need R MALC output — check if the JSON has it (with_r variant)
r_y0 <- b$r_bb_p_y0; r_y1 <- b$r_bb_p_y1

report("Y0", b$p_marg_y0_raw, b$p_y0_true, r_y0)
report("Y1", b$p_marg_y1_raw, b$p_y1_true, r_y1)
