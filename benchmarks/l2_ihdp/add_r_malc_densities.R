## Read a dump_for_plot.py JSON, fit R MALC on the raw p_mat marginals,
## evaluate on Y_CENTERS, and write the R MALC densities to a new JSON.
##
## Usage:
##   Rscript add_r_malc_densities.R input.json output.json
##
## Default method: logConDens(smoothed=TRUE) — R's smoothed log-concave MLE.
## Override with env RMALC_METHOD ∈ {logConDens_smooth, logConDens_raw,
## get_fhatn_a0.5, get_fhatn_a1, get_fhatn_a2}.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript add_r_malc_densities.R <input.json> <output.json>")
in_path <- args[1]; out_path <- args[2]

suppressPackageStartupMessages({
  library(jsonlite); library(logcondens); library(logcondiscr)
})
source("/Users/furkandanisman/DensOLog_VS/Algorithms/MALC_Algorithm.R")

METHOD <- Sys.getenv("RMALC_METHOD", "logConDens_smooth")
cat(sprintf("[R] method=%s\n", METHOD))

blob <- fromJSON(in_path, simplifyVector = FALSE)
blob$Y_CENTERS <- unlist(blob$Y_CENTERS)
blob$edges_scaled <- unlist(blob$edges_scaled)
Y_CENTERS <- blob$Y_CENTERS
Y_BIN <- Y_CENTERS[2] - Y_CENTERS[1]
grid_centers <- 0.5 * (head(blob$edges_scaled, -1) + tail(blob$edges_scaled, -1))

normalise <- function(p) {
  p[is.na(p) | is.nan(p)] <- 0
  s <- sum(p) * Y_BIN
  if (s > 0) p / s else p
}

r_malc_eval <- function(p_marg) {
  n_synth <- 10000
  counts <- round(n_synth * p_marg)
  if (sum(counts) == 0) return(rep(0, length(Y_CENTERS)))
  x_synth <- rep(grid_centers, times = counts)
  set.seed(20180621)
  fit <- tryCatch({
    if (METHOD == "logConDens_smooth")   logConDens(x_synth, smoothed = TRUE, print = FALSE)
    else if (METHOD == "logConDens_raw")   logConDens(x_synth, smoothed = FALSE, print = FALSE)
    else if (METHOD == "get_fhatn_a0.5") get_fhatn(x_synth, blob$edges_scaled, B = 10000, alpha = 0.5)$fhatn
    else if (METHOD == "get_fhatn_a1")   get_fhatn(x_synth, blob$edges_scaled, B = 10000, alpha = 1)$fhatn
    else if (METHOD == "get_fhatn_a2")   get_fhatn(x_synth, blob$edges_scaled, B = 10000, alpha = 2)$fhatn
    else stop(paste("unknown method:", METHOD))
  }, error = function(e) { cat("  R fit error:", conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) return(rep(NA_real_, length(Y_CENTERS)))
  ev <- tryCatch(evaluateLogConDens(Y_CENTERS, fit, which = 4)[, 5],
                 error = function(e) rep(NA_real_, length(Y_CENTERS)))
  normalise(ev)
}

blob$r_bb_method <- METHOD
blob$r_bb_p_y0 <- r_malc_eval(unlist(blob$p_marg_y0_raw))
blob$r_bb_p_y1 <- r_malc_eval(unlist(blob$p_marg_y1_raw))
# CATE via naive convolution of R marginals — independence assumption
# (R can't do 2D MALC easily; this at least gives an R-side τ)
p0 <- blob$r_bb_p_y0; p1 <- blob$r_bb_p_y1
if (any(is.na(p0)) || any(is.na(p1))) {
  blob$r_bb_p_tau <- rep(NA_real_, length(unlist(blob$TAU_CENTERS)))
} else {
  n <- length(Y_CENTERS)
  # p(τ) = ∫ p1(y0 + τ) p0(y0) dy0, on TAU_CENTERS grid
  # Same as summary_ihdp's naive_p_tau but done in R
  conv <- rev(convolve(p1, rev(p0), type = "open")) * Y_BIN
  tau_native <- seq(-(n-1), n-1) * Y_BIN
  TAU_CENTERS <- unlist(blob$TAU_CENTERS)
  blob$r_bb_p_tau <- approx(tau_native, conv, xout = TAU_CENTERS,
                             rule = 2, yleft = 0, yright = 0)$y
  s <- sum(blob$r_bb_p_tau) * (TAU_CENTERS[2] - TAU_CENTERS[1])
  if (s > 0) blob$r_bb_p_tau <- blob$r_bb_p_tau / s
}

write_json(blob, out_path, auto_unbox = TRUE, digits = NA)
cat(sprintf("[done] %s\n", out_path))
