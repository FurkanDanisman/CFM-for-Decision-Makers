## Compare Python's malc_1d_cvxpy vs R's log-concave MLE variants on
## the SAME IHDP query marginals + truth. Prints per-query L2 side by side.
##
## Reads a JSON blob produced by dump_marginals_for_R.py.
##
## Usage:
##   Rscript compare_malc_python_vs_r.R ihdp_marginals_r0.json

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript compare_malc_python_vs_r.R <json>")
json_path <- args[1]

suppressPackageStartupMessages({
  library(jsonlite)
  library(logcondens)
  library(logcondiscr)
})

# Load the DensOLog algorithms (get_fhatn lives here)
source("/Users/furkandanisman/DensOLog_VS/Algorithms/MALC_Algorithm.R")

# simplifyVector=FALSE keeps the queries list as a list-of-lists rather
# than collapsing to a data.frame (which breaks per-query $p_y0_true access).
blob <- fromJSON(json_path, simplifyVector = FALSE)
# But convert scalar-list fields back to vectors for convenience
blob$edges_scaled <- unlist(blob$edges_scaled)
blob$Y_CENTERS <- unlist(blob$Y_CENTERS)
blob$J <- as.integer(blob$J)
blob$n_queries <- as.integer(blob$n_queries)
blob$realization <- as.integer(blob$realization)
J <- blob$J
Y_CENTERS <- blob$Y_CENTERS
edges_scaled <- blob$edges_scaled
grid_centers <- 0.5 * (head(edges_scaled, -1) + tail(edges_scaled, -1))
Y_BIN <- Y_CENTERS[2] - Y_CENTERS[1]

cat(sprintf("Loaded %s: realization=%d  J=%d  n_queries=%d\n",
            json_path, blob$realization, J, blob$n_queries))
cat(sprintf("Y_CENTERS: %d pts on [%.2f, %.2f]\n\n",
            length(Y_CENTERS), min(Y_CENTERS), max(Y_CENTERS)))

# ── L2 helper on uniform grid ─────────────────────────────────────
l2_dist <- function(p, q, dx) sqrt(sum((p - q)^2) * dx)

# ── Normalise a density on Y_CENTERS to integrate to 1 ────────────
normalise <- function(p) {
  p[is.na(p) | is.nan(p)] <- 0
  s <- sum(p) * Y_BIN
  if (s > 0) p / s else p
}

# ── R MALC evaluator: fit + evaluate on Y_CENTERS ─────────────────
r_malc_eval <- function(p_marg, method = c("logConDens_smooth", "logConDens_raw",
                                             "get_fhatn_a0.5", "get_fhatn_a1", "get_fhatn_a2")) {
  method <- match.arg(method)
  # Synthesise samples matching the discrete probs (get_fhatn's recipe)
  n_synth <- 10000
  counts <- round(n_synth * p_marg)
  if (sum(counts) == 0) return(rep(0, length(Y_CENTERS)))
  x_synth <- rep(grid_centers, times = counts)

  set.seed(20180621)   # get_fhatn needs .Random.seed
  fit <- tryCatch({
    if (method == "logConDens_smooth") {
      logConDens(x_synth, smoothed = TRUE, print = FALSE)
    } else if (method == "logConDens_raw") {
      logConDens(x_synth, smoothed = FALSE, print = FALSE)
    } else if (method == "get_fhatn_a0.5") {
      get_fhatn(x_synth, edges_scaled, B = 10000, alpha = 0.5)$fhatn
    } else if (method == "get_fhatn_a1") {
      get_fhatn(x_synth, edges_scaled, B = 10000, alpha = 1)$fhatn
    } else {
      get_fhatn(x_synth, edges_scaled, B = 10000, alpha = 2)$fhatn
    }
  }, error = function(e) { cat("  R fit error:", conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) return(rep(NA, length(Y_CENTERS)))
  ev <- tryCatch(evaluateLogConDens(Y_CENTERS, fit, which = 4)[, 5],
                 error = function(e) rep(NA, length(Y_CENTERS)))
  normalise(ev)
}

R_METHODS <- c("logConDens_smooth", "logConDens_raw",
                "get_fhatn_a0.5", "get_fhatn_a1", "get_fhatn_a2")

# Per-query L2 accumulators
py_l2_y0 <- numeric(0); py_l2_y1 <- numeric(0)
r_l2_y0 <- setNames(rep(list(numeric(0)), length(R_METHODS)), R_METHODS)
r_l2_y1 <- setNames(rep(list(numeric(0)), length(R_METHODS)), R_METHODS)

for (i in seq_len(blob$n_queries)) {
  q <- blob$queries[[i]]
  p_y0_true <- unlist(q$p_y0_true); p_y1_true <- unlist(q$p_y1_true)
  py_p_y0   <- unlist(q$py_malc_p_y0); py_p_y1 <- unlist(q$py_malc_p_y1)
  p_marg_y0 <- unlist(q$p_marg_y0_raw); p_marg_y1 <- unlist(q$p_marg_y1_raw)

  # Python L2 (should match what summary_ihdp.py prints)
  py_l2_y0[i] <- l2_dist(py_p_y0, p_y0_true, Y_BIN)
  py_l2_y1[i] <- l2_dist(py_p_y1, p_y1_true, Y_BIN)

  # R L2 for each method
  for (m in R_METHODS) {
    rp0 <- r_malc_eval(p_marg_y0, method = m)
    rp1 <- r_malc_eval(p_marg_y1, method = m)
    r_l2_y0[[m]][i] <- l2_dist(rp0, p_y0_true, Y_BIN)
    r_l2_y1[[m]][i] <- l2_dist(rp1, p_y1_true, Y_BIN)
  }
  cat(sprintf("  q=%2d done\n", as.integer(q$q)))
}

cat("\n=== L2 y0 (mean ± SEM) — lower is better ===\n")
mean_sem <- function(v) {
  v <- v[is.finite(v)]
  if (length(v) < 2) sprintf("%.4f", mean(v))
  else sprintf("%.4f ± %.4f", mean(v), sd(v) / sqrt(length(v)))
}
cat(sprintf("  Py malc_1d_cvxpy         : %s   (n=%d)\n",
            mean_sem(py_l2_y0), sum(is.finite(py_l2_y0))))
for (m in R_METHODS) {
  cat(sprintf("  R %-24s : %s   (n=%d)\n",
              m, mean_sem(r_l2_y0[[m]]), sum(is.finite(r_l2_y0[[m]]))))
}

cat("\n=== L2 y1 (mean ± SEM) ===\n")
cat(sprintf("  Py malc_1d_cvxpy         : %s\n", mean_sem(py_l2_y1)))
for (m in R_METHODS) {
  cat(sprintf("  R %-24s : %s\n", m, mean_sem(r_l2_y1[[m]])))
}

cat("\n=== Wins: how often does each R method beat Py per-query? ===\n")
for (m in R_METHODS) {
  wins_y0 <- sum(r_l2_y0[[m]] < py_l2_y0, na.rm = TRUE)
  wins_y1 <- sum(r_l2_y1[[m]] < py_l2_y1, na.rm = TRUE)
  cat(sprintf("  R %-24s : y0=%d/%d  y1=%d/%d\n", m,
              wins_y0, blob$n_queries, wins_y1, blob$n_queries))
}
