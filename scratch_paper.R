em_loop = function(K) {
    # Initializing data
    set.seed(1234567890)
    max_it = 1 # max number of EM iterations
    min_change = 0.1 # min change in log likelihood between two consecutive EM iterations
    N = 1000 # number of training points
    D = 10 # number of dimensions
    x = matrix(nrow=N, ncol = D) # training data
    true_pi = vector(length = K) # true mixing coefficients
    true_mu = matrix(nrow = K, ncol = D) # true conditional distributions
    true_pi = c(rep(1/K, K))
    if (K == 2) {
        true_mu[1,] = c(0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1,1)
        true_mu[2,] = c(0.5,0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9,0)
        plot(true_mu[1,], type = "o", xlab = "dimension", col = "blue",
             ylim = c(0,1), main = "True")
        points(true_mu[2,], type="o", xlab = "dimension", col = "red",
               main = "True")
    } else if (K == 3) {
        true_mu[1,] = c(0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1,1)
        true_mu[2,] = c(0.5,0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9,0)
        true_mu[3,] = c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
        plot(true_mu[1,], type = "o", xlab = "dimension", col = "blue", ylim=c(0,1),
             main = "True")
        points(true_mu[2,], type = "o", xlab = "dimension", col = "red",
               main = "True")
        points(true_mu[3,], type = "o", xlab = "dimension", col = "green",
               main = "True")
    } else {
        true_mu[1,] = c(0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1,1)
        true_mu[2,] = c(0.5,0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9,0)
        true_mu[3,] = c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
        true_mu[4,] = c(0.3,0.5,0.5,0.7,0.5,0.5,0.5,0.5,0.4,0.5)
        plot(true_mu[1,], type = "o", xlab = "dimension", col = "blue",
             ylim = c(0,1), main = "True")
        points(true_mu[2,], type = "o", xlab = "dimension", col = "red",
               main = "True")
        points(true_mu[3,], type = "o", xlab = "dimension", col = "green",
               main = "True")
        points(true_mu[4,], type = "o", xlab = "dimension", col = "yellow",
               main = "True")
    }
    z = matrix(nrow = N, ncol = K) # fractional component assignments
    pi = vector(length = K) # mixing coefficients
    mu = matrix(nrow = K, ncol = D) # conditional distributions
    llik = vector(length = max_it) # log likelihood of the EM iterations
    # Producing the training data
    for(n in 1:N) {
        k = sample(1:K, 1, prob=true_pi)
        for(d in 1:D) {
            x[n,d] = rbinom(1, 1, true_mu[k,d])
        }
    }
    # Random initialization of the paramters
    pi = runif(K, 0.49, 0.51)
    pi = pi / sum(pi)
    for(k in 1:K) {
        mu[k,] = runif(D, 0.49, 0.51)
    }
    #EM algorithm
    for(it in 1:max_it) {
        # Plotting mu
        # Defining plot title
        title = paste0("Iteration", it)
        if (K == 2) {
            plot(mu[1,], type = "o", xlab = "dimension", col = "blue", ylim = c(0,1), main = title)
            points(mu[2,], type = "o", xlab = "dimension", col = "red", main = title)
        } else if (K == 3) {
            plot(mu[1,], type = "o", xlab = "dimension", col = "blue", ylim = c(0,1), main = title)
            points(mu[2,], type = "o", xlab = "dimension", col = "red", main = title)
            points(mu[3,], type = "o", xlab = "dimension", col = "green", main = title)
        } else {
            plot(mu[1,], type = "o", xlab = "dimension", col = "blue", ylim = c(0,1), main = title)
            points(mu[2,], type = "o", xlab = "dimension", col = "red", main = title)
            points(mu[3,], type = "o", xlab = "dimension", col = "green", main = title)
            points(mu[4,], type = "o", xlab = "dimension", col = "yellow", main = title)
        }
        Sys.sleep(0.5)
        # E-step: Computation of the fractional component assignments
        for (n in 1:N) {
            # Creating empty matrix (column 1:K = p_x_given_k; column K+1 = p(x|all k)
            p_x = matrix(data = c(rep(1,K), 0), nrow = 1, ncol = K+1)
            
            # Calculating p(x|k) and p(x|all k)
            for (k in 1:K) {
                # Calculating p(x|k)
                for (d in 1:D) {
                    p_x[1,k] = p_x[1,k] * (mu[k,d]^x[n,d]) * (1-mu[k,d])^(1-x[n,d])
                }
                p_x[1,k] = p_x[1,k] * pi[k] # weighting with pi[k]
                # Calculating p(x|all k) (denominator)
                p_x[1,K+1] = p_x[1,K+1] + p_x[1,k]
            }
            # Calculating z for n and all k
            for (k in 1:K) {
                z[n,k] = p_x[1,k] / p_x[1,K+1]
            }
        }
        print(z)
        # Log likelihood computation
        for (n in 1:N) {
            for (k in 1:K) {
                log_term = 0
                for (d in 1:D) {
                    log_term = log_term + x[n,d] * log(mu[k,d]) + (1-x[n,d]) * log(1-mu[k,d])
                }
                llik[it] = llik[it] + z[n,k] * (log(pi[k]) + log_term)
            }
        }
        cat("iteration: ", it, "log likelihood: ", llik[it], "\n")
        flush.console()
        # Stop if the log likelihood has not changed significantly
        if (it != 1) {
            if (abs(llik[it] - llik[it-1]) < min_change) {
                break
            }
        }
        # M-step: ML parameter estimation from the data and fractional component assignments
        # Updating pi
        for (k in 1:K) {
            pi[k] = sum(z[,k])/N
        }
        # Updating mu
        for (k in 1:K) {
            mu[k,] = 0
            for (n in 1:N) {
                mu[k,] = mu[k,] + x[n,] * z[n,k]
            }
            mu[k,] = mu[k,] / sum(z[,k])
        }
    }
    # Printing pi, mu and development of log likelihood at the end
    return(list(
        pi = pi,
        mu = mu,
        logLikelihoodDevelopment = plot(llik[1:it],
                                        type = "o",
                                        main = "Development of the log likelihood",
                                        xlab = "iteration",
                                        ylab = "log likelihood")
    ))
}
a <- em_loop()
