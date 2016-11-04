from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt


def run_logistic_regression(hyperparameters, small=False, test=False):
    # TODO specify training data
    if small==False:
        train_inputs, train_targets = load_train()
    else:
        train_inputs, train_targets = load_train_small()

    valid_inputs, valid_targets = load_valid()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    
    # weights = np.random.seed(0)
    weights = np.random.normal(0, 0.05, (M+1, 1))
    # weights = np.zeros((M+1, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        if t % 10 == 0:
            print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
                   "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                       t+1, f / N, cross_entropy_train, frac_correct_train*100,
                       cross_entropy_valid, frac_correct_valid*100)
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]

        # Calculate cross entropy and classfication rate of the test set
        if test == True and t == hyperparameters['num_iterations']-1:
            test_inputs, test_targets = load_test()
            predictions_test = logistic_predict(weights, test_inputs)
            cross_entropy_test, frac_correct_test= evaluate(test_targets, predictions_test)
            print ("TEST CE:{:.6f} TEST FRAC:{:2.2f}").format(cross_entropy_test, frac_correct_test*100)

    return logging, N

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.05,
                    'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 450,
                    'weight_decay': 0 # related to standard deviation of weight prior
                    }

    cal_Test = True
    load_small = False

    # average over multiple runs
    num_runs = 1

    if hyperparameters['weight_regularization'] == True:
        weight_decay = [0.001, 0.01, 0.1, 1.0]
    else:
        weight_decay = [0]
    for alpha in weight_decay:
        hyperparameters['weight_decay'] = alpha
        logging = np.zeros((hyperparameters['num_iterations'], 5))
        print "Hyperparameters: "
        print hyperparameters
        for i in xrange(num_runs):
            res, N = run_logistic_regression(hyperparameters, load_small, cal_Test)
            logging += res
        logging /= num_runs

        print logging[-1, :]
        # TODO generate plots
        n_itr = np.arange(0, hyperparameters['num_iterations']) + 1

        if hyperparameters['weight_regularization'] == False:
            plt.plot(n_itr, logging[:, 1], 'r-', label='Cross Entropy Train')
            plt.plot(n_itr, logging[:, 3], 'g-', label='Cross Entropy Validation')
            plt.xlabel('Number of iterations')
            plt.ylabel('Cross Entropy')
            if load_small == True:
                plt.ylim([0, 40])
            plt.legend(loc='upper right')
            plt.title('Cross Entropy Vs. number of iterations')
            plt.show()
        else:
            plt.plot(n_itr, logging[:, 0] * N, 'r-', label='Loss on Training Data')
            plt.plot(n_itr, logging[:, 3], 'g-', label='Cross Entropy on Validation Data')
            # plt.plot(n_itr, logging[:, 1], 'b-', label='Cross Entropy on Training Data')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
            if load_small == True:
                plt.ylim([0, 40])
            plt.legend(loc='upper right')
            plt.title('Loss Vs. number of iterations')
            plt.show()
