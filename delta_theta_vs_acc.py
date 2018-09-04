from delta_theta_models import get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST
import numpy as np

mean_shifts = np.array([.01, .015, .02, .025, .03, .035, .04, .045, .05, .055, .06, .065, .07, .075, .08, .085, .09, .095, .1])
variance=.005

n_trials = 5
print('const mulitplier')

up_accs = np.zeros([n_trials, len(mean_shifts), 6])
pert_accs = np.zeros([n_trials, len(mean_shifts), 6])

for j in range(len(mean_shifts)):

    for i in range(n_trials):

        up_accs[i, j, 0], pert_accs[i, j, 0] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.00, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='none')
        up_accs[i, j, 1], pert_accs[i, j, 1] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='l2')
        up_accs[i, j, 2], pert_accs[i, j, 2] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp')
        up_accs[i, j, 3], pert_accs[i, j, 3] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='diag_adam')
        up_accs[i, j, 4], pert_accs[i, j, 4] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='hvp_adam')
        up_accs[i, j, 5], pert_accs[i, j, 5] = get_acc_for_nonzero_gaussian_perturbed_two_layer_model_MNIST(mean_shifts[j], variance, const_multiplier=0.05, n_tot_iters=7000, n_fisher_iters=2000, record_tensorboard=False, regularizer_mode='l2_adam')

np.savetxt('unperturbed_shift_accs_none.csv', up_accs[:,:,1], delimiter=',',fmt='%f')
np.savetxt('unperturbed_shift_accs_l2.csv', up_accs[:,:,2], delimiter=',',fmt='%f')
np.savetxt('unperturbed_shift_accs_hvp.csv', up_accs[:,:,3], delimiter=',',fmt='%f')
np.savetxt('unperturbed_shift_accs_diag_adam.csv', up_accs[:,:,4], delimiter=',',fmt='%f')
np.savetxt('unperturbed_shift_accs_hvp_adam.csv', up_accs[:,:,5], delimiter=',',fmt='%f')
np.savetxt('unperturbed_shift_accs_l2_adam.csv', up_accs[:,:,6], delimiter=',',fmt='%f')

np.savetxt('perturb_shift_accs_none.csv', pert_accs[:,:,1], delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs_l2.csv', pert_accs[:,:,2], delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs_hvp.csv', pert_accs[:,:,3], delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs_diag_adam.csv', pert_accs[:,:,4], delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs_hvp_adam.csv', pert_accs[:,:,5], delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs_l2_adam.csv', pert_accs[:,:,6], delimiter=',',fmt='%f')



