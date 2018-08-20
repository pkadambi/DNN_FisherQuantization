from linear_model_experiment import get_acc_for_gaussian_perturbed_logistic_model_MNIST
import numpy as np

# shift_pctage = np.array([.05, .1, .5, .7, 1])
shift_pctage = np.array([.03,.05, .1, .15, .25, .3])
const_multiplier = 1.

# const_multiplier = np.zeros(np.shape(shift_pctage))
n_trials = 5
print('const mulitplier')
print(str(.1*const_multiplier))

print(len(shift_pctage))

up_accs = np.zeros([n_trials, len(shift_pctage)])
pert_accs = np.zeros([n_trials, len(shift_pctage)])
a,b = get_acc_for_gaussian_perturbed_logistic_model_MNIST(shift_pctage=shift_pctage[5],
                                                                                     const_multiplier=0.)
print('unperturbed')
print(a)
print('pert')
print(b)
exit()
for j in range(len(shift_pctage)):

    for i in range(n_trials):

        up_accs[i,j], pert_accs[i,j] = get_acc_for_gaussian_perturbed_logistic_model_MNIST(shift_pctage=shift_pctage[j], const_multiplier=const_multiplier[j])

np.savetxt('unperturbed_shift_accs.csv', up_accs, delimiter=',',fmt='%f')
np.savetxt('perturb_shift_accs.csv', pert_accs, delimiter=',',fmt='%f')