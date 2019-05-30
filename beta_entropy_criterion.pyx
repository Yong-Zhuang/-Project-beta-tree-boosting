from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t
from libc.math cimport sqrt, pow
from libc.math cimport abs
cdef class BetaEntropy(ClassificationCriterion):
    cdef double beta = 0.001
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy += pow(count_k,beta)
            sum_total += self.sum_stride
        entropy = (1-entropy)/(1-pow(2,float(1-beta)))
        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left += pow(count_k,beta)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right += pow(count_k,beta)

            sum_left += self.sum_stride
            sum_right += self.sum_stride
        entropy_left = (1-entropy_left)/(1-pow(2,float(1-beta)))
        entropy_right = (1-entropy_right)/(1-pow(2,float(1-beta)))
        impurity_left[0] = entropy_left / self.n_outputs  
        impurity_right[0] = entropy_right / self.n_outputs
