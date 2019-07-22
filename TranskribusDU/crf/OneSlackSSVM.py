# -*- coding: utf-8 -*-

"""
    An extension of the pystruct OneSlackSSVM module to have a fit_with_valid
    method on it

    Copyright Xerox(C) 2016 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
"""

from time import time

import numpy as np
import cvxopt.solvers

from pystruct.learners import OneSlackSSVM as Pystruct_OneSlackSSVM
from pystruct.learners.one_slack_ssvm import NoConstraint


class OneSlackSSVM(Pystruct_OneSlackSSVM):
    """
    Same as its parent with an additional method: fit_with_valid
    """

    def __init__(self, model, max_iter=10000, C=1.0, check_constraints=False,
                 verbose=0, negativity_constraint=None, n_jobs=1,
                 break_on_bad=False, show_loss_every=0, tol=1e-3,
                 inference_cache=0, inactive_threshold=1e-5,
                 inactive_window=50, logger=None, cache_tol='auto',
                 switch_to=None):
        
        Pystruct_OneSlackSSVM.__init__(self, model, max_iter=max_iter, C=C, check_constraints=check_constraints,
                 verbose=verbose, negativity_constraint=negativity_constraint, n_jobs=n_jobs,
                 break_on_bad=break_on_bad, show_loss_every=show_loss_every, tol=tol,
                 inference_cache=inference_cache, inactive_threshold=inactive_threshold,
                 inactive_window=inactive_window, logger=logger, cache_tol=cache_tol,
                 switch_to=switch_to)
        
        
        
    def fit_with_valid(self, X, Y, lX_vld, lY_vld, constraints=None
                       , warm_start=False, initialize=True
                       , valid_every=50):
        """Learn parameters using cutting plane method.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        lX_vld, lY_vld : iterable X and Y validation set
        
        contraints : ignored

        warm_start : bool, default=False
            Whether we are warmstarting from a previous fit.

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
            
        valid_every : integer. Periodic check with validation set to get best model
         
        """
        best_iteration = -1
        try:
            self._fit_valid_best_score
            print("score of best model: %.6f"%self._fit_valid_best_score)
        except:
            self._fit_valid_best_score = -99999
            
        
        if self.verbose:
            print("Training 1-slack dual structural SVM")
        cvxopt.solvers.options['show_progress'] = self.verbose > 3
        if initialize:
            self.model.initialize(X, Y)

        # parse cache_tol parameter
        if self.cache_tol is None or self.cache_tol == 'auto':
            self.cache_tol_ = self.tol
        else:
            self.cache_tol_ = self.cache_tol

        if not warm_start:
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.objective_curve_, self.primal_objective_curve_ = [], []
            self.cached_constraint_ = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])
            self.inference_cache_ = None
            self.timestamps_ = [time()]
        elif warm_start == "soft":
            self.w = np.zeros(self.model.size_joint_feature)
            constraints = []
            self.alphas = []  # dual solutions
            # append constraint given by ground truth to make our life easier
            constraints.append((np.zeros(self.model.size_joint_feature), 0))
            self.alphas.append([self.C])

        else:
            constraints = self.constraints_
        self.last_slack_ = -1

        # get the joint_feature of the ground truth
        if getattr(self.model, 'rescale_C', False):
            joint_feature_gt = self.model.batch_joint_feature(X, Y, Y)
        else:
            joint_feature_gt = self.model.batch_joint_feature(X, Y)

        try:
            # catch ctrl+c to stop training

            for iteration in range(self.max_iter):
                # main loop
                cached_constraint = False
                if self.verbose > 0:
                    print("----- %d -----"%iteration)
                if self.verbose > 2:
                    print(self)
                try:
                    Y_hat, djoint_feature, loss_mean = self._constraint_from_cache(
                        X, Y, joint_feature_gt, constraints)
                    cached_constraint = True
                except NoConstraint:
                    try:
                        Y_hat, djoint_feature, loss_mean = self._find_new_constraint(
                            X, Y, joint_feature_gt, constraints)
                        self._update_cache(X, Y, Y_hat)
                    except NoConstraint:
                        if self.verbose:
                            print("no additional constraints")
                        if (self.switch_to is not None
                                and self.model.inference_method !=
                                self.switch_to):
                            if self.verbose:
                                print(("Switching to %s inference" %
                                      str(self.switch_to)))
                            self.model.inference_method_ = \
                                self.model.inference_method
                            self.model.inference_method = self.switch_to
                            continue
                        else:
                            break

                self.timestamps_.append(time() - self.timestamps_[0])
                self._compute_training_loss(X, Y, iteration)
                constraints.append((djoint_feature, loss_mean))

                # compute primal objective
                last_slack = -np.dot(self.w, djoint_feature) + loss_mean
                primal_objective = (self.C * len(X)
                                    * max(last_slack, 0)
                                    + np.sum(self.w ** 2) / 2)
                self.primal_objective_curve_.append(primal_objective)
                self.cached_constraint_.append(cached_constraint)

                objective = self._solve_1_slack_qp(constraints,
                                                   n_samples=len(X))

                # update cache tolerance if cache_tol is auto:
                if self.cache_tol == "auto" and not cached_constraint:
                    self.cache_tol_ = (primal_objective - objective) / 4

                self.last_slack_ = np.max([(-np.dot(self.w, djoint_feature) + loss_mean)
                                           for djoint_feature, loss_mean in constraints])
                self.last_slack_ = max(self.last_slack_, 0)

                if self.verbose > 0:
                    # the cutting plane objective can also be computed as
                    # self.C * len(X) * self.last_slack_ + np.sum(self.w**2)/2
                    print(("cutting plane objective: %f, primal objective %f"
                          % (objective, primal_objective)))
                # we only do this here because we didn't add the gt to the
                # constraints, which makes the dual behave a bit oddly
                self.objective_curve_.append(objective)
                self.constraints_ = constraints
                if self.logger is not None:
                    if iteration % valid_every == 0:
                        cur_score = self.score(lX_vld, lY_vld)
                        #print(self._fit_valid_best_score, cur_score)
                        if cur_score > self._fit_valid_best_score:
                            best_iteration = iteration
                            self._fit_valid_best_score = cur_score
                            self.logger(self, 'final')
                            if self.verbose > 0: print("Current model is best with validation score=%.6f" % self._fit_valid_best_score)
                        else:
                            # we save the last model, even if it is not the best, in case of warm start
                            self.logger.save(self, self.logger.file_name + "._last_")
                            print("Current validation score=%.6f  (best=%.6f at iteration %d)" % (cur_score, self._fit_valid_best_score, best_iteration))

                if self.verbose > 5:
                    print((self.w))
                    
                
                
        except KeyboardInterrupt:
            pass
        if self.verbose and self.n_jobs == 1:
            print(("calls to inference: %d" % self.model.inference_calls))
        # compute final objective:
        self.timestamps_.append(time() - self.timestamps_[0])
        primal_objective = self._objective(X, Y)
        self.primal_objective_curve_.append(primal_objective)
        self.objective_curve_.append(objective)
        self.cached_constraint_.append(False)

        if self.logger is not None:
            cur_score = self.score(lX_vld, lY_vld)
            # print("finished ", self._fit_valid_best_score, cur_score)
            if cur_score > self._fit_valid_best_score:
                self._fit_valid_best_score = cur_score
                best_iteration = iteration
                self.logger(self, 'final')
        if self.verbose > 0: print("Best model saved at iteration %d: validation score=%.6f" % (best_iteration, self._fit_valid_best_score))
                

        if self.verbose > 0:
            print(("final primal objective: %f gap: %f  (validation score: %.6f)"
                  % (primal_objective, primal_objective - objective, cur_score)))

        return self

