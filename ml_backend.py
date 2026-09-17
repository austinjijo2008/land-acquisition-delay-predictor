"""
ML backend shim used by train_and_export.py.

Tries to import the real scikit-learn implementations first (StandardScaler,
LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
train_test_split, accuracy_score, confusion_matrix, classification_report).

If scikit-learn/scipy cannot be imported (e.g. blocked by a machine's
Application Control / WDAC policy, as happened during development of this
project), falls back to pure-NumPy reimplementations of the same algorithms
with the same constructor arguments and the same predict/predict_proba/fit
interface, so train_and_export.py and app.py work unchanged either way.

The fallback logistic regression solves the exact same objective scikit-learn
uses by default (L2-regularized log-loss, C=1.0, intercept unregularized) via
Newton-Raphson/IRLS, which converges to the same unique optimum since the
objective is convex. The fallback decision tree/random forest are CART/Gini
implementations mirroring scikit-learn's defaults.

BACKEND module-level string reports which path is active ("sklearn" or
"numpy_fallback"); train_and_export.py records this in metrics.json for
transparency.
"""

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    BACKEND = "sklearn"

except ImportError:
    BACKEND = "numpy_fallback"

    def _to_array(X):
        return np.asarray(X, dtype=float)

    class StandardScaler:
        def fit(self, X):
            X = _to_array(X)
            self.mean_ = X.mean(axis=0)
            scale = X.std(axis=0, ddof=0)
            scale = np.where(scale == 0, 1.0, scale)
            self.scale_ = scale
            return self

        def transform(self, X):
            X = _to_array(X)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
        rng = np.random.RandomState(random_state)
        y_arr = np.asarray(stratify if stratify is not None else y)
        train_parts, test_parts = [], []
        for c in np.unique(y_arr):
            idx = np.where(y_arr == c)[0].copy()
            rng.shuffle(idx)
            n_test = int(round(len(idx) * test_size))
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def confusion_matrix(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        return np.array([[tn, fp], [fn, tp]])

    def classification_report(y_true, y_pred, output_dict=False):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        support0, support1 = tn + fp, tp + fn
        total = support0 + support1

        def safe_div(a, b):
            return a / b if b else 0.0

        precision0, recall0 = safe_div(tn, tn + fn), safe_div(tn, support0)
        precision1, recall1 = safe_div(tp, tp + fp), safe_div(tp, support1)
        f1_0 = safe_div(2 * precision0 * recall0, precision0 + recall0)
        f1_1 = safe_div(2 * precision1 * recall1, precision1 + recall1)
        accuracy = safe_div(tp + tn, total)
        report = {
            "0": {"precision": precision0, "recall": recall0, "f1-score": f1_0, "support": int(support0)},
            "1": {"precision": precision1, "recall": recall1, "f1-score": f1_1, "support": int(support1)},
            "accuracy": accuracy,
            "macro avg": {
                "precision": (precision0 + precision1) / 2,
                "recall": (recall0 + recall1) / 2,
                "f1-score": (f1_0 + f1_1) / 2,
                "support": int(total),
            },
            "weighted avg": {
                "precision": safe_div(precision0 * support0 + precision1 * support1, total),
                "recall": safe_div(recall0 * support0 + recall1 * support1, total),
                "f1-score": safe_div(f1_0 * support0 + f1_1 * support1, total),
                "support": int(total),
            },
        }
        if output_dict:
            return report
        lines = [f"{k}: {v}" for k, v in report.items()]
        return "\n".join(lines)

    class LogisticRegression:
        """Newton-Raphson / IRLS solver for L2-regularized logistic regression,
        matching scikit-learn's default objective (penalty='l2', C=1.0,
        intercept unregularized)."""

        def __init__(self, C=1.0, max_iter=100, tol=1e-8, fit_intercept=True, **_ignored):
            self.C = C
            self.max_iter = max_iter
            self.tol = tol
            self.fit_intercept = fit_intercept

        def fit(self, X, y):
            X = _to_array(X)
            y = np.asarray(y, dtype=float)
            n, d = X.shape
            Xb = np.hstack([np.ones((n, 1)), X]) if self.fit_intercept else X
            p_dim = Xb.shape[1]
            w = np.zeros(p_dim)
            lam = 1.0 / self.C
            reg = np.full(p_dim, lam)
            if self.fit_intercept:
                reg[0] = 0.0
            eps = 1e-12
            for _ in range(max(self.max_iter, 25)):
                z = np.clip(Xb @ w, -35, 35)
                p = 1.0 / (1.0 + np.exp(-z))
                p_c = np.clip(p, eps, 1 - eps)
                grad = Xb.T @ (p_c - y) + reg * w
                wgt = np.maximum(p_c * (1 - p_c), 1e-10)
                H = (Xb * wgt[:, None]).T @ Xb + np.diag(reg) + np.eye(p_dim) * 1e-10
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(H, grad, rcond=None)[0]
                w_new = w - step
                if np.linalg.norm(w_new - w) < self.tol:
                    w = w_new
                    break
                w = w_new
            if self.fit_intercept:
                self.intercept_ = np.array([w[0]])
                self.coef_ = w[1:].reshape(1, -1)
            else:
                self.intercept_ = np.array([0.0])
                self.coef_ = w.reshape(1, -1)
            return self

        def decision_function(self, X):
            X = _to_array(X)
            return X @ self.coef_[0] + self.intercept_[0]

        def predict_proba(self, X):
            z = np.clip(self.decision_function(X), -35, 35)
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    class _Node:
        __slots__ = ["is_leaf", "prediction", "proba", "feature", "threshold", "left", "right", "n_samples", "gain"]

    def _best_split(X, y, feature_indices):
        n = X.shape[0]
        n1_total = float(np.sum(y == 1))
        n0_total = n - n1_total
        parent_gini = 1.0 - (n0_total / n) ** 2 - (n1_total / n) ** 2
        best_gain, best_feat, best_thr = -1.0, None, None
        for f in feature_indices:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            y_sorted = y[order]
            is1 = (y_sorted == 1).astype(float)
            cum1 = np.cumsum(is1)
            cum0 = np.cumsum(1 - is1)
            distinct = np.where(np.diff(col_sorted) > 1e-12)[0]
            if len(distinct) == 0:
                continue
            left_n = distinct + 1
            left1 = cum1[distinct]
            left0 = cum0[distinct]
            right_n = n - left_n
            right1 = cum1[-1] - left1
            right0 = cum0[-1] - left0
            valid = (left_n > 0) & (right_n > 0)
            if not np.any(valid):
                continue
            ln, rn = left_n[valid].astype(float), right_n[valid].astype(float)
            l1, l0, r1, r0 = left1[valid], left0[valid], right1[valid], right0[valid]
            left_gini = 1.0 - (l1 / ln) ** 2 - (l0 / ln) ** 2
            right_gini = 1.0 - (r1 / rn) ** 2 - (r0 / rn) ** 2
            weighted = (ln * left_gini + rn * right_gini) / n
            gain = parent_gini - weighted
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                pos = distinct[valid][i]
                best_gain = float(gain[i])
                best_feat = f
                best_thr = (col_sorted[pos] + col_sorted[pos + 1]) / 2.0
        return best_feat, best_thr, best_gain

    class DecisionTreeClassifier:
        def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None, **_ignored):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.max_features = max_features
            self.random_state = random_state

        def fit(self, X, y):
            X = _to_array(X)
            y = np.asarray(y, dtype=int)
            self.n_features_ = X.shape[1]
            self._rng = np.random.RandomState(self.random_state)
            self._importance = np.zeros(self.n_features_)
            self.root_ = self._build(X, y, depth=0)
            total = self._importance.sum()
            self.feature_importances_ = self._importance / total if total > 0 else self._importance
            return self

        def _leaf(self, y):
            n1 = int(np.sum(y == 1))
            total = len(y)
            n0 = total - n1
            node = _Node()
            node.is_leaf = True
            node.prediction = 1 if n1 >= n0 else 0
            node.proba = np.array([n0 / total, n1 / total]) if total else np.array([0.5, 0.5])
            return node

        def _build(self, X, y, depth):
            n = len(y)
            if n < self.min_samples_split or len(np.unique(y)) == 1 or (
                self.max_depth is not None and depth >= self.max_depth
            ):
                return self._leaf(y)
            d = X.shape[1]
            if self.max_features is None:
                feat_idx = np.arange(d)
            else:
                k = self.max_features if isinstance(self.max_features, int) else max(1, int(np.sqrt(d)))
                feat_idx = self._rng.choice(d, size=min(k, d), replace=False)
            feat, thr, gain = _best_split(X, y, feat_idx)
            if feat is None or gain <= 1e-12:
                return self._leaf(y)
            mask = X[:, feat] <= thr
            if mask.sum() == 0 or (~mask).sum() == 0:
                return self._leaf(y)
            self._importance[feat] += gain * n
            node = _Node()
            node.is_leaf = False
            node.feature = feat
            node.threshold = thr
            node.left = self._build(X[mask], y[mask], depth + 1)
            node.right = self._build(X[~mask], y[~mask], depth + 1)
            return node

        def _proba_one(self, x, node):
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            return node.proba

        def predict_proba(self, X):
            X = _to_array(X)
            return np.array([self._proba_one(x, self.root_) for x in X])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    class RandomForestClassifier:
        def __init__(self, n_estimators=100, random_state=None, max_depth=25, **_ignored):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.max_depth = max_depth

        def fit(self, X, y):
            X = _to_array(X)
            y = np.asarray(y, dtype=int)
            n, d = X.shape
            rng = np.random.RandomState(self.random_state)
            mf = max(1, int(np.sqrt(d)))
            self.trees_ = []
            for _ in range(self.n_estimators):
                boot_idx = rng.randint(0, n, size=n)
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth, max_features=mf,
                    random_state=int(rng.randint(0, 2**31 - 1)),
                )
                tree.fit(X[boot_idx], y[boot_idx])
                self.trees_.append(tree)
            self.n_features_ = d
            return self

        def predict_proba(self, X):
            X = _to_array(X)
            return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        @property
        def feature_importances_(self):
            imp = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
            s = imp.sum()
            return imp / s if s > 0 else imp
