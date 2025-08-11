from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import os

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'CSV dosyası yüklenmeli'}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        target_col = request.form.get('target') or df.columns[-1]
        selected_model = request.form.get('model', 'decision_tree')
        binary_classification = request.form.get('binary_classification', 'false').lower() == 'true'

        if target_col not in df.columns:
            return jsonify({'error': f'Hedef sütun ({target_col}) bulunamadı.'}), 400

        y = df[target_col]
        X = df.drop(columns=[target_col])

        if binary_classification and y.nunique() > 2:
            y = y.apply(lambda val: 1 if val == y.max() else 0)

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns

        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            X[num_cols] = pd.DataFrame(num_imputer.fit_transform(X[num_cols]), columns=num_cols)

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols)

        X = pd.get_dummies(X, columns=cat_cols)

        stratify_arg = y if binary_classification else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify_arg
        )

        model = None

        if selected_model == 'logistic':
            try:
                C = float(request.form.get('C', 1.0))
            except ValueError:
                return jsonify({'error': 'C parametresi sayı olmalıdır.'}), 400
            penalty = request.form.get('penalty', 'l2')
            solver = request.form.get('solver', 'lbfgs')

            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                return jsonify({'error': 'L1 penalty için solver liblinear veya saga olmalıdır.'}), 400

            model = LogisticRegression(max_iter=1000, class_weight='balanced', C=C, penalty=penalty, solver=solver)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif selected_model == 'svm':
            try:
                C = float(request.form.get('C', 1.0))
            except ValueError:
                return jsonify({'error': 'C parametresi sayı olmalıdır.'}), 400

            kernel = request.form.get('kernel', 'rbf')

            gamma_val = request.form.get('gamma', 'scale')
            if gamma_val not in ['scale', 'auto']:
                try:
                    gamma_val = float(gamma_val)
                except ValueError:
                    return jsonify({'error': 'Gamma parametresi "scale", "auto" veya float sayı olmalıdır.'}), 400

            model = SVC(class_weight='balanced', C=C, kernel=kernel, gamma=gamma_val)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif selected_model == 'random_forest':
            try:
                n_estimators = int(request.form.get('n_estimators', 100))
            except ValueError:
                return jsonify({'error': 'n_estimators parametresi tam sayı olmalıdır.'}), 400

            max_depth_val = request.form.get('max_depth')
            if max_depth_val:
                try:
                    max_depth = int(max_depth_val)
                except ValueError:
                    return jsonify({'error': 'max_depth parametresi tam sayı veya boş olmalıdır.'}), 400
            else:
                max_depth = None

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        elif selected_model == 'knn':
            try:
                n_neighbors = int(request.form.get('n_neighbors', 5))
            except ValueError:
                return jsonify({'error': 'n_neighbors parametresi tam sayı olmalıdır.'}), 400

            weights = request.form.get('weights', 'uniform')
            if weights not in ['uniform', 'distance']:
                return jsonify({'error': 'weights parametresi "uniform" veya "distance" olmalıdır.'}), 400

            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif selected_model == 'nb':
            try:
                var_smoothing = float(request.form.get('var_smoothing', 1e-9))
            except ValueError:
                return jsonify({'error': 'var_smoothing parametresi sayı olmalıdır.'}), 400

            model = GaussianNB(var_smoothing=var_smoothing)

        elif selected_model == 'xgboost':
            if xgboost_available:
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            else:
                return jsonify({'error': 'XGBoost yüklü değil. `pip install xgboost` ile yükleyin.'}), 400

        elif selected_model == 'decision_tree':
            model = DecisionTreeClassifier()

        else:
            return jsonify({'error': f'Geçersiz model türü: {selected_model}'}), 400

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        num_classes = y.nunique()
        average_type = 'macro' if num_classes > 2 else 'binary'

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average_type, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)

        return jsonify({
            'accuracy': round(acc, 4),
            'confusion_matrix': cm.tolist(),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'used_target': target_col,
            'features_used': X.columns.tolist(),
            'model_used': selected_model
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

