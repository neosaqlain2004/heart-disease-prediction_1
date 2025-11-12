# TODO List for Cardiovascular Disease Prediction Project

## Completed Steps
- [x] Create directories: /model, /templates, /static, /notebooks, /tests
- [x] Create requirements.txt with pinned versions
- [x] Create train.py: Full training pipeline with data loading, preprocessing, feature selection, model training/tuning, evaluation, and saving
- [x] Create model_utils.py: Helper functions for lazy loading, preprocessing, and prediction
- [x] Create app.py: Flask web app with GET / for UI, POST /predict for predictions, logging, and API key support
- [x] Create templates/index.html: Responsive form for features, prediction display, and required watermark/note
- [x] Create static/styles.css: Simple CSS for styling
- [x] Create static/script.js: JavaScript for form submission and displaying results
- [x] Create notebooks/EDA.ipynb: Basic exploratory data analysis notebook
- [x] Create tests/test_predict.py: Simple test for prediction function
- [x] Create README.md: Documentation with setup, usage, deployment instructions
- [x] Create Procfile: For Heroku-like deployment
- [x] Initiate training: Run `python train.py --sample-size 10000` (currently running)

## Pending Steps
- [ ] Wait for training to complete and verify models are saved to /model/
- [ ] Test the web app: Start server with `python app.py` and check UI at http://localhost:5000
- [ ] Test API endpoint: POST to /predict with sample data and verify response
- [ ] Run tests: Execute `python tests/test_predict.py` to ensure prediction functions work
- [ ] Verify deployment readiness: Check Procfile and ensure no errors in logs
- [ ] Final check: Ensure all files are in place and project is runnable

## Notes
- Training may take time due to hyperparameter tuning (RandomizedSearchCV with n_iter=50).
- If training fails, check logs and fix issues (e.g., memory, dependencies).
- After training, models should be in /model/ with extensions .pkl for sklearn models and .h5 for ANN.
- Lazy loading in model_utils.py ensures memory efficiency.
