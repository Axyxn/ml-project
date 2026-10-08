"""Fill the reference DOCX while preserving its original formatting and branding."""
import csv
import json
import re
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.cache' / 'report-tools'))
from lxml import etree as E

SOURCE = Path(r'C:\Users\aryan\Downloads\NewsLens ML.docx')
DEST = ROOT / 'reports' / 'Credit_Card_Fraud_Detection_Original_Format.docx'
page_path = ROOT / '.cache' / 'template_page_numbers.json'
page_numbers = json.loads(page_path.read_text(encoding='utf-8-sig')) if page_path.exists() else {}
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
W = '{' + NS['w'] + '}'
XML = '{http://www.w3.org/XML/1998/namespace}'
src = zipfile.ZipFile(SOURCE)
root = E.fromstring(src.read('word/document.xml'))
body = root.find('w:body', NS)
ps = body.findall('w:p', NS)
original_ps = [deepcopy(p) for p in ps]
tables = body.findall('w:tbl', NS)
parts = {}

def assign(nodes, text):
    """Change text alone, retaining every original run and its font properties."""
    if not nodes:
        return
    lengths = [len(n.text or '') for n in nodes]
    total = sum(lengths)
    cursor = 0
    for i, node in enumerate(nodes):
        end = len(text) if i == len(nodes)-1 else cursor + round(len(text)*lengths[i]/total) if total else cursor
        end = min(end, len(text))
        node.text = text[cursor:end]
        node.set(XML+'space', 'preserve')
        cursor = end

def set_p(p, text):
    # Text in diagrams is separate from the paragraph's own editable text.
    tokens = p.xpath('./w:r/w:t | ./w:r/w:tab | ./w:r/w:br | ./w:hyperlink/w:r/w:t', namespaces=NS)
    groups = [[]]
    for token in tokens:
        if token.tag == W+'t': groups[-1].append(token)
        elif token.tag in (W+'tab', W+'br'): groups.append([])
    segments = re.split('[\t\n]', text)
    if len(segments) == len(groups):
        for nodes, segment in zip(groups, segments): assign(nodes, segment)
    else:
        nodes = [n for g in groups for n in g]
        assign(nodes, text)
    if not any(groups) and text:
        run = E.SubElement(p, W+'r')
        prop = p.find('w:pPr/w:rPr', NS)
        if prop is not None: run.append(deepcopy(prop))
        t = E.SubElement(run, W+'t'); t.text = text; t.set(XML+'space','preserve')

def put(index, text): set_p(ps[index], text)

changes = {
0: '“Credit Card Fraud Detection”',
16: 'Prof. ____________________',
33: 'This is to certify that the mini-project entitled “Credit Card Fraud Detection”, (Group No. ________) of',
48: 'Prof. ____________________ Mini-Project Guide/Co-Guide',
63: 'This project report entitled “Credit Card Fraud Detection” by ____________________, ____________________, ____________________ and ____________________ is approved for the degree of Bachelors in Computer Science and Design, 2025-26.',
124: 'We would like to express our gratitude to our guide Prof. ____________________ and our Project Coordinator ____________________ for giving us the opportunity to undertake the Machine Learning mini-project “Credit Card Fraud Detection”. Their guidance helped us understand data preprocessing, imbalanced classification, model evaluation and practical deployment.',
125: 'We thank our Head of Department, Principal and institution for providing the support and resources needed for this project. We also acknowledge the ULB / Worldline dataset contributors and the open-source Python community for the data and tools used in the work. Finally, we thank our parents and friends for their support throughout the project.',
139: 'The Credit Card Fraud Detection project is a machine learning system for identifying potentially fraudulent transactions in the public ULB / Worldline dataset. Fraud is rare: the original 284,807 transactions contain only 492 frauds. After removing 1,081 exact duplicate records, the study uses 283,726 transactions, including 473 fraud cases. The cleaned data is divided into stratified 60% training, 20% validation and 20% test partitions with random seed 42.',
140: 'The project uses Python, pandas, scikit-learn and imbalanced-learn to compare Logistic Regression, Decision Tree and Random Forest, each with and without SMOTE, before and after bounded hyperparameter tuning. Median imputation, scaling and selection of the top 20 features by mutual information are fitted inside training pipelines. All sampling and learned transformations remain within training partitions or cross-validation folds to prevent leakage.',
141: 'Validation average precision selects the Random Forest baseline without SMOTE. Its operating threshold, 0.259462, is chosen by maximum validation F1 and frozen before test evaluation. On 56,746 test transactions, it detects 73 of 95 frauds, misses 22 and raises 10 false alerts. Precision is 0.8795, recall 0.7684, F1 0.8202, ROC-AUC 0.9538 and average precision 0.7792. These results describe the frozen benchmark split.',
142: 'The saved inference pipeline is integrated into the Streamlit application “Fraud Signal Studio” for single-transaction and CSV batch scoring. The system provides scores, review labels, visual summaries and downloadable results. Its PCA inputs are anonymized and its scores are uncalibrated. The project demonstrates a reproducible educational workflow rather than a validated financial decision service.',
188: '4.1\tImplementation Plan and Results',
202: '1                                                      Single transaction input\t',
203: '2                                                      Fraud example result\t',
204: '3                                                      Batch CSV scoring\t',
205: '4                                                      Invalid CSV message\t',
206: '5                                                      Class imbalance\t',
207: '6\t        Feature selection\t',
208: '7\t    Final threshold and confusion matrix\t',
209: '8\t Precision–recall curves\t',
210: '9\t   ROC curves\t',
217: 'Implementation Plan\t',
223: 'Credit-card transactions form a large and complex stream in which fraudulent activity is uncommon. A useful fraud detector must identify suspicious transactions while limiting false alerts that consume review effort and inconvenience legitimate customers. Machine learning can learn patterns from labelled transactions, but its evaluation must account for the rarity of fraud.',
224: 'In the selected public benchmark, fraud represents approximately 0.173% of raw transactions. A classifier that always predicts legitimate would achieve about 99.83% accuracy while detecting no fraud. This illustrates why accuracy alone is insufficient and why precision, recall, average precision and confusion counts are required.',
225: 'The Credit Card Fraud Detection system is designed to compare supervised classifiers under a controlled and reproducible workflow. It uses numerical Time, Amount and anonymized PCA inputs to predict Class 0 (legitimate) or Class 1 (fraud). Predictions are screening signals intended to support further review.',
226: 'The project is implemented in Python 3.12 using pandas and NumPy for data handling, scikit-learn for classification and evaluation, imbalanced-learn for SMOTE pipelines, matplotlib and seaborn for analysis figures, and joblib for model serialization. A Streamlit application provides a practical interface to the saved model.',
227: 'Three algorithms are examined: Logistic Regression, Decision Tree and Random Forest. Each is fitted with and without SMOTE, producing six baseline configurations. All six are also tuned using three candidate parameter combinations and three stratified folds on a bounded training sample. The independent validation partition selects the final model and decision threshold.',
228: 'The main objectives are correct handling of imbalance, prevention of data leakage, meaningful model comparison, transparent reporting and a consistent inference path. The report documents the dataset audit, preprocessing, feature selection, model selection, test results, demonstration and limitations of the completed project.',
232: 'The significance of this study lies in showing how a reproducible machine learning workflow can identify rare fraud patterns while exposing its errors. The focus is on measured performance under severe class imbalance rather than relying on a high overall accuracy figure.',
234: 'Precision indicates the reliability of fraud alerts, while recall indicates the proportion of actual fraud detected. False negatives represent missed fraud; false positives represent legitimate transactions sent for review. Reporting these separately makes the practical consequences of a classifier easier to assess.',
236: 'The Streamlit interface improves accessibility by allowing a user to enter compatible dataset values or upload a transaction CSV. It displays the saved review threshold, model score and results, and offers CSV downloads. Ordinary card details cannot substitute for V1–V28 because the original PCA transformation is unavailable.',
238: 'Methodologically, the project compares linear and nonlinear models with a common pipeline. Scaling, feature selection and optional resampling are confined to training data. The independent validation set chooses the model and threshold, and the untouched test set measures the final frozen operating rule.',
240: 'The experiment also demonstrates that SMOTE and tuning must be assessed empirically. In the saved validation results, SMOTE improves recall for some classifiers but increases false alerts or reduces average precision. A baseline remains eligible when a tuned configuration performs worse on the independent validation partition.',
242: 'The project provides an educational foundation for future work on temporal evaluation, probability calibration, drift monitoring and thresholds based on review capacity. Its results are limited to a short historical benchmark and should be interpreted with those limitations.',
250: 'Existing fraud detection approaches include fixed rules, manual investigation and supervised statistical models. A rule based only on transaction amount is easy to implement but can miss fraud at small values. The project’s training-only analysis shows overlap between legitimate and fraudulent amounts, supporting the use of multiple predictors.',
251: 'Logistic Regression offers an efficient regularized linear baseline. A Decision Tree can represent nonlinear rules but may overfit rare fraud samples. A Random Forest averages randomized trees to reduce variance, at the cost of more computation and less direct interpretability. These algorithms provide complementary comparisons for the same dataset.',
252: 'SMOTE, introduced by Chawla and colleagues, generates synthetic minority examples by interpolation between nearby minority points [3]. It may improve minority coverage, but it can also produce unrealistic or overlapping samples. Resampling the complete dataset before splitting would leak information; the project avoids this by using fold-local pipelines [4].',
253: 'The public ULB / Worldline benchmark and TensorFlow’s imbalanced-data tutorial provide the dataset context [1, 2]. Average precision is used for ranking because it summarizes the precision–recall relationship under imbalance [5]. Trapezoidal PR-AUC is reported separately because its interpolation differs from average precision.',
257: 'The problem is to classify a transaction using 30 numerical predictors: Time, Amount and V1–V28. The target is Class, where 0 is legitimate and 1 is fraud. The principal difficulty is that only 473 distinct fraud cases remain after exact deduplication, among 283,726 distinct transactions.',
259: 'Reliable evaluation requires disjoint partitions, natural fraud prevalence in validation and test, and training-only fitting of all transformations. Exact copies must not cross partition boundaries. Model and threshold selection must be completed without inspecting test performance.',
261: 'The proposed system addresses these requirements with a stratified split, median imputation, feature scaling, mutual-information selection, optional SMOTE and controlled model comparison. The final model and threshold are stored with schema, version and dataset metadata so that the application uses the same inference pipeline.',
267: 'The main objective of the Credit Card Fraud Detection project is to build and document a complete, reproducible classification workflow and a working demonstration for rare fraud detection using the real ULB / Worldline transactions.',
269: 'To audit the dataset, remove exact duplicates, check label conflicts and preserve real fraud examples and extreme values.',
270: 'To compare Logistic Regression, Decision Tree and Random Forest with and without SMOTE before and after hyperparameter tuning, using leakage-safe training pipelines.',
271: 'To select the model by validation average precision, freeze a validation-F1 threshold, report held-out test metrics, and deploy the saved pipeline in a single and batch Streamlit interface.',
273: 'The scope covers an educational public-data benchmark with 30 numerical predictors. It includes dataset preparation, exploratory analysis, feature selection, imbalance handling, six baseline and six tuned experiments, validation selection, final test evaluation and application demonstration.',
275: 'Data Preparation: load and validate the real CSV, remove 1,081 duplicate rows, and make stratified 60/20/20 partitions with seed 42.',
276: 'Preprocessing and Feature Selection: fit median imputation, RobustScaler for Time and Amount, StandardScaler for PCA components, and the top-20 mutual-information selector within training folds.',
277: 'Training and Saved Artifacts: evaluate three classifiers with and without SMOTE, perform bounded AP-based tuning, and serialize the selected complete joblib pipeline and metadata.',
278: 'Interactive Demonstration: score compatible single transactions or CSV batches, show model scores and alerts, validate inputs, and download scored results. Raw card-system integration is outside the implemented scope.',
289: 'The design methodology follows a reproducible sequence from problem definition and data audit to model comparison, frozen evaluation and application integration. Reusable modules separate data loading, preprocessing, training, metrics, reporting and inference, while the notebook presents eight ordered project components.',
292: 'Defined a binary classification problem with rare fraud and a practical tradeoff between missed fraud and false alerts.',
293: 'Specified 30 numerical inputs, Class as the target, meaningful imbalance metrics, compatible single and CSV batch scoring, and clear input errors.',
294: 'Established training-only preprocessing, validation-based selection, a frozen test evaluation and uncalibrated-score wording.',
296: 'Loaded the original ULB / Worldline CSV: 284,807 transactions and 492 frauds. The downloader records source information and SHA-256 for reproducibility.',
297: 'Audited missing values (none in the original data), class frequencies, Time/Amount distributions and training-only correlations. Strongest target correlations include V17, V14, V12 and V10.',
298: 'Removed 1,081 exact duplicate rows, leaving 283,726 records and 473 frauds. The stratified partitions have 170,235 training, 56,745 validation and 56,746 test records.',
299: 'Preprocessing & Feature Selection:',
300: 'Fitted median imputation, RobustScaler on Time and Amount, and StandardScaler on V1–V28 inside each training pipeline. Outliers remain because they can be genuine fraud signals.',
301: 'Estimated continuous mutual information on up to 60,000 stratified training rows and retained the predetermined top 20 predictors. Each cross-validation training fold relearns this selection.',
302: 'Selected in the saved full-training fit: Time, V1–V12, V14, V16–V19, V21 and V27. Amount and nine PCA components were not selected; all 30 inputs are still required by the raw pipeline schema.',
303: 'Compared each classifier with and without SMOTE at a minority-to-majority ratio of 0.10, applied after scaling and selection during training only; inference skips sampling.',
305: 'Python 3.12: programming language for data handling, training, evaluation and the demonstration interface.',
306: 'scikit-learn and imbalanced-learn: classifiers, feature selection, stratified searches, evaluation and SMOTE pipelines.',
307: 'Streamlit: single-transaction and CSV batch interface; joblib: complete inference pipeline and supporting artifacts.',
308: 'pandas, NumPy, matplotlib, seaborn and Jupyter: analysis, plots, tabular reports and the executed presentation notebook.',
310: 'Benchmarked Logistic Regression, Decision Tree and Random Forest with and without SMOTE, producing six baselines and six tuned configurations.',
311: 'Performed three-candidate, three-fold stratified RandomizedSearchCV using average precision on up to 60,000 natural-prevalence training rows; refitted each best candidate on all training rows.',
312: 'Chose Random Forest without SMOTE at baseline parameters by validation AP 0.8659. Froze the validation-F1 threshold at 0.259462 before test scoring. Saved the pipeline, scaler, feature list and metadata.',
313: 'Application Development:',
314: 'Implemented the Streamlit app “Fraud Signal Studio” with a single-transaction tab, example selector, numeric input form, and a batch CSV tab with templates and downloads.',
315: 'Loaded the complete saved pipeline and validated feature names, scaler consistency, selected features, library version and operating threshold. Raw inputs pass through preprocessing exactly once.',
316: 'Rejected empty input, missing or duplicate columns, nonnumeric values, missing or infinite values, and negative Time/Amount. Optional Class is ignored during inference.',
318: 'On 56,746 test transactions, the selected threshold produces TN=56,641, FP=10, FN=22 and TP=73. Precision=0.8795, recall=0.7684, F1=0.8202, ROC-AUC=0.9538, AP=0.7792 and PR-AUC=0.7791.',
319: 'Verified 23 automated tests, a 28-cell notebook with zero errors, and local health/page HTTP 200. Actual browser checks exercised a fraud example, valid CSV upload and an invalid CSV message.',
323: 'UI Design - Single Transaction Input',
326: 'Fraud Example Result',
330: 'Batch CSV Scoring',
333: 'Invalid CSV Message',
336: 'Training Class Imbalance',
339: 'Mutual-Information Feature Selection',
342: 'Final Threshold and Confusion Matrix',
345: 'Test Precision–Recall Curves',
348: 'Test ROC Curves',
376: 'Frontend and application: Streamlit 1.50.0; numerical forms, visual summaries, CSV upload and result downloads.',
377: 'Runtime: Python 3.12; the saved experiment records Python 3.12.5. No dedicated GPU is required by these classifiers.',
378: 'ML and analysis: scikit-learn 1.7.2, imbalanced-learn 0.14.0, pandas 2.3.3, NumPy 2.2.6, SciPy 1.16.2, matplotlib 3.10.7 and seaborn 0.13.2.',
379: 'Storage and testing: joblib 1.5.2, JSON metadata, CSV data/results, Jupyter notebook, pytest 8.4.2 and requests 2.32.5.',
380: 'Tooling: project virtual environment, Git, terminal/PowerShell and pinned requirements.txt. Reserve space for the approximately 151 MB raw CSV and generated artifacts.',
401: 'The Credit Card Fraud Detection project demonstrates a complete machine learning workflow for an extremely imbalanced benchmark. It audits and deduplicates the real data, fits preprocessing and optional SMOTE within training pipelines, compares three algorithms before and after tuning, freezes selection using validation, and evaluates the final rule on untouched test data.',
402: '▪ The selected baseline Random Forest without SMOTE detects 73 of 95 test frauds with 10 false alerts at threshold 0.259462. Its precision is 0.8795, recall 0.7684 and F1 0.8202. The working Streamlit app uses the saved pipeline for single and CSV batch predictions, supported by tests and an executed notebook.',
403: '▪ The study is limited by anonymized PCA inputs, a two-day 2013 dataset, only 95 test frauds, random rather than temporal evaluation, a small parameter search and uncalibrated scores. These constraints limit generalization and explain why the system is presented as an educational demonstration.',
408: 'Future work should improve the evaluation protocol, probability interpretation and operational decision process before any wider use. The following directions build on the measured strengths and limitations of the current project:',
409: 'Temporal Evaluation: train on earlier transactions and evaluate on a later period. If identifiers become available, separate cardholders across partitions to assess generalization more realistically.',
410: 'Imbalance Comparisons: evaluate class weighting and balanced forests alongside SMOTE. Measure alert workload and missed fraud, rather than assuming resampling is beneficial.',
411: 'Search and Uncertainty: expand the tuning budget, use larger or nested cross-validation, examine feature-count ablations and quantify variation caused by the small number of fraud cases.',
412: 'Calibration and Thresholds: calibrate scores on independent data and choose an operating threshold from actual review capacity, losses and customer-impact costs.',
413: 'Interpretation and Input Integration: obtain a compatible upstream transformation before accepting raw card-system data, and study explanations that respect the anonymized feature limitations.',
414: 'Monitoring and Human Review: track transaction and score drift, retrain on more recent data, and include analysts in validating flagged transactions and collecting feedback.',
419: '[1] ULB / Worldline. Credit Card Fraud Detection dataset, Kaggle.\nhttps://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
420: '[2] TensorFlow. Classification on imbalanced data.\nhttps://www.tensorflow.org/tutorials/structured_data/imbalanced_data',
421: '[3] Chawla, Bowyer, Hall and Kegelmeyer. SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16, 321–357 (2002).\nhttps://doi.org/10.1613/jair.953',
422: '[4] imbalanced-learn. Common pitfalls and recommended practices.\nhttps://imbalanced-learn.org/stable/common_pitfalls.html',
423: '[5] scikit-learn. average_precision_score documentation.\nhttps://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html',
}
for index, text in changes.items(): put(index, text)

# Student fields appear as both lists and tables in the supplied document.
for index in [7,8,9,10,37,38,39,40]: put(index, '________________________\t______________')
for t in tables[:2]:
    for row in t.findall('w:tr',NS):
        for cell in row.findall('w:tc',NS):
            cell_ps = cell.findall('w:p',NS)
            for j,p in enumerate(cell_ps): set_p(p, '____________________' if j==0 else '')

activities = [
    'Problem definition, requirements and reproducible environment setup.',
    'Dataset download, provenance checks, validation and duplicate audit.',
    'Stratified split and training-only exploratory data analysis.',
    'Training-only imputation, scaling and mutual-information selection.',
    'Six baseline fits: three algorithms with and without SMOTE.',
    'Bounded stratified hyperparameter tuning and full-training refits.',
    'Validation model/threshold selection and frozen test evaluation.',
    'Saved artifacts, executed notebook and Streamlit single/batch interface.',
    'Automated checks, real browser screenshots and final documentation.'
]
for row,text in zip(tables[2].findall('w:tr',NS)[1:],activities):
    cell=row.findall('w:tc',NS)[1]
    for j,p in enumerate(cell.findall('w:p',NS)): set_p(p,text if j==0 else '')

# Retain the existing flowchart shapes, arrows, locations and fonts.
flow = {'Start':'Start','Admin Login':'Load Dataset','Authentication':'Check Schema',
        'Create Elections':'Deduplicate','Add Candidates':'Split Data','Start Elections':'Train Models',
        'User Votes':'Validate Scores','Validation':'Check Inputs','Save votes + Update Count':'Select Model + Threshold',
        'Abort and show message':'Correct Input','Valid':'Valid','Invalid':'Invalid',
        'Admin ends election':'Freeze Choice','Calculate votes':'Test Metrics','Publish result':'Save Pipeline',
        'Send Email Notifications':'Score New Inputs','End':'End'}
for textbox in ps[286].xpath('.//w:txbxContent',namespaces=NS):
    label=''.join(textbox.xpath('.//w:t/text()',namespaces=NS))
    assign(textbox.xpath('.//w:t',namespaces=NS),flow[label])

# Replace project screenshots/plots only. Original logo/stamp media stay byte-identical.
assets = [ROOT/'reports/assets/app_single_input.png',ROOT/'reports/assets/app_fraud_result.png',
          ROOT/'reports/assets/app_batch_results.png',ROOT/'reports/assets/app_invalid_csv.png',
          ROOT/'figures/class_imbalance.png',ROOT/'figures/feature_selection.png',
          ROOT/'figures/final_threshold_and_confusion.png',ROOT/'figures/test_pr_curves.png',
          ROOT/'figures/test_roc_curves.png']
for i,path in enumerate(assets,3): parts[f'word/media/image{i}.png']=path.read_bytes()

# Fill the implementation section with project evidence using template paragraph/table styles.
def clone_p(text, sample=223):
    p=deepcopy(original_ps[sample])
    for el in p.xpath('.//w:bookmarkStart | .//w:bookmarkEnd',namespaces=NS): el.getparent().remove(el)
    set_p(p,text)
    return p

def clone_table(headers, rows):
    t=deepcopy(tables[2]); tr=t.findall('w:tr',NS)
    header=deepcopy(tr[0]); row_sample=deepcopy(tr[1])
    for r in tr: t.remove(r)
    def row_text(row,values):
        for cell,value in zip(row.findall('w:tc',NS),values):
            for j,p in enumerate(cell.findall('w:p',NS)): set_p(p,str(value) if j==0 else '')
    row_text(header,headers);t.append(header)
    for values in rows:
        row=deepcopy(row_sample);row_text(row,values);t.append(row)
    return t

anchor=ps[393]
extra=[clone_p('4.2 Dataset and Reproducibility',230),
clone_p('The dataset SHA-256 is 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89. The cleaned split contains 170,235 training rows (284 frauds), 56,745 validation rows (94 frauds) and 56,746 test rows (95 frauds). Seed 42 and the pinned dependencies reproduce the reported setup. The feature selector and search use at most 60,000 natural-prevalence training rows; each search fold contains about 33–34 validation frauds.'),
clone_p('4.3 Model Comparison',230),
clone_p('The following validation results use the common threshold of 0.5. No resampling is denoted by None. AP is average precision; P, R and F1 are precision, recall and F1. The highest validation AP selects Random Forest without SMOTE at baseline parameters. The test set did not guide this choice.')]
comparison=list(csv.DictReader((ROOT/'results/model_comparison.csv').open(encoding='utf8')))
rows=[]
for r in comparison:
    if r['partition']=='validation' and r['stage'] in ['baseline','tuned']:
        rows.append((f"{r['algorithm']} | {r['imbalance']} | {r['stage']}",
                     'AP={:.4f}; P={:.4f}; R={:.4f}; F1={:.4f}'.format(*[float(r[k]) for k in ['average_precision','precision','recall','f1']])))
extra.append(clone_table(['Configuration','Validation performance'],rows))
extra += [clone_p('The untuned Random Forest without SMOTE has validation AP 0.8659, compared with 0.8579 for its tuned version. SMOTE improves baseline Logistic Regression recall from 0.660 to 0.894 but decreases precision from 0.899 to 0.418 and increases false alerts from 7 to 117. Baseline Random Forest SMOTE raises recall from 0.830 to 0.851 but lowers AP from 0.8659 to 0.8326. Improvements must therefore be assessed against alert workload.'),
clone_p('4.4 Selected Model and Held-out Results',230),
clone_p('The saved classifier is the baseline Random Forest with 80 trees, maximum depth 14, minimum leaf size 2 and max_samples 0.7. It remains fitted on training only. Validation F1 selects the operating threshold 0.2594623, frozen before evaluating test data. The held-out results below refer to this operating threshold.')]
m=json.loads((ROOT/'models/metadata.json').read_text(encoding='utf8'))['final_test_metrics']
extra.append(clone_table(['Test measure','Value'],[(label,f'{m[key]:.4f}') for label,key in [('Accuracy','accuracy'),('Precision','precision'),('Recall','recall'),('F1','f1'),('ROC-AUC','roc_auc'),('Average precision','average_precision'),('Trapezoidal PR-AUC','pr_auc')]]+[('TN / FP / FN / TP','56,641 / 10 / 22 / 73')]))
extra += [clone_p('At threshold 0.5, the same model catches 69 frauds, misses 26 and raises 3 false alerts. The frozen threshold catches four more frauds while adding seven false alerts; its test F1 falls slightly from 0.8263 to 0.8202. This test outcome was not used to revise the threshold. AP and ROC-AUC remain unchanged because the score ranking is unchanged.'),
clone_p('4.5 Saved Files, Verification and Demonstration',230),
clone_p('models/best_model.joblib is the complete inference pipeline. scaler.joblib, preprocessor.joblib, selected_features.joblib and metadata.json support consistency checks. results/ contains the audit, splits, comparisons, tuning parameters, frozen selection, findings and real example transactions. figures/ contains analysis and evaluation plots. src/ supplies reusable modules; tests/ covers data, pipelines, metrics, artifacts, inference and the app.'),
clone_p('The automated suite passed 23 tests on 8 October 2026. Saved notebook execution records 28 cells and zero errors; server verification records HTTP 200 for health and page startup. Browser checks for the report exercised a real fraud example, actual CSV upload and a readable invalid-input error. Synthetic fixtures are used only in tests, not in the reported experiment.'),
clone_p('Run from the project root: python -m pip install -r requirements.txt; python scripts/download_data.py; python scripts/train.py; python -m pytest -q; python scripts/execute_notebook.py; python -m streamlit run app.py. The app normally serves at http://localhost:8501. Use a real example for single scoring or upload a CSV containing Time, V1–V28 and Amount. Any Class column is ignored during prediction.'),
clone_p('The scores are uncalibrated screening signals. The PCA transformation is unavailable, the dataset spans only two days in 2013, the random split is weaker than future-time validation, and there are only 95 test frauds. The small search and fixed top-20 feature count are classroom choices; mutual information can miss interactions and SMOTE may introduce unrealistic points. Bank-specific loss and review costs were not available and were not invented.')]
for node in extra: anchor.addprevious(node)

# Keep the existing contents/list layouts and add page fields in their original typography.
abbrev=deepcopy(original_ps[215]);set_p(abbrev,'List of Abbreviation')
ps[218].addprevious(abbrev)
ps[218].addprevious(clone_p('AP: Average Precision; PR-AUC: Precision–Recall Area Under the Curve; ROC-AUC: Receiver Operating Characteristic Area Under the Curve; PCA: Principal Component Analysis; SMOTE: Synthetic Minority Over-sampling Technique; CV: Cross-validation; TN/FP/FN/TP: True Negative / False Positive / False Negative / True Positive.'))
annex=deepcopy(original_ps[397]);set_p(annex,'Annexure A - Weekly Progress Report')
ps[424].addprevious(annex)
ps[424].addprevious(deepcopy(tables[2]))
bookmark_id=1000
def bookmark(target,name):
    global bookmark_id
    a=E.Element(W+'bookmarkStart');a.set(W+'id',str(bookmark_id));a.set(W+'name',name)
    b=E.Element(W+'bookmarkEnd');b.set(W+'id',str(bookmark_id))
    target.insert(1 if target.find('w:pPr',NS) is not None else 0,a);target.append(b)
    bookmark_id+=1
def page_field(p,name):
    # Remove the old manual page number without changing its runs or tabs.
    tokens=p.xpath('./w:r/w:t | ./w:r/w:tab',namespaces=NS)
    tabs=[i for i,t in enumerate(tokens) if t.tag==W+'tab']
    if tabs:
        for t in tokens[tabs[-1]+1:]:
            if t.tag==W+'t':t.text=''
    else:
        r=E.SubElement(p,W+'r');E.SubElement(r,W+'tab')
    f=E.SubElement(p,W+'fldSimple');f.set(W+'instr',f' PAGEREF {name} ')
    r=E.SubElement(f,W+'r')
    props=p.xpath('./w:r/w:rPr',namespaces=NS)
    if props:r.append(deepcopy(props[-1]))
    t=E.SubElement(r,W+'t');t.text=str(page_numbers.get(name,' '))
refs={164:95,165:120,166:137,167:161,168:195,169:215,170:abbrev,
      171:218,173:221,174:230,175:246,177:248,178:255,179:264,
      180:286,182:283,183:288,184:323,185:373,186:386,188:385,189:397,191:396,192:406,193:annex,
      202:323,203:326,204:330,205:333,206:336,207:339,208:342,209:345,210:348,217:385}
put(192,'Future Scope')
for index,target in refs.items():
    name=f'ProjectRef{index}';bookmark(ps[target] if isinstance(target,int) else target,name);page_field(ps[index],name)

# Correct reference hyperlinks in their existing containers and typography.
relationships=E.fromstring(src.read('word/_rels/document.xml.rels'))
links={'rId19':'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
       'rId20':'https://www.tensorflow.org/tutorials/structured_data/imbalanced_data',
       'rId21':'https://doi.org/10.1613/jair.953',
       'rId22':'https://imbalanced-learn.org/stable/common_pitfalls.html',
       'rId23':'https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html'}
for rel in relationships:
    if rel.get('Id') in links: rel.set('Target',links[rel.get('Id')])
parts['word/_rels/document.xml.rels']=E.tostring(relationships,xml_declaration=True,encoding='UTF-8',standalone=True)
core=E.fromstring(src.read('docProps/core.xml'))
for tag in ['{http://purl.org/dc/elements/1.1/}creator','{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy']:
    element=core.find(tag)
    if element is not None:element.text=''
parts['docProps/core.xml']=E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
if page_numbers.get('TotalPages'):
    app=E.fromstring(src.read('docProps/app.xml'))
    app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages').text=str(page_numbers['TotalPages'])
    parts['docProps/app.xml']=E.tostring(app,xml_declaration=True,encoding='UTF-8',standalone=True)

# A preservation record checks original paragraph formatting, run fonts, stamps and package resources.
def structural(el):
    return (el.tag,tuple(sorted(el.attrib.items())),el.text,tuple(structural(c) for c in el))
def properties(p):
    return [structural(x) for x in p.xpath('./w:pPr | ./w:r/w:rPr | ./w:hyperlink/w:r/w:rPr',namespaces=NS)]
changed_props=[i for i,(old,new) in enumerate(zip(original_ps,ps)) if properties(old)!=properties(new)]
assert not changed_props, f'Original font/paragraph properties changed: {changed_props}'
for index in [20,25,55,90,115]:
    assert structural(ps[index])==structural(original_ps[index]), 'College branding paragraph changed'
alltext=''.join(root.xpath('.//w:t/text()',namespaces=NS))
for oldterm in ['NewsLens','Intra-college voting','Admin Login','User Votes','Kishna','Chaudhari','Joshi','Kakade','12317001','12247030']:
    assert oldterm not in alltext, oldterm
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED) as out:
    for info in src.infolist(): out.writestr(deepcopy(info),parts.get(info.filename,src.read(info.filename)))
with zipfile.ZipFile(DEST) as out:
    for name in ['word/styles.xml','word/fontTable.xml','word/numbering.xml','word/theme/theme1.xml','word/footer1.xml','word/media/image1.png','word/media/image2.png']:
        assert src.read(name)==out.read(name),name
    original_root=E.fromstring(src.read('word/document.xml'))
    assert structural(original_root.find('w:body/w:sectPr',NS))==structural(root.find('w:body/w:sectPr',NS)), 'Page size or margins changed'
    assert out.testzip() is None, 'Invalid ZIP package'
    assert all(term in alltext for term in ['New Horizon Institute of Technology and Management','University of Mumbai'])
record={'source':str(SOURCE),'output':str(DEST),'original_paragraph_formats_and_fonts_preserved':425,
        'college_logo_and_stamp_media_byte_identical':True,'styles_fonts_theme_numbering_footer_byte_identical':True,
        'student_details_blank':True,'project_images_replaced':9,'added_project_evidence_nodes':len(extra),
        'measured_page_count':page_numbers.get('TotalPages'),'page_references_populated':len(page_numbers)-1 if page_numbers else 0}
record['page_size_and_margins_preserved']=True
(ROOT/'reports/template_preservation.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record,indent=2))
