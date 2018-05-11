from utils.dataset import Dataset
from utils.model import Model
from utils.scorer import report_score
import csv


# Train a model and evaluate it. Evaluation can be either on development or test data.
# The model can be the baseline or the final model.
def execute_demo(language, is_baseline = True, use_test = False):
    
    data = Dataset(language)
    model = Model(language, is_baseline)
    model.train(data.trainset)
    
    if is_baseline:
        mod = "baseline"
    else:
        mod = "final"
    if use_test:
        print("Evaluating {} model on {} using Test set".format(mod, language))
        predictions = model.test(data.testset)
        gold_labels = [sent['gold_label'] for sent in data.testset]
        print("{} instances of training data, {} instances of evaluation data".format(len(data.trainset), len(data.testset)))
    else:
        print("Evaluating {} model on {} using Development set".format(mod, language))
        predictions = model.test(data.devset)
        gold_labels = [sent['gold_label'] for sent in data.devset]
        print("{} instances of training data, {} instances of evaluation data".format(len(data.trainset), len(data.devset)))
    
    report_score(gold_labels, predictions, detailed = False)

# Print number of errors and create file with wrongly classified words,  
# predicted and correct labels
def analyse_errors(data, gold_labels, predictions, language):
     words = [sent['target_word'] for sent in data.devset]
     errors = []
     for i, sent in enumerate(data.devset):
         if predictions[i] != gold_labels[i]:
             errors.append((words[i], predictions[i], gold_labels[i] ))
    
     with open('{}errors.csv.'.format(language), 'w', encoding = 'utf-8') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['word', 'predicted', 'correct'])   
        for row in errors:
            csv_out.writerow(row)
     print("number of errors", len(errors))   

if __name__ == '__main__':
    execute_demo('english', True, True)
    execute_demo('spanish', True, True)
    execute_demo('english', False, True)
    execute_demo('spanish', False, True)

