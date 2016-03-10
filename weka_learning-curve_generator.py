"""
Instance learning curve generator built using py-weka
"""
import csv
import random
import sys
import pdb
import traceback

import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
from weka.classifiers import Classifier, Evaluation

def build_and_classify(classifier, classifier_name, approach_name, infile, percentage='10'):
    """
    Creates model and classifies against input data. Returns accuracy statistics
    """
    # set seed so results are consistent
    random.seed('iot')

    # load data
    loader = Loader(classname='weka.core.converters.CSVLoader')
    data = loader.load_file(infile)
    data.class_is_last()

    # convert all numeric attributes to nominal
    to_nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
                        options=['-R', 'first-last'])
    to_nominal.inputformat(data)
    data = to_nominal.filter(data)

    # randomize data with constant seed
    randomize = Filter(classname='weka.filters.unsupervised.instance.Randomize',
                       options=['-S', '42'])
    randomize.inputformat(data)

    data = randomize.filter(data)

    # create training set and testing set
    train_percent_filter = Filter(classname='weka.filters.unsupervised.instance.RemovePercentage',
                                  options=['-P', percentage, '-V'])
    train_percent_filter.inputformat(data)

    train = train_percent_filter.filter(data)
    test = data

    # build and test classifier
    classifier.build_classifier(train)
    evaluation = Evaluation(train)
    evaluation.test_model(classifier, test)

    # return results as array
    results = [
        approach_name,
        classifier_name,
        percentage,
        evaluation.percent_correct,
        evaluation.weighted_f_measure
    ]
    return results

def learning_curve(classifier, classifier_name, approach_name, infile, percentages=None):
    """
    Creates learning curve by building classifier using multiple percent blocks of data.
    Returns array of curve values.
    """
    # check if no percentages were sent it, default to every 5%
    if percentages is None:
        percentages = range(5, 101, 5)

    # create percentages to map classifier on
    percentage_array = [str(x) for x in percentages]

    # create output
    curve_output = []

    # train and test
    # use each percentage i of the data set as training (whole dataset as testing)
    for i in percentage_array:
        curve_output.append(build_and_classify(classifier, classifier_name, approach_name,
                                               infile, percentage=i))
    return curve_output

def multi_file_curve(classifier, classifier_name, name_list, in_file_list, percentages=None):
    """
    Runs learning_curve on list of files.
    """
    # default percentages to 5% intervals if none
    if percentages is None:
        percentages = range(5, 101, 5)

    if len(name_list) != len(in_file_list):
        raise Exception('name_list and in_file_list must be of the same size')
    output = []
    file_count = len(in_file_list)
    files_remaining = file_count

    print '\nBeginning ' + classifier_name + '. ' + str(files_remaining) + ' files remaining...'

    for i in range(file_count):
        output.extend(learning_curve(classifier, classifier_name,
                                     classifier_name + '_' + name_list[i],
                                     in_file_list[i], percentages))
        files_remaining -= 1
        print classifier_name + ': Finished file ' + in_file_list[i] + '. ' + \
              str(files_remaining) + ' files remaining.'
    print classifier_name + ' completed.'
    return output

def main():
    """
    Specify list of files to multi_file_curve, classify, and export results as csv.
    """
    try:
        # start up a JVM to run weka on
        jvm.start(max_heap_size='512m')

        # classifiers
        naive_bayes = Classifier(classname='weka.classifiers.bayes.NaiveBayes')
        zero_r = Classifier(classname='weka.classifiers.rules.ZeroR')
        bayes_net = Classifier(classname='weka.classifiers.bayes.BayesNet',
                               options=['-D', '-Q', 'weka.classifiers.bayes.net.search.local.K2',
                                        '--', '-P', '1', '-S', 'BAYES', '-E',
                                        'weka.classifiers.bayes.net.estimate.SimpleEstimator',
                                        '--', '-A', '0.5'])
        d_tree = Classifier(classname='weka.classifiers.trees.J48',
                            options=['-C', '0.25', '-M', '2'])

        file_list = [
            'data/aggregated_data.csv'
        ]

        name_list = [
            'multi-class'
        ]

        # classify and export
        percent_range = range(1, 101, 1)
        zero_r_curves = multi_file_curve(classifier=zero_r, classifier_name='zero_r',
                                         name_list=name_list, in_file_list=file_list,
                                         percentages=percent_range)
        naive_bayes_curves = multi_file_curve(classifier=naive_bayes, classifier_name='naive_bayes',
                                              name_list=name_list, in_file_list=file_list,
                                              percentages=percent_range)
        bayes_net_curves = multi_file_curve(classifier=bayes_net, classifier_name='bayes_net',
                                            name_list=name_list, in_file_list=file_list,
                                            percentages=percent_range)
        d_tree_curves = multi_file_curve(classifier=d_tree, classifier_name='d_tree',
                                         name_list=name_list, in_file_list=file_list,
                                         percentages=percent_range)

        # export
        csv_header = [
            'approach',
            'classifier',
            'percentage_dataset_training',
            'accuracy',
            'f_measure'
        ]
        with open('analysis/learning_curves.csv', 'wb') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(csv_header)
            for r in zero_r_curves:
                csv_writer.writerow(r)
            for r in naive_bayes_curves:
                csv_writer.writerow(r)
            for r in bayes_net_curves:
                csv_writer.writerow(r)
            for r in d_tree_curves:
                csv_writer.writerow(r)

    except RuntimeError:
        typ, value, tb = sys.exc_info()
        print typ
        print value
        print tb

        traceback.print_exc()
        pdb.post_mortem(tb)
    finally:
        jvm.stop()

# Run code
if __name__ == '__main__':
    main()
