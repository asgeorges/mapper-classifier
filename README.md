
# mapper-classifier

## INTRODUCTION

This repository contains the initial PoC software to build a mapper based classifier.  The package here does not scale
well - however, work is currently being done to refactor this code into a polished package.  If you have questions about
this effort, please contact Jacek Cyranka: jcyranka@gmail.com

We have developed a robust classifier that is based on topological data analysis.  It’s a novel approach to
classification tasks that is both very general and powerful.  As a specific use case, its robustness makes it
particularly useful for tasks in computer vision.  Here, minor “defects” in the image can easily fool current state of
the art methods, whereas our algorithm far outperforms these.

This sw package allows the user to construct a robust mapper classifier in the task of computer vision for the MNIST
and Fashion-MNIST datasets.  The code refactoring will also make our sw consistent with current classes implemented in
scikit-learn.

If you would like to refer to the theoretical foundation and seminal paper, please refer to our paper.

The authors of the paper and software package are:
    Alex Georges: alexgrgs2314@gmail.com
    Jacek Cyranka: jcyranka@gmail.com
Feel free to contact the authors about any questions you may have in the sw, theory, or otherwise.

## DIRECTORY STRUCTURE

data_temp/: This is where we store some critical files that will be used later on in the workflow.  These can also be
useful debugging tools.

data_test_fashion_unnormalized/: Where raw test file(s) are kept for fashion.  You can change the naming here to however
 you want.  There are parameters in the scripts you will be running that allow for these name changes.

data_train_fashion_unnormalized/: Same as above.

images/: Where we output our images to.

lib/: Where all the heavy lifting scripts are located.  All the processing happens in files here.  Wrappers happen
in src/.  More on these scripts below.

results/: Where we output some critical files, including produced matrix mapper objects.

results_final/:  Where we output results from an entire adversary run.  See the workflow below.


## WORKFLOW
### There are various pieces of the workflow you can implement.  We'll go largest to smallest in terms of the pipeline.
### The first block will be what directory to navigate to, then what command(s) to run.  The second block of code will be the workflow pipeline that will be run by your command(s).

### In either workflow, you will have to change parameters, including directory naming schemes (sorry!) according to your analysis.  The defaults may not behave well for you.


1) If you want to run an entire adversary run (i.e. determine how robust matrix mapper classifiers are with respect to
varying amounts of noise), this is the workflow:
    cd src/
    ./doit.sh

        doit.sh -> runner.py -> pymapper.py
                             -> joiner.py
                             -> predictor.py
                             -> adversary.py

2) If you want to just create the matrix mapper objects (train and/or testing):
    cd src/
    python doit.py

        doit.py -> pymapper.py


## SCRIPTS


__src/doit.sh__: Bash wrapper script.  Params are input here then passed to runner.py.  Useful if you want to do many
adversary runs.

__src/runner.py__: Python wrapper script.  Combines all other python modules used to test robustness.

__src/doit.py__: Script that facilitates running other specific modules, rather than the entire workflow.  For instance, you
 can run just pymapper() from here.  Useful for testing specific piece of the workflow.

__lib/tools.py__: This is used to consolidate repeated functions used throughout the code and generally just help with
cleanup.

__lib/pymapper.py__: This code does all the heavy lifting - matrix mapper objects are constructed here.

__lib/mnist_mapper_light_template1D.r__:  The R implementation of the sw that produces a single mapper.

__lib/predictor.py__: Contains various end classifiers you can attach to the train committee and test committee of mapper
objects

__lib/adversary.py__: Iteratively perturbs data and records l2-norms when data points are misclassified through entire
workflow

__lib/plotter.py__: Plots final data of l2 norms so we can compare how the various methods perform

__lib/variational*.py__: Scripts constructed by the keras team that implement VAE

__lib/joiner.py__: This joins all the independent matrix mapper objects, which are in csv form.  In our analysis, this
function would merge 20 files for train and then 20 files for test.




