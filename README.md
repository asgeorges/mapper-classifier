# mapper-classifier
This repository contains the initial PoC software to build a mapper based classifier.  The package here does not scale well - however, work is currently being done to refactor this code into a polished package.  If you have questions about this effort, please contact Jacek Cyranka: jcyranka@gmail.com  

We have developed a robust classifier that is based on topological data analysis.  It’s a novel approach to classification tasks that is both very general and powerful.  As a specific use case, its robustness makes it particularly useful for tasks in computer vision.  Here, minor “defects” in the image can easily fool current state of the art methods, whereas our algorithm far outperforms these.  

This sw package allows the user to construct a robust mapper classifier in the task of computer vision for the MNIST and Fashion-MNIST datasets.  The code refactoring will also make our sw consistent with current classes implemented in scikit-learn.  

If you would like to refer to the theoretical foundation and seminal paper, please refer to:  The authors of the paper and software package are Jacek Cyranka & Alex Georges. 
