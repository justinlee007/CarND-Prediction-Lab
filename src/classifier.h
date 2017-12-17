#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using Eigen::ArrayXd;

class GNB {
 public:

  vector<string> possible_labels = {"left", "keep", "right"};

  ArrayXd left_means;
  ArrayXd left_sds;
  double left_prior;

  ArrayXd keep_means;
  ArrayXd keep_sds;
  double keep_prior;

  ArrayXd right_means;
  ArrayXd right_sds;
  double right_prior;

  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Trains the classifier with N data points and labels.
   *
   *      INPUTS
   *      data - array of N observations
   *     - Each observation is a tuple with 4 values: s, d,
   *       s_dot and d_dot.
   *     - Example : [
   *             [3.5, 0.1, 5.9, -0.02],
   *             [8.0, -0.3, 3.0, 2.2],
   *             ...
   *         ]
   *
   * @param data
   * @param labels array of N labels - Each label is one of "left", "keep", or "right".
   */
  void train(vector<vector<double>> data, vector<string> labels);

  /**
   * Once trained, this method is called and expected to return a predicted behavior for the given observation.
   *
   * @param sample observation - a 4 tuple with s, d, s_dot, d_dot.  Example: [3.5, 0.1, 8.5, -0.2]
   * @return
   */
  string predict(vector<double> sample);

};

#endif



