package com.sogoeslight.task.regression

import com.sogoeslight.task.preparation.Data.{attributes, learning, test}
import com.sogoeslight.task.regression.Utility._

import scala.collection.mutable.ArrayBuffer

object LogisticRegression extends App { // binomial in our case

  lazy val size: Int = attributes.length
  lazy val weights: ArrayBuffer[Double] = ArrayBuffer.fill[Double](size + 1)(0) // 4 inputs and bias

  def trainModel(ds: Array[Array[Double]]): Unit = { // searching for the best logistic coefficients
    var prediction: Double = 0 // initial value

    for (row <- ds) {
      prediction = sigmoid(-calculateOutput(row, weights)) // weighted sum
      val error: Double = row.last - prediction

      iterateWeightCoefs(weights, error, row, prediction, size)
    }

  }

  lazy val learningAccuracy: Double = testModel(learning, weights)
  lazy val testAccuracy: Double = testModel(test, weights)

  def calculate(): Unit = {
    trainModel(learning)
    printRegressionInfo("3 - Logistic Regression", weights, learningAccuracy, testAccuracy)
  }

  calculate()
}
