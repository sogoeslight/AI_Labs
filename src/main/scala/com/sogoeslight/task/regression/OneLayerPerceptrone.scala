package com.sogoeslight.task.regression

import com.sogoeslight.task.preparation.Data.{learning, test, attributes}
import Utility._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object OneLayerPerceptrone extends App {

  lazy val size: Int = attributes.length
  lazy val weights: ArrayBuffer[Double] = ArrayBuffer.fill[Double](size + 1)(Random.nextDouble()) // initialize weight with random [0, 1] numbers
  lazy val epoch: Int = 3000 // 30 000 - best

  def trainModel(ds: Array[Array[Double]], isTrace: Boolean): Unit = { // searching for the best logistic coefficients
    var prediction: Double = 0 // initial value

    for (index <- 0 to epoch) {
      var sum_error: Double = 0

      for (row <- ds) {
        prediction = sigmoid(-calculateOutput(row, weights)) // weighted sum
        val error: Double = row.last - prediction
        sum_error += error * error

        iterateWeightCoefs(weights, error, row, prediction, size)
      }

      if (isTrace) println(s"Epoch #$index, current error -> ${sum_error / ds.length}")
    }
  }

  lazy val learningAccuracy: Double = testModel(learning, weights)
  lazy val testAccuracy: Double = testModel(test, weights)

  def calculate(trace: Boolean): Unit = {
    trainModel(learning, trace) // set second argument as True to display "max_iter" - error on epoch
    printRegressionInfo("4 - One Layer Perceptrone", weights, learningAccuracy, testAccuracy)
  }

  calculate(false)
}
