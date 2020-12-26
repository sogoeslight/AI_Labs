package com.sogoeslight.task.regression

import com.sogoeslight.task.preparation.Data.attributes
import com.sogoeslight.task.regression.Neuron.calculateOutputsOfLayers

import scala.collection.mutable.ArrayBuffer

object Utility {
  val rate: Double = 0.2 // learning rate (usually -> [0.1, 0.3]) - regulates how much coefs changes each iteration

  def iterateWeightCoefs(weights: ArrayBuffer[Double], error: Double, row: Array[Double], prediction: Double, size: Int): Unit = {
    for (i <- attributes.indices) { // new weight coefficients
      weights(i) += rate * error * prediction * (1 - prediction) * row(i)
    }
    weights(size) += rate * error * prediction * (1 - prediction)
  }

  def calculateOutput(row: Array[Double], weights: ArrayBuffer[Double]): Double = {
    var output: Double = 0

    for (index <- attributes.indices) {
      output += row(index) * weights(index)
    }

    output += weights.last // free term
    output
  }

  // log functions
  def sigmoid(output: Double): Double = 1 / (1 + math.exp(-output))
  def hyperbolicTangent(output: Double, bias: Int = 1): Double = 2 / (1 + math.exp(-2 * output)) - bias

  def testModel(ds: Array[Array[Double]], weights: ArrayBuffer[Double]): Double = {
    var accuracy: Double = 0

    for (row <- ds) {
      val prediction: Double = sigmoid(calculateOutput(row, weights))

      if (prediction.round == row.last) accuracy += 1
    }

    accuracy / ds.length * 100 // in %s
  }

  def testMultiLayerClassificationModel(ds: Array[Array[Double]], network: ArrayBuffer[ArrayBuffer[Neuron]]): Double = {
    var accuracy: Double = 0

    for (row <- ds) {
      calculateOutputsOfLayers(network, row)

      val prediction: Double = network.last(0).output

      if (prediction.round == row.last)
        accuracy += 1
    }

    accuracy / ds.length * 100 // in %s
  }

  def testMultiLayerRegressionModel(ds: Array[Array[Double]], network: ArrayBuffer[ArrayBuffer[Neuron]]): Double = {
    var accuracy: Double = 0

    val avg: Double = ds.transpose.last.sum / ds.length
    val disp: Double = math.sqrt(ds.transpose.last.map(elem => math.pow(elem - avg, 2)).sum / ds.length)

    for (row <- ds) {
      calculateOutputsOfLayers(network, row)

      val prediction: Double = network.last(0).output

      if (math.pow(prediction - row.last, 2) >= disp)
        accuracy += 1
    }

    accuracy / ds.length * 100 // in %s
  }

  def printWeights(network: ArrayBuffer[ArrayBuffer[Neuron]]): Unit = {
    println("- Model:")
    network.indices
      .foreach(layerIndex => network(layerIndex).indices
        .foreach(neuronIndex => {
          network(layerIndex)(neuronIndex).weights.dropRight(1).indices
            .foreach(weightIndex => println(s"    Coef (Layer[$layerIndex] Neuron[$neuronIndex] Weight[$weightIndex])" +
              s" = ${network(layerIndex)(neuronIndex).weights(weightIndex)}"))
          println(s"                              Free term = ${network(layerIndex)(neuronIndex).weights.last}\n")
        }))
  }

  def printMultiLayerPerceptroneInfo(labAndMethod: String, network: ArrayBuffer[ArrayBuffer[Neuron]], learning: Double, test: Double, weights: Boolean): Unit = {
    println(s"\nLAB $labAndMethod\n")
    if (weights) printWeights(network)
    println(f"- Learning samples accuracy: $learning%.2f%%")
    println(f"- Test samples accuracy: $test%.2f%%")
    println("\n____________________________________________________________________________________")
  }

  def printRegressionInfo(labAndMethod: String, weights: ArrayBuffer[Double], learning: Double, test: Double): Unit = {
    println(s"\nLAB $labAndMethod\n")
    println("- Model:")
    for ((coef, index) <- weights.dropRight(1).zipWithIndex) println(s"    Coef $index = $coef")
    println("    Free term = " + weights.last)

    println(f"\n- Learning samples accuracy: $learning%.2f%%")
    println(f"- Test samples accuracy: $test%.2f%%")
    println("\n____________________________________________________________________________________")
  }
}
