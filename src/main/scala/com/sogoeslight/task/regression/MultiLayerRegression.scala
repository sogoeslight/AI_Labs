package com.sogoeslight.task.regression

import com.sogoeslight.task.preparation.Data_Akbilgic
import com.sogoeslight.task.preparation.Data_Akbilgic.{convertedData, ds}
import com.sogoeslight.task.regression.Neuron.{addLayer, calculateOutputsOfLayers}
import com.sogoeslight.task.regression.Utility._

import scala.collection.mutable.ArrayBuffer

object MultiLayerRegression extends App {

  lazy val epoch: Int = 1000
  lazy val network = new ArrayBuffer[ArrayBuffer[Neuron]]()
  lazy val bias = 1
  lazy val size: Int = Data_Akbilgic.attributes.length

  def pearsonCorrelationCoef(): Unit = {
    val data: Array[Array[Double]] = convertedData.clone()
    val cols: Array[Array[Double]] = data.transpose
    val dsVariables: Array[String] = ds.head

    for (x <- dsVariables.indices) {
      for (y <- x + 1 until dsVariables.length) {
        val couples: Array[(Double, Double)] = cols(x) zip cols(y)
        val sum_X: Double = cols(x).sum
        val sum_Y: Double = cols(y).sum
        val coef: Double = (data.length * couples.map{ case (i, j) => i * j }.sum - sum_X * sum_Y) /
          math.sqrt((data.length * cols(x).map(math.pow(_, 2)).sum - math.pow(sum_X, 2)) *
            (data.length * cols(y).map(math.pow(_, 2)).sum) - math.pow(sum_Y, 2))
        println(s"    ${dsVariables(x)} - ${dsVariables(y)} correlation coefficient = $coef")
      }
      println()
    }

    println()
  }

  def trainModel(ds: Array[Array[Double]]): Unit = {
    network.addOne(new ArrayBuffer[Neuron](1)) // Activation layer
    network.last.addOne(Neuron(network.head.length + bias, sigmoid = true))

    for (_ <- 0 to epoch) {
      for (row <- ds) {
        calculateOutputsOfLayers(network, row)

        val ideal: Double = row.last // 0 or 1
        val actual: Double = network.last.head.output
        var fDerivative: Double = 1 - actual * actual // for tangh
        network.last.head.error = (ideal - actual) * fDerivative

        for (layerIndex <- network.tail.indices) { // calculate error
          for ((neuron, index) <- network(layerIndex).zipWithIndex) {
            var sum: Double = 0

            fDerivative = 1 - neuron.output * neuron.output
            for (nextLayer <- network(layerIndex + 1)) {
              sum += nextLayer.weights(index) * nextLayer.error
            }

            neuron.error = fDerivative * sum
          }
        }

        for (layerIndex <- network.indices) { // calculate weight's gradients
          for (neuron <- network(layerIndex)) {
            if (layerIndex != 0) {
              for (weight <- 0 until neuron.weights.length - bias) {
                val output: Double = network(layerIndex - 1)(weight).output
                neuron.gradients(weight) = output * neuron.error
              }
              if (bias == 1) {
                val output: Double = neuron.weights(neuron.weights.length - bias)
                neuron.gradients(neuron.weights.length - bias) = output * neuron.error
              }
            } else {
              for (index <- Data_Akbilgic.attributes.indices) {
                neuron.gradients(index) = row(index) * neuron.error
              }

              neuron.gradients(neuron.gradients.size - 1) = neuron.weights.last * neuron.error
            }
          }
        }

        for ((layer, layerIndex) <- network.zipWithIndex) { // recalculate weights
          for ((neuron, neuronIndex) <- layer.zipWithIndex) {
            for (index <- neuron.weights.indices) { // learning rate * (weight's gradient + previous weight's dif)
              var weightsDif: Double = 0
              if (layerIndex == 0) weightsDif = rate * neuron.gradients(index) // rate = epsilon, first layer weights difference is 0, so we do not include it in the formula
              else weightsDif = rate * (neuron.gradients(index) + network(layerIndex - 1)(neuronIndex).weightsDif(index))
              neuron.weightsDif(index) = weightsDif
              neuron.weights(index) += weightsDif
            }
          }
        }

      }
    }
  }

  lazy val learningAccuracy: Double = testMultiLayerRegressionModel(Data_Akbilgic.learning, network)
  lazy val testAccuracy: Double = testMultiLayerRegressionModel(Data_Akbilgic.test, network)

  def calculate(showWeights: Boolean): Unit = {
    println("Pearson: ")
    pearsonCorrelationCoef()
    addLayer(network, amountOfNeurons = 3, size, hyperbolicTangent = true)
    addLayer(network, amountOfNeurons = 3, size, hyperbolicTangent = true)
    trainModel(Data_Akbilgic.learning)
    printMultiLayerPerceptroneInfo(labAndMethod = "6 - Multi Layer Perceptrone",
        network, learning = learningAccuracy, test = testAccuracy, weights = showWeights)
  }

  //calculate(showWeights = true)
}
