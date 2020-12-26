package com.sogoeslight.task.regression

import com.sogoeslight.task.regression.Utility.{calculateOutput, hyperbolicTangent, sigmoid}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

case class Neuron(attributesCount: Int, sigmoid: Boolean = false, hyperbolicTangent: Boolean = false) {
  val weights: ArrayBuffer[Double] = ArrayBuffer.fill[Double](attributesCount)(1 + Random.nextDouble() * -2)
  var weightsDif: ArrayBuffer[Double] = ArrayBuffer.fill(attributesCount)(0) // difference with prev iteration
  val gradients: ArrayBuffer[Double] = ArrayBuffer.fill(attributesCount)(0)
  var error: Double = 1
  var output: Double = 0
}

object Neuron {

  def addLayer(network: ArrayBuffer[ArrayBuffer[Neuron]], amountOfNeurons: Int, size: Int, bias: Int = 1,
               sigmoid: Boolean = false, hyperbolicTangent: Boolean = false): Unit = {
    network.addOne(new ArrayBuffer[Neuron](size))

    for (_ <- 0 until amountOfNeurons) {
      if (network.tail.nonEmpty) network.last.addOne(Neuron(network.head.length + bias, sigmoid, hyperbolicTangent))
      else network.last.addOne(Neuron(size, sigmoid, hyperbolicTangent)) // first layer receives all attributes as an input
    }
  }

  def calculateOutputsOfLayers(network: ArrayBuffer[ArrayBuffer[Neuron]], row: Array[Double], bias: Int = 1): Unit = {
    var output: Double = 0

    for ((layer, layerIndex) <- network.zipWithIndex) {
        if (layerIndex == 0) {
          for (neuron <- layer) {
            if (neuron.sigmoid) neuron.output = sigmoid(calculateOutput(row, neuron.weights))
            else if (neuron.hyperbolicTangent) neuron.output = hyperbolicTangent(calculateOutput(row, neuron.weights))
            else neuron.output = calculateOutput(row, neuron.weights)
          }
        } else {
          for (neuron <- layer) {
            output = 0

            for (index <- 0 until neuron.weights.length - bias) {
              output += network(layerIndex - 1)(index).output * neuron.weights(index)
            }

            if (bias == 1) output += neuron.weights(neuron.weights.length - 1)

            if (neuron.sigmoid) neuron.output = sigmoid(output)
            else if (neuron.hyperbolicTangent) neuron.output = hyperbolicTangent(output)
            else neuron.output = output
          }
        }
    }
  }
}